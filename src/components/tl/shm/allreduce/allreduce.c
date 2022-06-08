/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "allreduce.h"

enum
{
    ALLREDUCE_STAGE_START,
    ALLREDUCE_STAGE_BASE_TREE_REDUCE,
    ALLREDUCE_STAGE_TOP_TREE_REDUCE,
    ALLREDUCE_STAGE_BASE_TREE_BCAST,
    ALLREDUCE_STAGE_TOP_TREE_BCAST,
    ALLREDUCE_STAGE_BCAST_COPY_OUT,
    ALLREDUCE_STAGE_BCAST_READ_CHECK,
};

static void ucc_tl_shm_allreduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task        = ucc_derived_of(coll_task,
                                                    ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team        = TASK_TEAM(task);
    ucc_coll_args_t   *args        = &TASK_ARGS(task);
    ucc_rank_t         rank        = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         root        = (ucc_rank_t)args->root;
    ucc_tl_shm_seg_t  *seg         = task->seg;
    ucc_tl_shm_tree_t *bcast_tree  = task->allreduce.bcast_tree;
    ucc_tl_shm_tree_t *reduce_tree = task->allreduce.reduce_tree;
    ucc_memory_type_t  mtype       = args->dst.info.mem_type;
    ucc_datatype_t     dt          = args->dst.info.datatype;
    size_t             count       = args->dst.info.count;
    size_t             data_size   = count * ucc_dt_size(dt);
    int                is_inline   = data_size <= team->max_inline;
    int                is_op_root  = rank == root;
    ucc_tl_shm_ctrl_t *my_ctrl;

next_stage:
    switch (task->stage) {
    case ALLREDUCE_STAGE_START:
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        SHMCHECK_GOTO(ucc_tl_shm_reduce_seg_ready(seg, task->seg_ready_seq_num,
                                                  team, reduce_tree), task,
                                                                      out);
        if (reduce_tree->base_tree) {
            task->stage = ALLREDUCE_STAGE_BASE_TREE_REDUCE;
        } else {
            task->stage = ALLREDUCE_STAGE_TOP_TREE_REDUCE;
        }
        if (UCC_IS_INPLACE(*args) && !is_op_root) {
            args->src.info.buffer = args->dst.info.buffer;
        }
        goto next_stage;
    case ALLREDUCE_STAGE_BASE_TREE_REDUCE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, reduce_tree->base_tree,
                      is_inline, count, dt, mtype, args), task, out);
        task->cur_child = 0;
        if (reduce_tree->top_tree) {
            task->stage = ALLREDUCE_STAGE_TOP_TREE_REDUCE;
        } else {
            args->src.info.buffer = args->dst.info.buffer; // needed to fit bcast api
            task->stage = bcast_tree->top_tree ?
                          ALLREDUCE_STAGE_TOP_TREE_BCAST :
                          ALLREDUCE_STAGE_BASE_TREE_BCAST;
            task->tree = bcast_tree;
            task->seq_num++; /* finished reduce, need seq_num to be updated for bcast */
        }
        my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
        goto next_stage;
    case ALLREDUCE_STAGE_TOP_TREE_REDUCE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, reduce_tree->top_tree,
                      is_inline, count, dt, mtype, args), task, out);
        args->src.info.buffer = args->dst.info.buffer; // needed to fit bcast api
        task->stage = bcast_tree->top_tree ? ALLREDUCE_STAGE_TOP_TREE_BCAST :
                                             ALLREDUCE_STAGE_BASE_TREE_BCAST;
        task->tree = bcast_tree;
        task->seq_num++; /* finished reduce, need seq_num to be updated for bcast */
        goto next_stage;
    case ALLREDUCE_STAGE_TOP_TREE_BCAST:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_WR) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, bcast_tree->top_tree,
                          is_inline, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, bcast_tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        if (bcast_tree->base_tree) {
            task->stage = ALLREDUCE_STAGE_BASE_TREE_BCAST;
        } else {
            task->stage = ALLREDUCE_STAGE_BCAST_COPY_OUT;
        }
        goto next_stage;
    case ALLREDUCE_STAGE_BASE_TREE_BCAST:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_RW) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, bcast_tree->base_tree,
                          is_inline, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, bcast_tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        task->stage = ALLREDUCE_STAGE_BCAST_COPY_OUT;
        goto next_stage;
    case ALLREDUCE_STAGE_BCAST_COPY_OUT:
        ucc_tl_shm_bcast_copy_out(task);
        task->stage = ALLREDUCE_STAGE_BCAST_READ_CHECK;
    case ALLREDUCE_STAGE_BCAST_READ_CHECK:
        SHMCHECK_GOTO(ucc_tl_shm_bcast_check_read_ready(task), task, out);
        break;
    }

    /* task->seq_num was updated between reduce and bcast, needs to be reset
       to fit general collectives order, as allreduce is a single collective */
    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num - 1;
    /* allreduce done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task,
                                     "shm_allreduce_progress_done", 0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_allreduce_start", 0);
    UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(task, team);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_allreduce_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     tl_team,
                                     ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_tl_shm_task_t *task;
    ucc_status_t       status;

    if (UCC_IS_PERSISTENT(coll_args->args) ||
        coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_shm_get_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    TASK_ARGS(task).root = 0;
    team->perf_params_bcast(&task->super);
    /* values from team->perf_params_bcast(&task->super) */
    task->allreduce.bcast_base_radix      = task->base_radix;
    task->allreduce.bcast_top_radix       = task->top_radix;
    task->allreduce.bcast_base_tree_only  = task->base_tree_only;

    team->perf_params_reduce(&task->super);
    /* values from team->perf_params_reduce(&task->super) */
    task->allreduce.reduce_base_radix     = task->base_radix;
    task->allreduce.reduce_top_radix      = task->top_radix;
    task->allreduce.reduce_base_tree_only = task->base_tree_only;

    task->super.post     = ucc_tl_shm_allreduce_start;
    task->super.progress = ucc_tl_shm_allreduce_progress;
    task->stage          = ALLREDUCE_STAGE_START;

    status = ucc_tl_shm_tree_init(team, TASK_ARGS(task).root,
                                  task->allreduce.bcast_base_radix,
                                  task->allreduce.bcast_top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_BCAST,
                                  task->allreduce.bcast_base_tree_only,
                                  &task->allreduce.bcast_tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm bcast tree");
        return status;
    }

    status = ucc_tl_shm_tree_init(team, TASK_ARGS(task).root,
                                  task->allreduce.reduce_base_radix,
                                  task->allreduce.reduce_top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_REDUCE,
                                  task->allreduce.reduce_base_tree_only,
                                  &task->allreduce.reduce_tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm reduce tree");
        return status;
    }

    task->tree = task->allreduce.reduce_tree;
    *task_h = &task->super;
    return UCC_OK;
}
