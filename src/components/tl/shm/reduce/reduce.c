/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_shm.h"
#include "reduce.h"

enum
{
    REDUCE_STAGE_START,
    REDUCE_STAGE_BASE_TREE,
    REDUCE_STAGE_TOP_TREE,
};

static inline ucc_status_t ucc_tl_shm_dt_reduce(void *dst, void **srcs, int n_srcs,
                                                size_t count, ucc_datatype_t dt,
                                                ucc_reduction_op_t op, ucc_ee_executor_t *exec)
{
    ucc_ee_executor_task_t     *etask;
    ucc_status_t                status;
    ucc_ee_executor_task_args_t eargs;

    eargs.flags           = UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT;
    eargs.task_type       = UCC_EE_EXECUTOR_TASK_REDUCE;
    eargs.reduce.dst      = dst;
    eargs.reduce.srcs_ext = srcs;
    eargs.reduce.count    = count;
    eargs.reduce.dt       = dt;
    eargs.reduce.op       = op;
    eargs.reduce.n_srcs   = n_srcs;

    status = ucc_ee_executor_task_post(exec, &eargs, &etask);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    if (etask) {
        do {
            status = ucc_ee_executor_task_test(etask);
        } while (status > 0);
        ucc_ee_executor_task_finalize(etask);
    }
    return status;
}


ucc_status_t
ucc_tl_shm_reduce_read(ucc_tl_shm_team_t *team, ucc_tl_shm_seg_t *seg,
                       ucc_tl_shm_task_t *task, ucc_kn_tree_t *tree,
                       int is_inline, size_t count, ucc_datatype_t dt,
                       ucc_coll_args_t *args)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_sn_t    seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_rank_t         root      = (ucc_rank_t)args->root;
    ucc_rank_t         radix     = tree->radix;
    void              *dst;
    ucc_tl_shm_ctrl_t *child_ctrl, *my_ctrl;
    ucc_rank_t         child;
    int                i, j, batch, ready, num_ready;
    ucc_status_t       status;
    void              *srcs[radix];

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->n_children == 0) {
        /* I am leaf so I dont need to read, only notify parent*/
        if (tree == task->tree->base_tree || task->tree->base_tree == NULL) {
            /* I am leaf in base tree so need to copy from user buffer into my shm */
            dst = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            memcpy(dst, args->src.info.buffer, count * ucc_dt_size(dt));
            ucc_memory_cpu_store_fence();
        }
        my_ctrl->pi2 = seq_num; //signals to parent
        return UCC_OK;
    }

    num_ready = 0;
    for (i = task->cur_child; i < tree->n_children; i++) {
        batch      = ucc_min(radix - 1, tree->n_children - task->cur_child);
        child      = tree->children[i];
        child_ctrl = ucc_tl_shm_get_ctrl(seg, team, child);
        ready      = 0;
        for (j = 0; j < n_polls; j++) {
            if (child_ctrl->pi2 == seq_num) {
                ucc_memory_cpu_fence();
                ready = 1;
                num_ready++;
                srcs[num_ready] = is_inline ? child_ctrl->data
                    : ucc_tl_shm_get_data(seg, team, child);
                break;
            }
        }
        if (!ready) {
            return UCC_INPROGRESS;
        }
        if (num_ready == batch) {
            dst = ((root == team_rank)
                   ? args->dst.info.buffer : (is_inline ? my_ctrl->data
                      : ucc_tl_shm_get_data(seg, team, team_rank)));

            srcs[0] = (task->first_reduce ? ((UCC_IS_INPLACE(*args) &&
                                              args->root == team_rank)
                                             ? args->dst.info.buffer
                                             : args->src.info.buffer) : dst);
            task->first_reduce = 0;
            status             = ucc_tl_shm_dt_reduce(dst, srcs, batch + 1,
                                                      count, dt, args->op,
                                                      task->executor);
            if (ucc_unlikely(UCC_OK != status)) {
                task->super.super.status = status;
                return status;
            }

            task->cur_child += batch;
            num_ready = 0;
        }
    }
    ucc_memory_cpu_store_fence();
    if (tree->parent != UCC_RANK_INVALID) {
        my_ctrl->pi2 = seq_num; //signals to parent
    }
    return UCC_OK;
}

static void ucc_tl_shm_reduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         root = (ucc_rank_t)args.root;
    ucc_tl_shm_seg_t * seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_op_root = rank == root;
    int                is_inline;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    ucc_tl_shm_ctrl_t *my_ctrl;

    if (is_op_root) {
        count = args.dst.info.count;
        dt    = args.dst.info.datatype;
    } else {
        count = args.src.info.count;
        dt    = args.src.info.datatype;
    }
    data_size = count * ucc_dt_size(dt);
    is_inline = data_size <= team->max_inline;

next_stage:
    switch (task->stage) {
    case REDUCE_STAGE_START:
        /* checks if previous collective has completed on the seg */
        SHMCHECK_GOTO(ucc_tl_shm_check_seg_ready(task, tree, 1), task, out);
        if (tree->base_tree) {
            task->stage = REDUCE_STAGE_BASE_TREE;
        } else {
            task->stage = REDUCE_STAGE_TOP_TREE;
        }
        goto next_stage;
    case REDUCE_STAGE_BASE_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->base_tree,
                      is_inline, count, dt, &args), task, out);
        task->cur_child = 0;
        if (tree->top_tree) {
            task->stage = REDUCE_STAGE_TOP_TREE;
            goto next_stage;
        }
        break;
    case REDUCE_STAGE_TOP_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->top_tree,
                      is_inline, count, dt, &args), task, out);
        break;
    }

    my_ctrl     = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;

    /* reduce done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_rr_done", 0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_reduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    task->stage = REDUCE_STAGE_START;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_start", 0);

    status = ucc_coll_task_get_executor(coll_task, &task->executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_reduce_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *     tl_team,
                                    ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t     *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_tl_shm_task_t     *task;
    ucc_tl_shm_pp_reduce_t params;
    ucc_status_t           status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_PERSISTENT(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_shm_get_task(coll_args, team);
    ucc_tl_shm_task_reset(task, team, TASK_ARGS(task).root);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    team->perf_params_reduce(&params.super, task);

    status = ucc_tl_shm_tree_init(
        team, coll_args->args.root, params.super.base_radix,
        params.super.top_radix, UCC_COLL_TYPE_REDUCE,
        params.super.base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }

    task->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_shm_reduce_start;
    task->super.progress = ucc_tl_shm_reduce_progress;
    *task_h = &task->super;
    return UCC_OK;
}
