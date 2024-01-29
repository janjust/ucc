/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "../tl_shm.h"
#include "bcast.h"
#include "utils/arch/cpu.h"

enum
{
    BCAST_STAGE_START,
    BCAST_STAGE_BASE_TREE,
    BCAST_STAGE_TOP_TREE,
    BCAST_STAGE_COPY_OUT,
    BCAST_STAGE_READ_CHECK,
};

ucc_status_t ucc_tl_shm_bcast_write(ucc_tl_shm_team_t *team,
                                    ucc_tl_shm_seg_t * seg,
                                    ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    size_t data_size)
{
    ucc_rank_t      team_rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_sn_t seq_num   = task->seq_num;
    uint32_t        n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    int is_op_root            = (team_rank == (ucc_rank_t)TASK_ARGS(task).root);
    ucc_tl_shm_ctrl_t *my_ctrl;
    void *             src;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->parent == UCC_RANK_INVALID) {
        /* i am root of the tree*/
        /* If the tree root is global OP root he can copy data out from
           origin user src buffer.
           Otherwise, it must be base_tree in 2lvl alg,
           and the data of the tree root is in the local shm (ctrl or data) */
        src = is_op_root
                  ? TASK_ARGS(task).src.info.buffer
                  : (is_inline ? my_ctrl->data
                               : ucc_tl_shm_get_data(seg, team, team_rank));
        ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline, src,
                                    data_size);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            src = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline,
                                        src, data_size);
            task->data_rank = team_rank;
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_shm_bcast_read(ucc_tl_shm_team_t *team,
                                   ucc_tl_shm_seg_t * seg,
                                   ucc_tl_shm_task_t *task, ucc_kn_tree_t *tree,
                                   int is_inline, int *is_op_root,
                                   size_t data_size)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_sn_t    seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void *             src, *dst;
    ucc_tl_shm_ctrl_t *parent_ctrl, *my_ctrl;
    ucc_rank_t         parent;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);
    if (*is_op_root) {
        /* Only global op root needs to copy the data from user src to its shm */
        if (*is_op_root == 1) {
            dst = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            memcpy(dst, TASK_ARGS(task).src.info.buffer, data_size);
            ucc_memory_cpu_store_fence();
            (*is_op_root)++;
        }
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }
    parent = tree->parent;
    if (parent == UCC_RANK_INVALID) {
        /* I'm the root of the tree and NOT is_op_root. It means the tree is
           base tree and i already have the data in my shm via top_tree step
           (read or write). Just notify children. */
        ucc_assert(my_ctrl->pi == seq_num);
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            task->data_rank = parent;
            parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (tree == task->tree->top_tree || tree->n_children > 0) {
                src = is_inline ? parent_ctrl->data
                                : ucc_tl_shm_get_data(seg, team, parent);
                dst = is_inline ? my_ctrl->data
                                : ucc_tl_shm_get_data(seg, team, team_rank);
                memcpy(dst, src, data_size);
                ucc_memory_cpu_store_fence();
                my_ctrl->rr     = seq_num;
                task->data_rank = team_rank;
                ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
            }
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

void ucc_tl_shm_bcast_copy_out(ucc_tl_shm_task_t *task, size_t data_size)
{
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         root      = (ucc_rank_t)args.root;
    ucc_rank_t         rank      = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t * seg       = task->seg;
    int                is_inline = data_size <= team->max_inline;
    ucc_tl_shm_ctrl_t *my_ctrl, *data_ctrl;
    void *             src;

    if (rank != root) {
        my_ctrl   = ucc_tl_shm_get_ctrl(seg, team, rank);
        data_ctrl = ucc_tl_shm_get_ctrl(seg, team, task->data_rank);
        src       = is_inline ? data_ctrl->data
                              : ucc_tl_shm_get_data(seg, team, task->data_rank);
        memcpy(args.src.info.buffer, src, data_size);
        ucc_memory_cpu_store_fence();
        my_ctrl->rr = task->seq_num;
    }
}

ucc_status_t ucc_tl_shm_bcast_check_read_ready(ucc_tl_shm_task_t *task)
{
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_tl_shm_seg_t * seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    ucc_tl_shm_ctrl_t *ctrl;
    int                i;

    if (tree->top_tree && tree->top_tree->n_children > 0 &&
        (task->progress_alg == BCAST_RW || task->progress_alg == BCAST_RR)) {
        for (i = 0; i < tree->top_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->top_tree->children[i]);
            if (ctrl->rr < task->seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    if (tree->base_tree && tree->base_tree->n_children > 0 &&
        (task->progress_alg == BCAST_WR || task->progress_alg == BCAST_RR)) {
        for (i = task->cur_child; i < tree->base_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->base_tree->children[i]);
            if (ctrl->rr < task->seq_num) {
                task->cur_child = i;
                return UCC_INPROGRESS;
            }
        }
    }
    return UCC_OK;
}

static void ucc_tl_shm_bcast_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size =
        args.src.info.count * ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t)args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t * seg        = task->seg;
    ucc_tl_shm_tree_t *tree       = task->tree;
    int                is_inline  = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_tl_shm_ctrl_t *my_ctrl;

next_stage:
    switch (task->stage) {
    case BCAST_STAGE_START:
        if ((tree->base_tree && tree->base_tree->n_children > 0) ||
            (tree->base_tree == NULL && tree->top_tree->n_children > 0)) {
            /* checks if previous collective has completed on the seg
                TODO: can be optimized if we detect bcast->reduce pattern.*/
            SHMCHECK_GOTO(ucc_tl_shm_check_seg_ready(task, tree, 0), task, out);
        }
        if (tree->top_tree) {
            task->stage = BCAST_STAGE_TOP_TREE;
        } else {
            task->stage = BCAST_STAGE_BASE_TREE;
        }
        goto next_stage;
    case BCAST_STAGE_TOP_TREE:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_WR) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task,
                                                 tree->top_tree, is_inline,
                                                 data_size),
                          task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        if (tree->base_tree) {
            task->stage = BCAST_STAGE_BASE_TREE;
        } else {
            task->stage = BCAST_STAGE_COPY_OUT;
        }
        goto next_stage;
    case BCAST_STAGE_BASE_TREE:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_RW) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task,
                                                 tree->base_tree, is_inline,
                                                 data_size),
                          task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        /* fall through */
    case BCAST_STAGE_COPY_OUT:
        ucc_tl_shm_bcast_copy_out(task, data_size);
        task->cur_child = 0;
        task->stage = BCAST_STAGE_READ_CHECK;
        /* fall through */
    case BCAST_STAGE_READ_CHECK:
        SHMCHECK_GOTO(ucc_tl_shm_bcast_check_read_ready(task), task, out);
        break;
    }

    my_ctrl     = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_rw_progress_done",
                                     0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    task->stage = BCAST_STAGE_START;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_start", 0);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_bcast_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     tl_team,
                                   ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t    *team   = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_tl_shm_pp_bcast_t params = {.super        = {0},
                                    .progress_alg = BCAST_WW};
    ucc_tl_shm_task_t    *task;
    ucc_status_t          status;

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_PERSISTENT(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_shm_get_task(coll_args, team);
    ucc_tl_shm_task_reset(task, team, UCC_RANK_INVALID);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    team->perf_params_bcast(&params.super, task);
    task->progress_alg = params.progress_alg;

    // coverity[divide_by_zero:FALSE]
    status = ucc_tl_shm_tree_init(
        team, coll_args->args.root, params.super.base_radix,
        params.super.top_radix, UCC_COLL_TYPE_BCAST,
        params.super.base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }

    // coverity[uninit_use:FALSE]
    task->super.post     = ucc_tl_shm_bcast_start;
    task->super.progress = ucc_tl_shm_bcast_progress;

    *task_h = &task->super;
    return UCC_OK;
}
