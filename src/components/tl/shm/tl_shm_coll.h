/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_COLL_H_
#define UCC_TL_SHM_COLL_H_

#include "tl_shm.h"

typedef struct ucc_tl_shm_task {
    ucc_coll_task_t                 super;
    ucc_tl_shm_seg_t *              seg;
    ucc_tl_shm_tree_t *             tree;
    ucc_tl_shm_sn_t                 seq_num;
    ucc_tl_shm_last_posted_t        prev;
    int                             stage;
    int                             first_reduce;
    ucc_tl_shm_bcast_progress_alg_t progress_alg;
    ucc_rank_t                      data_rank;
    ucc_rank_t                      cur_child;
    void *                          src_buf;
    ucc_ee_executor_t              *executor;
    struct {
        ucc_tl_shm_tree_t *bcast_tree;
        ucc_tl_shm_tree_t *reduce_tree;
    } allreduce;
} ucc_tl_shm_task_t;

ucc_status_t ucc_tl_shm_coll_finalize(ucc_coll_task_t *coll_task);

static inline void ucc_tl_shm_task_reset(ucc_tl_shm_task_t *task,
                                         ucc_tl_shm_team_t *team,
                                         ucc_rank_t         root)
{
    int seg_id;

    task->first_reduce = 1;
    task->cur_child    = 0;
    task->seq_num      = team->seq_num++;
    seg_id             = task->seq_num % team->n_concurrent;
    task->seg          = &team->segs[seg_id * team->n_base_groups];
    task->prev         = team->last_posted[seg_id];
    team->last_posted[seg_id].seq_num     = task->seq_num;
    team->last_posted[seg_id].reduce_root = root;
}

static inline ucc_tl_shm_task_t *
ucc_tl_shm_get_task(ucc_base_coll_args_t *coll_args, ucc_tl_shm_team_t *team)
{
    ucc_tl_shm_context_t *ctx =
        ucc_derived_of(team->super.super.context, ucc_tl_shm_context_t);
    ucc_tl_shm_task_t *task = ucc_mpool_get(&ctx->req_mp);

    if (ucc_unlikely(!task)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate task");
        return NULL;
    }

    UCC_TL_SHM_PROFILE_REQUEST_NEW(task, "tl_shm_task", 0);
    ucc_coll_task_init(&task->super, coll_args, &team->super.super);
    task->super.finalize = ucc_tl_shm_coll_finalize;
    task->super.triggered_post = ucc_triggered_post;
    return task;
}

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task);

ucc_status_t ucc_tl_shm_tree_init(ucc_tl_shm_team_t *team, ucc_rank_t root,
                                  ucc_rank_t base_radix, ucc_rank_t top_radix,
                                  ucc_coll_type_t coll_type, int base_tree_only,
                                  ucc_tl_shm_tree_t **tree_p);

void ucc_tl_shm_tree_cleanup(ucc_tl_shm_tree_t *tree);

static inline ucc_tl_shm_ctrl_t *
ucc_tl_shm_get_ctrl(ucc_tl_shm_seg_t *seg, ucc_tl_shm_team_t *team,
                    ucc_rank_t rank /* rank within a TL team */)
{
    int        group     = ucc_ep_map_eval(team->rank_group_id_map, rank);
    ucc_rank_t grank     = ucc_ep_map_eval(team->group_rank_map, rank);
    size_t     ctrl_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;

    return PTR_OFFSET(seg[group].ctrl, ctrl_size * grank);
}

static inline void *
ucc_tl_shm_get_data(ucc_tl_shm_seg_t *seg, ucc_tl_shm_team_t *team,
                    ucc_rank_t rank) /* rank withing a TL team */
{
    int        group     = ucc_ep_map_eval(team->rank_group_id_map, rank);
    ucc_rank_t grank     = ucc_ep_map_eval(team->group_rank_map, rank);
    size_t     data_size = team->arch_data_size;

    return PTR_OFFSET(seg[group].data, data_size * grank);
}

#define SHMCHECK_GOTO(_cmd, _task, _label)                                     \
    do {                                                                       \
        ucc_status_t _status = (_cmd);                                         \
        if (UCC_OK != _status) {                                               \
            _task->super.status = _status;                                     \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

static inline ucc_status_t ucc_tl_shm_bcast_seg_ready(ucc_tl_shm_seg_t *seg,
                                                      ucc_tl_shm_sn_t   seq_num,
                                                      ucc_tl_shm_team_t *team,
                                                      ucc_tl_shm_tree_t *tree)
{
    ucc_tl_shm_ctrl_t *ctrl;
    int                i;

    ctrl = ucc_tl_shm_get_ctrl(seg, team, UCC_TL_TEAM_RANK(team));
    if (ctrl->ci != seq_num) {
        return UCC_INPROGRESS;
    }

    if (tree->top_tree) {
        for (i = 0; i < tree->top_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->top_tree->children[i]);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    if (tree->base_tree) {
        for (i = 0; i < tree->base_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->base_tree->children[i]);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_shm_reduce_seg_ready(ucc_tl_shm_seg_t *seg,
                                                       ucc_tl_shm_sn_t seq_num,
                                                       ucc_tl_shm_team_t *team,
                                                       ucc_tl_shm_tree_t *tree)
{

    ucc_tl_shm_ctrl_t *ctrl;
    ucc_rank_t         parent;

    ctrl = ucc_tl_shm_get_ctrl(seg, team, UCC_TL_TEAM_RANK(team));
    if (ctrl->ci != seq_num) {
        return UCC_INPROGRESS;
    }

    if (tree->base_tree) {
        parent = tree->base_tree->parent;
        if (parent != UCC_RANK_INVALID) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    if (tree->top_tree) {
        parent = tree->top_tree->parent;
        if (parent != UCC_RANK_INVALID) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_shm_check_seg_ready(ucc_tl_shm_task_t *task,
                                                      ucc_tl_shm_tree_t *tree,
                                                      int                is_reduce)
{
    ucc_tl_shm_team_t *team    = TASK_TEAM(task);
    ucc_tl_shm_seg_t  *seg     = task->seg;
    ucc_tl_shm_sn_t    seq_num = task->prev.seq_num;
    ucc_tl_shm_ctrl_t *ctrl;

    if (task->prev.reduce_root != UCC_RANK_INVALID) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, task->prev.reduce_root);
        if (ctrl->ci < seq_num) {
            return UCC_INPROGRESS;
        }
        return UCC_OK;
    } else {
        if (is_reduce) {
            return ucc_tl_shm_reduce_seg_ready(seg, seq_num, team, tree);
        } else {
            return ucc_tl_shm_bcast_seg_ready(seg, seq_num, team, tree);
        }
    }
}


static inline void
ucc_tl_shm_copy_to_children(ucc_tl_shm_seg_t *seg, ucc_tl_shm_team_t *team,
                            ucc_kn_tree_t *tree, ucc_tl_shm_sn_t seq_num,
                            int is_inline, void *src, size_t data_size)
{
    ucc_tl_shm_ctrl_t *ctrl;
    void *             dst;
    int                i;

    for (i = 0; i < tree->n_children; i++) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        dst  = is_inline ? ctrl->data
                         : ucc_tl_shm_get_data(seg, team, tree->children[i]);
        memcpy(dst, src, data_size);
        ucc_memory_cpu_store_fence();
        ctrl->pi = seq_num;
    }
}

static inline void ucc_tl_shm_signal_to_children(ucc_tl_shm_seg_t * seg,
                                                 ucc_tl_shm_team_t *team,
                                                 ucc_tl_shm_sn_t    seq_num,
                                                 ucc_kn_tree_t *    tree)
{
    ucc_tl_shm_ctrl_t *ctrl;
    int                i;

    for (i = 0; i < tree->n_children; i++) {
        ctrl     = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        ctrl->pi = seq_num;
    }
}

#endif
