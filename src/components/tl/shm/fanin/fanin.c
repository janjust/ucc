/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "../tl_shm.h"
#include "fanin.h"

enum
{
    FANIN_STAGE_START,
    FANIN_STAGE_BASE_TREE,
    FANIN_STAGE_TOP_TREE,
};

static void ucc_tl_shm_fanin_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t * seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    ucc_tl_shm_ctrl_t *my_ctrl;

next_stage:
    switch (task->stage) {
    case FANIN_STAGE_START:
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect fanin->reduce pattern.*/
        SHMCHECK_GOTO(ucc_tl_shm_check_seg_ready(task, tree, 1), task, out);
        if (tree->base_tree) {
            task->stage = FANIN_STAGE_BASE_TREE;
        } else {
            task->stage = FANIN_STAGE_TOP_TREE;
        }
        goto next_stage;
    case FANIN_STAGE_BASE_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_fanin_signal(team, seg, task, tree->base_tree),
                      task, out);
        if (tree->top_tree) {
            task->stage = FANIN_STAGE_TOP_TREE;
            goto next_stage;
        }
        break;
    case FANIN_STAGE_TOP_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_fanin_signal(team, seg, task, tree->top_tree),
                      task, out);
        break;
    }

    my_ctrl     = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* fanin done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanin_progress_done", 0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_fanin_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    ucc_tl_shm_task_reset(task, team, UCC_RANK_INVALID);
    task->stage = FANIN_STAGE_START;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanin_start", 0);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
 }

ucc_status_t ucc_tl_shm_fanin_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     tl_team,
                                   ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_rank_t base_radix   = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanin_base_radix;
    ucc_rank_t top_radix    = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanin_top_radix;
    int        bto          = UCC_TL_SHM_TEAM_LIB(team)->cfg.base_tree_only;
    ucc_tl_shm_task_t *task;
    ucc_status_t       status;

    task = ucc_tl_shm_get_task(coll_args, team);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_shm_tree_init(
        team, coll_args->args.root, base_radix, top_radix,
        UCC_COLL_TYPE_FANIN, bto, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }

    task->super.post     = ucc_tl_shm_fanin_start;
    task->super.progress = ucc_tl_shm_fanin_progress;

    *task_h = &task->super;
    return UCC_OK;
}
