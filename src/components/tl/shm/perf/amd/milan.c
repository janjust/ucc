/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

TL_SHM_PERF_KEY_DECLARE(amd_milan_2_64, AMD, MILAN,
                        BCAST_RW, 0, 4, 2, BCAST_WR, 0, 8, 8,
                        0, 8, 4, 0, 8, 2,
                        SEG_LAYOUT_SOCKET, 2, 8192, 64, 64);

static void ucc_tl_shm_amd_milan_8_16_bcast(ucc_tl_shm_perf_params_t *params,
                                            ucc_tl_shm_task_t        *task)
{
    ucc_tl_shm_team_t *team      = TASK_TEAM(task);
    size_t             data_size =
        ucc_coll_args_msgsize(&task->super.bargs.args, UCC_TL_TEAM_RANK(team),
                              UCC_TL_TEAM_SIZE(team));
    ucc_tl_shm_pp_bcast_t *p = ucc_derived_of(params, ucc_tl_shm_pp_bcast_t);

    p->super.base_tree_only = 0;
    if (data_size <= team->max_inline) {
        p->progress_alg         = BCAST_WW;
        p->super.base_radix     = 2;
        p->super.top_radix      = 4;
    } else {
        p->progress_alg         = BCAST_WR;
        p->super.base_radix     = 4;
        p->super.top_radix      = 4;
    }
}

TL_SHM_PERF_KEY_DECLARE_REDUCE(amd_milan_8_16, 0, 4, 8, 0, 8, 4);

TL_SHM_PERF_KEY_DECLARE_BASE(amd_milan_8_16, AMD, MILAN,
                             ucc_tl_shm_amd_milan_8_16_bcast,
                             ucc_tl_shm_amd_milan_8_16_reduce,
                             SEG_LAYOUT_SOCKET, 8, 8192,
                             16, 16, 16, 16, 16, 16, 16, 16);
