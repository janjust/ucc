/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

TL_SHM_PERF_KEY_DECLARE(amd_genoa_2_96, AMD, GENOA,
                        BCAST_RW, 0, 4, 2, BCAST_WR, 0, 8, 8,
                        0, 8, 4, 0, 8, 2,
                        SEG_LAYOUT_SOCKET, 2, 8192, 96, 96);

static void ucc_tl_shm_amd_genoa_8_24_bcast(ucc_tl_shm_perf_params_t *params,
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
        p->super.base_radix     = 4;
        p->super.top_radix      = 2;
    } else {
        p->progress_alg         = BCAST_RR;
        p->super.base_radix     = 4;
        p->super.top_radix      = 8;
    }
}

TL_SHM_PERF_KEY_DECLARE_REDUCE(amd_genoa_8_24, 0, 4, 2, 0, 12, 8);

TL_SHM_PERF_KEY_DECLARE_BASE(amd_genoa_8_24, AMD, GENOA,
                             ucc_tl_shm_amd_genoa_8_24_bcast,
                             ucc_tl_shm_amd_genoa_8_24_reduce,
                             SEG_LAYOUT_SOCKET, 8, 8192,
                             24, 24, 24, 24, 24, 24, 24, 24);
