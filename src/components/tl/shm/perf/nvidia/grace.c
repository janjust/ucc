/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

TL_SHM_PERF_KEY_DECLARE(nvidia_grace_1_72, NVIDIA, GRACE,
                        BCAST_WW, 0, 4, 4,
                        BCAST_RR, 0, 12, 4,
                        1, 12, 2, 1, 8, 2,
                        SEG_LAYOUT_CONTIG, 1, 8192, 72);
