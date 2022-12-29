/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

TL_SHM_PERF_KEY_DECLARE(intel_skylake_2_20, INTEL, SKYLAKE,
                        BCAST_WW, 0, 2, 2, BCAST_WR, 0, 4, 2,
                        0, 5, 2, 0, 5, 2,
                        SEG_LAYOUT_SOCKET, 2, 8192, 20, 20);

TL_SHM_PERF_KEY_DECLARE(intel_skylake_2_28, INTEL, SKYLAKE,
                        BCAST_WW, 0, 2, 2, BCAST_WR, 0, 7, 2,
                        0, 2, 2, 0, 2, 2,
                        SEG_LAYOUT_SOCKET, 2, 8192, 28, 28);
