/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm_coll_perf_params.h"

ucc_tl_shm_perf_key_t* ucc_tl_shm_perf_params[UCC_TL_SHM_N_PERF_PARAMS] =
{
    &intel_broadwell_2_14,
    &intel_broadwell_2_16,
    &intel_broadwell_1_14,
    &intel_broadwell_1_8,
    &intel_skylake_2_20,
    &intel_skylake_2_28,
    &amd_rome_2_64,
    &amd_rome_8_16,
    &amd_milan_2_64,
    &amd_milan_8_16,
    &amd_milan_16_32,
    &amd_genoa_2_96,
    &amd_genoa_8_24,
    &nvidia_grace_1_16,
    &nvidia_grace_1_32,
    &nvidia_grace_1_64,
    &nvidia_grace_1_72,
    &nvidia_grace_2_16,
    &nvidia_grace_2_32,
    &nvidia_grace_2_64,
    &nvidia_grace_2_72,
    NULL
};

// First check if perf_params match cpu/vendor/group/group_size. If not, then match just cpu/vendor
ucc_tl_shm_perf_key_t* ucc_tl_shm_perf_params_generic[UCC_TL_SHM_N_PERF_PARAMS_GENERIC] =
{
    &nvidia_grace,
    NULL
};
