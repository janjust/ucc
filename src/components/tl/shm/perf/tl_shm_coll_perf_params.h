/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm.h"
#include "../tl_shm_coll.h"

extern ucc_tl_shm_perf_key_t intel_broadwell_2_14;
extern ucc_tl_shm_perf_key_t intel_broadwell_2_16;
extern ucc_tl_shm_perf_key_t intel_broadwell_1_14;
extern ucc_tl_shm_perf_key_t intel_broadwell_1_8;
extern ucc_tl_shm_perf_key_t intel_skylake_2_20;
extern ucc_tl_shm_perf_key_t intel_skylake_2_28;
extern ucc_tl_shm_perf_key_t amd_rome_2_64;
extern ucc_tl_shm_perf_key_t amd_rome_8_16;
extern ucc_tl_shm_perf_key_t amd_milan_2_64;
extern ucc_tl_shm_perf_key_t amd_milan_8_16;

static inline void
ucc_tl_shm_perf_params_generic_bcast(ucc_tl_shm_perf_params_t *params,
                                     ucc_tl_shm_task_t        *task)
{
    ucc_tl_shm_pp_bcast_t *p = ucc_derived_of(params, ucc_tl_shm_pp_bcast_t);

    p->progress_alg         = TASK_LIB(task)->cfg.bcast_alg;
    p->super.base_tree_only = TASK_LIB(task)->cfg.base_tree_only;
    p->super.base_radix     = TASK_LIB(task)->cfg.bcast_base_radix;
    p->super.top_radix      = p->super.base_tree_only ? 0 :
                                  TASK_LIB(task)->cfg.bcast_top_radix;
}

static inline void
ucc_tl_shm_perf_params_generic_reduce(ucc_tl_shm_perf_params_t *params,
                                      ucc_tl_shm_task_t        *task)
{
    ucc_tl_shm_pp_reduce_t *p = ucc_derived_of(params, ucc_tl_shm_pp_reduce_t);

    p->super.base_tree_only = TASK_LIB(task)->cfg.base_tree_only;
    p->super.base_radix     = TASK_LIB(task)->cfg.reduce_base_radix;
    p->super.top_radix      = p->super.base_tree_only ? 0 :
                                  TASK_LIB(task)->cfg.reduce_top_radix;
}
#define TL_SHM_PERF_KEY_DECLARE_BCAST(_name,                          \
 /* bcast inline params */      _b_alg1, _b_bto1, _b_br1, _b_tr1,     \
 /* bcast large params  */      _b_alg2, _b_bto2, _b_br2, _b_tr2)     \
                                                                      \
    static void ucc_tl_shm_ ## _name ## _bcast(                       \
        ucc_tl_shm_perf_params_t *params,                             \
        ucc_tl_shm_task_t        *task)                               \
    {                                                                 \
        ucc_tl_shm_team_t *team      = TASK_TEAM(task);               \
        size_t             data_size =                                \
            ucc_coll_args_msgsize(&task->super.bargs.args,            \
                                  UCC_TL_TEAM_RANK(team),             \
                                  UCC_TL_TEAM_SIZE(team));            \
        ucc_tl_shm_pp_bcast_t *p =                                    \
            ucc_derived_of(params, ucc_tl_shm_pp_bcast_t);            \
                                                                      \
        if (data_size <= team->max_inline) {                          \
            p->progress_alg         = _b_alg1;                        \
            p->super.base_tree_only = _b_bto1;                        \
            p->super.base_radix     = _b_br1;                         \
            p->super.top_radix      = _b_tr1;                         \
        } else {                                                      \
            p->progress_alg         = _b_alg2;                        \
            p->super.base_tree_only = _b_bto2;                        \
            p->super.base_radix     = _b_br2;                         \
            p->super.top_radix      = _b_tr2;                         \
        }                                                             \
    }                                                                 \

#define TL_SHM_PERF_KEY_DECLARE_REDUCE(_name,                         \
 /* reduce inline params */     _r_bto1, _r_br1, _r_tr1,              \
 /* reduce large params  */     _r_bto2, _r_br2, _r_tr2)              \
                                                                      \
    static void ucc_tl_shm_ ## _name ## _reduce(                      \
        ucc_tl_shm_perf_params_t *params,                             \
        ucc_tl_shm_task_t        *task)                               \
    {                                                                 \
        ucc_tl_shm_team_t *team      = TASK_TEAM(task);               \
        size_t             data_size =                                \
            ucc_coll_args_msgsize(&task->super.bargs.args,            \
                                  UCC_TL_TEAM_RANK(team),             \
                                  UCC_TL_TEAM_SIZE(team));            \
        ucc_tl_shm_pp_reduce_t *p =                                   \
            ucc_derived_of(params, ucc_tl_shm_pp_reduce_t);           \
                                                                      \
        if (data_size <= team->max_inline) {                          \
            p->super.base_tree_only = _r_bto1;                        \
            p->super.base_radix     = _r_br1;                         \
            p->super.top_radix      = _r_tr1;                         \
        } else {                                                      \
            p->super.base_tree_only = _r_bto2;                        \
            p->super.base_radix     = _r_br2;                         \
            p->super.top_radix      = _r_tr2;                         \
        }                                                             \
    }                                                                 \

#define TL_SHM_PERF_KEY_DECLARE_BASE(_name, _vendor, _model,          \
                                     _bcast_fn, _reduce_fn,           \
                                     _layout, _n_groups, _ds, ...)    \
    ucc_tl_shm_perf_key_t _name = {                                   \
        .cpu_vendor  = UCC_CPU_VENDOR_ ## _vendor,                    \
        .cpu_model   = UCC_CPU_MODEL_ ## _vendor ## _ ## _model,      \
        .label       = UCC_PP_MAKE_STRING(_name),                     \
        .groups      = { __VA_ARGS__},                                \
        .n_groups    = _n_groups,                                     \
        .layout      = _layout,                                       \
        .bcast_func  = _bcast_fn,                                     \
        .reduce_func = _reduce_fn,                                    \
        .ds          = _ds}

#define TL_SHM_PERF_KEY_DECLARE(_name, _vendor, _model,               \
 /* bcast inline params */      _b_alg1, _b_bto1, _b_br1, _b_tr1,     \
 /* bcast large params  */      _b_alg2, _b_bto2, _b_br2, _b_tr2,     \
 /* reduce inline params */     _r_bto1, _r_br1, _r_tr1,              \
 /* reduce large params  */     _r_bto2, _r_br2, _r_tr2,              \
                                _layout, _n_groups, _ds, ...)         \
    TL_SHM_PERF_KEY_DECLARE_BCAST(_name, _b_alg1, _b_bto1, _b_br1,    \
                         _b_tr1, _b_alg2, _b_bto2, _b_br2, _b_tr2)    \
    TL_SHM_PERF_KEY_DECLARE_REDUCE(_name, _r_bto1, _r_br1,            \
                         _r_tr1, _r_bto2, _r_br2, _r_tr2)             \
    TL_SHM_PERF_KEY_DECLARE_BASE(_name, _vendor, _model,              \
                                 ucc_tl_shm_ ## _name ## _bcast,      \
                                 ucc_tl_shm_ ## _name ## _reduce,     \
                                 _layout, _n_groups, _ds, __VA_ARGS__)
