/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_H_
#define UCC_TL_SHM_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_math.h"
#include "utils/arch/cpu.h"
#include "tl_shm_knomial_pattern.h"

#include <assert.h>
#include <errno.h>
#include <sys/shm.h>
#include <sys/types.h>

#ifndef UCC_TL_SHM_DEFAULT_SCORE
#define UCC_TL_SHM_DEFAULT_SCORE 100
#endif

#ifdef HAVE_PROFILING_TL_SHM
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_SHM_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_SHM_PROFILE_FUNC_VOID     UCC_PROFILE_FUNC_VOID
#define UCC_TL_SHM_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_SHM_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_SHM_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

#define UCC_TL_SHM_SUPPORTED_COLLS                                             \
    (UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_REDUCE | UCC_COLL_TYPE_BARRIER |      \
     UCC_COLL_TYPE_FANIN | UCC_COLL_TYPE_FANOUT | UCC_COLL_TYPE_ALLREDUCE)

typedef enum ucc_tl_shm_bcast_progress_alg
{
    BCAST_WW,
    BCAST_WR,
    BCAST_RR,
    BCAST_RW,
    BCAST_LAST
} ucc_tl_shm_bcast_progress_alg_t;

typedef enum ucc_tl_shm_seg_layout
{
    SEG_LAYOUT_CONTIG,
    SEG_LAYOUT_SOCKET,
    SEG_LAYOUT_MIXED,
    SEG_LAYOUT_LAST
} ucc_tl_shm_seg_layout_t;

typedef enum ucc_tl_shm_group_mode
{
    GROUP_BY_NUMA,
    GROUP_BY_SOCKET,
    GROUP_BY_AUTO,
    GROUP_BY_LAST
} ucc_tl_shm_group_mode_t;

typedef struct ucc_kn_tree ucc_kn_tree_t;

typedef struct ucc_tl_shm_iface {
    ucc_tl_iface_t super;
} ucc_tl_shm_iface_t;

extern ucc_tl_shm_iface_t ucc_tl_shm;

typedef struct ucc_tl_shm_lib_config {
    ucc_tl_lib_config_t             super;
    uint32_t                        max_concurrent;
    uint32_t                        data_size;
    uint32_t                        ctrl_size;
    uint32_t                        bcast_base_radix;
    uint32_t                        bcast_top_radix;
    uint32_t                        reduce_base_radix;
    uint32_t                        reduce_top_radix;
    uint32_t                        fanin_base_radix;
    uint32_t                        fanin_top_radix;
    uint32_t                        fanout_base_radix;
    uint32_t                        fanout_top_radix;
    uint32_t                        barrier_base_radix;
    uint32_t                        barrier_top_radix;
    uint32_t                        max_trees_cached;
    uint32_t                        n_polls;
    uint32_t                        base_tree_only;
    uint32_t                        set_perf_params;
    ucc_tl_shm_seg_layout_t         layout;
    ucc_tl_shm_bcast_progress_alg_t bcast_alg;
    ucc_tl_shm_group_mode_t         group_mode;
} ucc_tl_shm_lib_config_t;

typedef struct ucc_tl_shm_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_shm_context_config_t;

typedef struct ucc_tl_shm_lib {
    ucc_tl_lib_t            super;
    ucc_tl_shm_lib_config_t cfg;
} ucc_tl_shm_lib_t;
UCC_CLASS_DECLARE(ucc_tl_shm_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_shm_context {
    ucc_tl_context_t            super;
    ucc_tl_shm_context_config_t cfg;
    ucc_mpool_t                 req_mp;
} ucc_tl_shm_context_t;
UCC_CLASS_DECLARE(ucc_tl_shm_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef uint64_t ucc_tl_shm_sn_t;

typedef struct ucc_tl_shm_ctrl {
    volatile ucc_tl_shm_sn_t pi;       /* bcast/fanout producer index */
    volatile ucc_tl_shm_sn_t pi2;      /* reduce/fanin consumer index */
    volatile ucc_tl_shm_sn_t ci;       /* consumer index */
    volatile ucc_tl_shm_sn_t rr;       /* consumer index */
    char                     data[1];  /* start of inline data */
} ucc_tl_shm_ctrl_t;

typedef struct ucc_tl_shm_seg {
    volatile void *ctrl; /* control array = start of seg */
    volatile void *data; /* start of the data segments */
} ucc_tl_shm_seg_t;

typedef struct ucc_tl_shm_tree {
    ucc_kn_tree_t *base_tree; /* tree for base group, can be NULL if the group
                                 does not exists or the process is not part of it */
    ucc_kn_tree_t *top_tree; /* tree for leaders group, can be NULL if the group
                                 does not exists or the process is not part of it */
    int            cached;
} ucc_tl_shm_tree_t;

typedef struct ucc_tl_shm_tree_cache_key {
    ucc_rank_t      base_radix;
    ucc_rank_t      top_radix;
    ucc_rank_t      root;
    ucc_coll_type_t coll_type;
    int             base_tree_only;
} ucc_tl_shm_tree_cache_key_t;

typedef struct ucc_tl_shm_tree_cache_elems {
    ucc_tl_shm_tree_cache_key_t key;
    ucc_tl_shm_tree_t *         tree;
} ucc_tl_shm_tree_cache_elems_t;

typedef struct ucc_tl_shm_tree_cache {
    size_t                         size;
    ucc_tl_shm_tree_cache_elems_t *elems;
} ucc_tl_shm_tree_cache_t;

typedef struct ucc_tl_shm_perf_params {
    int        base_tree_only;
    ucc_rank_t base_radix;
    ucc_rank_t top_radix;
} ucc_tl_shm_perf_params_t;

typedef struct ucc_tl_shm_pp_bcast {
    ucc_tl_shm_perf_params_t        super;
    ucc_tl_shm_bcast_progress_alg_t progress_alg;
} ucc_tl_shm_pp_bcast_t;

typedef struct ucc_tl_shm_pp_reduce {
    ucc_tl_shm_perf_params_t super;
} ucc_tl_shm_pp_reduce_t;

typedef struct ucc_tl_shm_task ucc_tl_shm_task_t;
typedef void (*perf_params_fn_t)(ucc_tl_shm_perf_params_t *params,
                                 ucc_tl_shm_task_t        *task);

#define UCC_TL_SHM_MAX_BASE_GROUPS 32
#define UCC_TL_SHM_N_PERF_PARAMS   11

typedef struct ucc_tl_shm_perf_key {
    ucc_cpu_vendor_t        cpu_vendor;
    ucc_cpu_model_t         cpu_model;
    ucc_rank_t              groups[UCC_TL_SHM_MAX_BASE_GROUPS];
    ucc_rank_t              n_groups;
    perf_params_fn_t        bcast_func;
    perf_params_fn_t        reduce_func;
    ucc_tl_shm_seg_layout_t layout;
    const char *            label;
    uint32_t                ds;
} ucc_tl_shm_perf_key_t;

extern ucc_tl_shm_perf_key_t *ucc_tl_shm_perf_params[UCC_TL_SHM_N_PERF_PARAMS];

typedef struct ucc_tl_shm_last_posted {
    ucc_tl_shm_sn_t seq_num;
    ucc_rank_t      reduce_root;
} ucc_tl_shm_last_posted_t;

typedef struct ucc_tl_shm_team {
    ucc_tl_team_t             super;
    ucc_tl_shm_seg_t *        segs;
    ucc_tl_shm_sn_t           seq_num;
    ucc_tl_shm_last_posted_t* last_posted;
    size_t                    max_inline;
    uint32_t                  n_base_groups;
    uint32_t                  my_group_id;
    int                       n_concurrent;
    int                       is_group_leader;
    int                       need_free_group_rank_map;
    int                       need_free_rank_group_id_map;
    ucc_sbgp_t *              base_groups;
    ucc_sbgp_t *              leaders_group;
    ucc_topo_t *              topo;
    void **                   shm_buffers;
    ucc_ep_map_t              ctx_map;
    ucc_ep_map_t              group_rank_map;
    ucc_ep_map_t              rank_group_id_map;
    size_t                    ctrl_size;
    size_t                    data_size;
    uint32_t                  arch_data_size;
    ucc_tl_shm_tree_cache_t  *tree_cache;
    ucc_tl_shm_seg_layout_t   layout;
    ucc_status_t              status;
    int *                     allgather_dst;
    void *                    oob_req;
    volatile uint32_t        *init_sync;
    perf_params_fn_t          perf_params_bcast;
    perf_params_fn_t          perf_params_reduce;
} ucc_tl_shm_team_t;

UCC_CLASS_DECLARE(ucc_tl_shm_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_shm_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_shm_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_shm_lib_t))
#define UCC_TL_SHM_TEAM_LIB(_team)                                             \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_shm_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#endif
