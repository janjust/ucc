/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "tl_shm_knomial_pattern.h"
#include "perf/tl_shm_coll_perf_params.h"
#include "core/ucc_ee.h"
#include "core/ucc_team.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_sys.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "barrier/barrier.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"
#include "allreduce/allreduce.h"
#include <sys/stat.h>

#define SHM_MODE                                                               \
    (IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IWOTH | S_IRGRP | S_IWGRP)

static ucc_rank_t ucc_tl_shm_team_rank_to_group_id(ucc_tl_shm_team_t *team,
                                                   ucc_rank_t         r)
{
    int i, j;
    for (i = 0; i < team->n_base_groups; i++) {
        for (j = 0; j < team->base_groups[i].group_size; j++) {
            if (r == ucc_ep_map_eval(team->base_groups[i].map, j)) {
                /* found team rank r in base group i */
                break;
            }
        }
        if (j < team->base_groups[i].group_size) {
            break;
        }
    }
    ucc_assert(i < team->n_base_groups && j < team->base_groups[i].group_size);
    return i;
}

static ucc_status_t ucc_tl_shm_rank_group_id_map_init(ucc_tl_shm_team_t *team)
{
    ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = (ucc_rank_t *)ucc_malloc(team_size * sizeof(*ranks));
    int         i;

    if (!ranks) {
        return UCC_ERR_NO_MEMORY;
    }

    //TODO opt for single group
    for (i = 0; i < team_size; i++) {
        ranks[i] = ucc_tl_shm_team_rank_to_group_id(team, i);
    }
    team->rank_group_id_map =
        ucc_ep_map_from_array(&ranks, team_size, team_size, 1);
    if (ranks) {
        team->need_free_rank_group_id_map = 1;
    }
    return UCC_OK;
}

static ucc_rank_t ucc_tl_shm_team_rank_to_group_rank(ucc_tl_shm_team_t *team,
                                                     ucc_rank_t         r)
{
    ucc_rank_t group_id = ucc_ep_map_eval(team->rank_group_id_map, r);
    ucc_rank_t i;

    for (i = 0; i < team->base_groups[group_id].group_size; i++) {
        if (ucc_ep_map_eval(team->base_groups[group_id].map, i) == r) {
            break;
        }
    }
    ucc_assert(i < team->base_groups[group_id].group_size);
    return i;
}

static ucc_status_t ucc_tl_shm_group_rank_map_init(ucc_tl_shm_team_t *team)
{
    ucc_rank_t  team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t *ranks = (ucc_rank_t *)ucc_malloc(team_size * sizeof(*ranks)); /* NOLINT to supress clang_tidy check of team size == 0, already checked in core */

    ucc_rank_t  i;

    if (!ranks) {
        return UCC_ERR_NO_MEMORY;
    }
    for (i = 0; i < team_size; i++) {
        ranks[i] = ucc_tl_shm_team_rank_to_group_rank(team, i);
    }
    team->group_rank_map =
        ucc_ep_map_from_array(&ranks, team_size, team_size, 1);
    if (ranks) {
        team->need_free_group_rank_map = 1;
    }
    return UCC_OK;
}

static int _compare(const void *a, const void *b)
{
    return *((ucc_rank_t *)a) < *((ucc_rank_t *)b) ? -1 : 1;
}

static inline int check_groups_key(ucc_tl_shm_team_t     *team,
                                   ucc_tl_shm_perf_key_t *key)
{
    ucc_rank_t groups[team->n_base_groups];
    int        i;

    if (team->n_base_groups == key->n_groups) {
        for (i = 0; i < team->n_base_groups; i++) {
            groups[i] = team->base_groups[i].group_size;
        }
        qsort(groups, team->n_base_groups, sizeof(ucc_rank_t), _compare);
        for (i = 0; i < team->n_base_groups; i++) {
            if (groups[i] != key->groups[i]) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

static inline void ucc_tl_shm_set_team_key(ucc_tl_shm_team_t *team, ucc_tl_shm_perf_key_t *key)
{
    team->perf_params_bcast  = key->bcast_func;
    team->perf_params_reduce = key->reduce_func;
    team->layout             = key->layout;
    team->arch_data_size     = key->ds;
    if (team->layout == SEG_LAYOUT_LAST) {
        team->layout = UCC_TL_SHM_TEAM_LIB(team)->cfg.layout;
    }
    if (0 == UCC_TL_TEAM_RANK(team)) {
        tl_debug(UCC_TL_TEAM_LIB(team), "using perf params: %s", key->label);
    }
}

static inline void ucc_tl_shm_set_perf_funcs(ucc_tl_shm_team_t *team)
{

    ucc_tl_shm_perf_key_t **key         = ucc_tl_shm_perf_params;
    ucc_tl_shm_perf_key_t **key_generic = ucc_tl_shm_perf_params_generic;
    ucc_cpu_vendor_t        vendor;
    ucc_cpu_model_t         model;

    vendor = ucc_arch_get_cpu_vendor();
    model  = ucc_arch_get_cpu_model();

    // First try to match based on all of (vendor, model, n_groups, group_size_0, group_size_1, ...)
    while (*key) {
        if ((*key)->cpu_vendor == vendor && (*key)->cpu_model == model &&
            check_groups_key(team, *key)) {
            ucc_tl_shm_set_team_key(team, *key);
            return;
        }
        key++;
    }

    // If no match so far, try again but with the generic key for (vendor, model)
    while (*key_generic) {
        if ((*key_generic)->cpu_vendor == vendor && (*key_generic)->cpu_model == model) {
            ucc_tl_shm_set_team_key(team, *key_generic);
            return;
        }
        key_generic++;
    }

    // Still no match, use the cfg's default values
    if (0 == UCC_TL_TEAM_RANK(team)) {
        tl_debug(UCC_TL_TEAM_LIB(team), "using perf params: generic");
    }
}

static void ucc_tl_shm_init_segs(ucc_tl_shm_team_t *team)
{
    size_t cfg_ctrl_size       = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t cfg_data_size       = team->arch_data_size;
    ucc_tl_shm_seg_layout_t sl = team->layout;
    size_t                  page_size = ucc_get_page_size();
    size_t ctrl_offset, data_offset, grp_ctrl_size,
        grp_data_size, grp_seg_size, grp_0_data_size;
    void * ctrl, *data;
    ucc_rank_t group_size;
    int        i, j;

    for (i = 0; i < team->n_concurrent; i++) {
        ctrl_offset = 0;
        data_offset = 0;
        for (j = 0; j < team->n_base_groups; j++) {
            group_size      = team->base_groups[j].group_size;
            grp_ctrl_size   = ucc_align_up(group_size * cfg_ctrl_size,
                                           page_size);
            grp_data_size   = group_size * cfg_data_size;
            grp_0_data_size = team->base_groups[0].group_size * cfg_data_size;
            grp_seg_size    = grp_ctrl_size + grp_data_size;

            if (sl == SEG_LAYOUT_CONTIG) {
                ctrl = PTR_OFFSET(team->shm_buffers[0],
                                  (team->ctrl_size + team->data_size) * i +
                                      ctrl_offset);
                data = PTR_OFFSET(ctrl,
                                  team->ctrl_size + data_offset - ctrl_offset);
                ctrl_offset += grp_ctrl_size;
                data_offset += grp_data_size;
            } else if (sl == SEG_LAYOUT_MIXED) {
                ctrl = PTR_OFFSET(team->shm_buffers[0],
                                  (team->ctrl_size + grp_0_data_size) * i +
                                      ctrl_offset);
                if (j == 0) {
                    data = PTR_OFFSET(ctrl, team->ctrl_size - ctrl_offset);
                } else {
                    data = PTR_OFFSET(team->shm_buffers[j], grp_data_size * i);
                }
                ctrl_offset += grp_ctrl_size;
            } else {
                ctrl = PTR_OFFSET(team->shm_buffers[j], grp_seg_size * i);
                data = PTR_OFFSET(ctrl, grp_ctrl_size);
            }
            team->segs[i * team->n_base_groups + j].ctrl = ctrl;
            team->segs[i * team->n_base_groups + j].data = data;
        }
    }
}

static ucc_status_t ucc_tl_shm_seg_alloc(ucc_tl_shm_team_t *team)
{
    ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team),
               team_size = UCC_TL_TEAM_SIZE(team);
    size_t     cfg_ctrl_size        = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t     cfg_data_size        = team->arch_data_size;
    ucc_tl_shm_seg_layout_t sl      = team->layout;
    size_t                  shmsize = 0;
    int                     shmid   = -1;
    ucc_team_oob_coll_t     oob     = UCC_TL_TEAM_OOB(team);
    ucc_rank_t              gsize;
    ucc_status_t            status;
    size_t                  page_size;

    gsize = team->base_groups[team->my_group_id].group_size;
    team->allgather_dst =
        (int *)ucc_malloc(sizeof(int) * (team_size + 1), "algather dst buffer");

    if (sl == SEG_LAYOUT_CONTIG) {
        if (team_rank == 0) {
            shmsize = team->n_concurrent * (team->ctrl_size + team->data_size);
        }
    } else if (sl == SEG_LAYOUT_MIXED) {
        if (team_rank == 0) {
            ucc_assert(team->is_group_leader);
            shmsize =
                team->n_concurrent * (team->ctrl_size + gsize * cfg_data_size);
        } else if (team->is_group_leader) {
            shmsize = team->n_concurrent * gsize * cfg_data_size;
        }
    } else if (team->is_group_leader) {
        page_size = ucc_get_page_size();
        shmsize   = team->n_concurrent *
                  (gsize * cfg_data_size +
                   ucc_align_up(gsize * cfg_ctrl_size, page_size));
    }
    /* LOWEST on node rank  within the comm will initiate the segment creation.
     * Everyone else will attach. */
    if (shmsize != 0) {
        shmsize += sizeof(uint32_t);
        shmid = shmget(IPC_PRIVATE, shmsize, SHM_MODE);
        if (shmid < 0) {
            tl_debug(team->super.super.context->lib,
                     "Root: shmget failed, shmid=%d, shmsize=%ld, errno: %s",
                     shmid, shmsize, strerror(errno));
            goto allgather;
        }
        team->shm_buffers[team->my_group_id] = (void *)shmat(shmid, NULL, 0);
        if (team->shm_buffers[team->my_group_id] == (void *)-1) {
            shmid                                = -2;
            team->shm_buffers[team->my_group_id] = NULL;
            tl_debug(team->super.super.context->lib, "shmat failed, errno: %s",
                     strerror(errno));
            goto allgather;
        }
        memset(team->shm_buffers[team->my_group_id], 0, shmsize);
        shmctl(shmid, IPC_RMID, NULL);
    }
allgather:
    team->allgather_dst[team_size] = shmid;
    status = oob.allgather(&team->allgather_dst[team_size], team->allgather_dst,
                           sizeof(int), oob.coll_info, &team->oob_req);
    if (UCC_OK != status) {
        tl_error(team->super.super.context->lib, "allgather failed");
        return status;
    }
    return UCC_OK;
}

static ucc_sbgp_type_t
ucc_tl_shm_get_group_sbgp_type(ucc_tl_shm_team_t *team)
{
    int                     numa_bound, sock_bound;
    ucc_tl_shm_group_mode_t gm;

    gm         = UCC_TL_SHM_TEAM_LIB(team)->cfg.group_mode;
    sock_bound = UCC_TL_CORE_CTX(team)->topo->sock_bound;
    numa_bound = UCC_TL_CORE_CTX(team)->topo->numa_bound;

    if (gm == GROUP_BY_SOCKET) {
        if (sock_bound != 1) {
            tl_error(UCC_TL_TEAM_LIB(team), "group_mode SOCKET can not be used"
                     " because processes are not bound to sockets");
            return UCC_SBGP_NUMA_LEADERS;
        }
        return UCC_SBGP_SOCKET_LEADERS;
    }

    if (gm == GROUP_BY_NUMA) {
        if (numa_bound != 1) {
            tl_error(UCC_TL_TEAM_LIB(team), "group_mode NUMA can not be used"
                     " because processes are not bound to numa nodes");
            return UCC_SBGP_SOCKET_LEADERS;
        } else {
            return UCC_SBGP_NUMA_LEADERS;
        }
    }

    /* both bindgins are available and gm == AUTO,
       use some bechmark based heuristics */
    if (UCC_CPU_VENDOR_INTEL == ucc_arch_get_cpu_vendor()) {
        /* On intel default by socket grouping is best */
        return UCC_SBGP_SOCKET_LEADERS;
    }
    return (ucc_topo_n_numas(team->topo) > ucc_topo_n_sockets(team->topo)) ?
        UCC_SBGP_NUMA_LEADERS : UCC_SBGP_SOCKET_LEADERS;
}

UCC_CLASS_INIT_FUNC(ucc_tl_shm_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_shm_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_shm_context_t);
    ucc_status_t status;
    int          n_sbgps, i, max_trees;
    ucc_rank_t   team_size;
    uint32_t     cfg_ctrl_size, group_size;
    ucc_subset_t subset;
    size_t       page_size;
    ucc_sbgp_type_t group_sbgp_type;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    if (!ucc_team_map_is_single_node(params->team, params->map)) {
        tl_debug(ctx->super.super.lib, "multi node team is not supported");
        return UCC_ERR_INVALID_PARAM;
    }

    if (NULL == UCC_TL_CORE_CTX(self)->topo) {
        /* CORE context does not have topo information -
         * local context mode */
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(self)->ctx_map,
                                      &UCC_TL_TEAM_MAP(self),
                                      &self->ctx_map);
    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to create ctx map");
        return status;
    }
    subset.map    = self->ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(self);
    team_size     = UCC_TL_TEAM_SIZE(self);
    cfg_ctrl_size = UCC_TL_SHM_TEAM_LIB(self)->cfg.ctrl_size;

    self->layout             = UCC_TL_SHM_TEAM_LIB(self)->cfg.layout;
    self->perf_params_bcast  = ucc_tl_shm_perf_params_generic_bcast;
    self->perf_params_reduce = ucc_tl_shm_perf_params_generic_reduce;
    self->seq_num            = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_concurrent;
    self->status             = UCC_INPROGRESS;
    self->n_concurrent       = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_concurrent;
    self->arch_data_size     = UCC_TL_SHM_TEAM_LIB(self)->cfg.data_size;
    self->max_inline         = cfg_ctrl_size - ucc_offsetof(ucc_tl_shm_ctrl_t,
                                                            data);
    self->need_free_rank_group_id_map = 0;
    self->need_free_group_rank_map = 0;
    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(self)->topo, &self->topo);

    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to init team topo");
        goto err_topo_init;
    }

    self->last_posted = ucc_calloc(sizeof(*self->last_posted),
                                   self->n_concurrent, "last_posted");
    if (!self->last_posted) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for last_posted array",
                 sizeof(*self->last_posted) * self->n_concurrent);
        status = UCC_ERR_NO_MEMORY;
        goto err_topo_cleanup;
    }
    for (i = 0; i < self->n_concurrent; i++) {
        self->last_posted[i].reduce_root = UCC_RANK_INVALID;
    }

    max_trees        = UCC_TL_SHM_TEAM_LIB(self)->cfg.max_trees_cached;
    self->tree_cache = (ucc_tl_shm_tree_cache_t *)ucc_malloc(
        sizeof(ucc_tl_shm_tree_cache_t), "tree_cache");

    if (!self->tree_cache) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tree_cache",
                 sizeof(ucc_tl_shm_tree_cache_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_tree;
    }

    self->tree_cache->elems = (ucc_tl_shm_tree_cache_elems_t *)ucc_malloc(
        max_trees * sizeof(ucc_tl_shm_tree_cache_elems_t), "tree_cache->elems");

    if (!self->tree_cache->elems) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tree_cache->elems",
                 max_trees * sizeof(ucc_tl_shm_tree_cache_elems_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_elems;
    }

    self->tree_cache->size = 0;
    group_sbgp_type        = ucc_tl_shm_get_group_sbgp_type(self);
    self->leaders_group    = ucc_topo_get_sbgp(self->topo, group_sbgp_type);

    if (self->leaders_group->status == UCC_SBGP_NOT_EXISTS ||
        self->leaders_group->group_size == team_size) {
        self->leaders_group->group_size = 0;
        self->base_groups   = ucc_topo_get_sbgp(self->topo, UCC_SBGP_NODE);
        self->n_base_groups = 1;
    } else {
        /* sbgp type is either SOCKET or NUMA
     * depending on the config: grouping type */
        self->n_base_groups = self->leaders_group->group_size;

        ucc_assert(self->n_base_groups ==
                   ((group_sbgp_type == UCC_SBGP_SOCKET_LEADERS)
                        ? self->topo->n_sockets
                        : self->topo->n_numas));

        if (group_sbgp_type == UCC_SBGP_SOCKET_LEADERS) {
            status = ucc_topo_get_all_sockets(self->topo, &self->base_groups,
                                              &n_sbgps);
        } else {
            status = ucc_topo_get_all_numas(self->topo, &self->base_groups,
                                            &n_sbgps);
        }
        if (UCC_OK != status) {
            tl_error(ctx->super.super.lib, "failed to get all base subgroups");
            goto err_sockets;
        }
    }
    self->is_group_leader = 0;
    for (i = 0; i < self->n_base_groups; i++) {
        if (UCC_TL_TEAM_RANK(self) ==
            ucc_ep_map_eval(self->base_groups[i].map, 0)) {
            self->is_group_leader = 1;
        }
    }
    //NOLINTNEXTLINE linter FP, n_base_groups and n_concurrent preset
    self->segs = (ucc_tl_shm_seg_t *)ucc_malloc(
        sizeof(ucc_tl_shm_seg_t) * self->n_base_groups * self->n_concurrent,
        "shm_segs");
    if (!self->segs) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for shm_segs",
                 sizeof(ucc_tl_shm_seg_t) * self->n_concurrent);
        status = UCC_ERR_NO_MEMORY;
        goto err_sockets;
    }

    /* the above call should return ALL socket/numa sbgps including size=1 subgroups */
    status = ucc_tl_shm_rank_group_id_map_init(self);
    if (UCC_OK != status) {
        goto err_segs;
    }

    status = ucc_tl_shm_group_rank_map_init(self);
    if (UCC_OK != status) {
        goto err_group_id_map;
    }

    self->my_group_id =
        ucc_ep_map_eval(self->rank_group_id_map, UCC_TL_TEAM_RANK(self));

    if (UCC_TL_SHM_TEAM_LIB(self)->cfg.set_perf_params) {
        ucc_tl_shm_set_perf_funcs(self);
    }

    self->data_size  = self->arch_data_size * team_size;
    self->ctrl_size  = 0;
    page_size        = ucc_get_page_size();

    for (i = 0; i < self->n_base_groups; i++) {
        group_size = self->base_groups[i].group_size;
        self->ctrl_size += ucc_align_up(group_size * cfg_ctrl_size,
                                        page_size);
    }

    //NOLINTNEXTLINE linter FP, n_base_groups preset
    self->shm_buffers = (void *)ucc_calloc(sizeof(void *), self->n_base_groups,
                                           "shm_buffers");
    if (!self->shm_buffers) {
        status = UCC_ERR_NO_MEMORY;
        goto err_ranks_map;
    }

    status = ucc_tl_shm_seg_alloc(self);
    if (UCC_OK != status) {
        goto err_buffers;
    }
    return UCC_OK;

err_buffers:
    ucc_free(self->shm_buffers);
err_ranks_map:
    if (self->need_free_group_rank_map) {
        ucc_free(self->group_rank_map.array.map);
        self->need_free_group_rank_map = 0;
    }
err_group_id_map:
    if (self->need_free_rank_group_id_map) {
        ucc_free(self->rank_group_id_map.array.map);
        self->need_free_rank_group_id_map = 0;
    }
err_segs:
    ucc_free(self->segs);
err_sockets:
    ucc_free(self->tree_cache->elems);
err_elems:
    ucc_free(self->tree_cache);
err_tree:
    ucc_free(self->last_posted);
err_topo_cleanup:
    ucc_topo_cleanup(self->topo);
err_topo_init:
    ucc_ep_map_destroy_nested(&self->ctx_map);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_shm_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_shm_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    int                i;

    for (i = 0; i < team->n_base_groups; i++) {
        if (team->shm_buffers[i]) {
            if (shmdt(team->shm_buffers[i]) == -1) {
                tl_error(team->super.super.context->lib, "shmdt failed");
                return UCC_ERR_NO_MESSAGE;
            }
        }
    }
    ucc_free(team->shm_buffers);
    for (i = 0; i < team->tree_cache->size; i++) {
        ucc_free(team->tree_cache->elems[i].tree->top_tree);
        ucc_free(team->tree_cache->elems[i].tree->base_tree);
        ucc_free(team->tree_cache->elems[i].tree);
    }
    ucc_free(team->tree_cache->elems);
    ucc_free(team->tree_cache);
    if (team->need_free_group_rank_map) {
        ucc_free(team->group_rank_map.array.map);
    }
    if (team->need_free_rank_group_id_map) {
        ucc_free(team->rank_group_id_map.array.map);
    }
    ucc_free(team->segs);
    ucc_free(team->last_posted);
    ucc_ep_map_destroy_nested(&team->ctx_map);
    ucc_topo_cleanup(team->topo);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_shm_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_shm_team_t * team   = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_team_oob_coll_t oob    = UCC_TL_TEAM_OOB(team);
    ucc_status_t        status;
    int                 i, shmid;
    ucc_rank_t          group_leader;
    struct shmid_ds     ds;

    if (team->oob_req) {
        status = oob.req_test(team->oob_req);
        if (status == UCC_INPROGRESS) {
            return UCC_INPROGRESS;
        }
        if (status != UCC_OK) {
            oob.req_free(team->oob_req);
            tl_error(team->super.super.context->lib, "oob req test failed");
            return status;
        }
        status = oob.req_free(team->oob_req);
        if (status != UCC_OK) {
            tl_error(team->super.super.context->lib, "oob req free failed");
            return status;
        }
        team->oob_req = NULL;

        for (i = 0; i < team->n_base_groups; i++) {
            group_leader = ucc_ep_map_eval(team->base_groups[i].map, 0);
            shmid        = team->allgather_dst[group_leader];

            if (shmid == -2 || (group_leader == 0 && shmid == -1)) {
                return UCC_ERR_NO_RESOURCE;
            }

            if (shmid == -1) {
                /* no shm seg from that group leader */
                continue;
            }

            if (UCC_TL_TEAM_RANK(team) != group_leader) {
                team->shm_buffers[i] = (void *)shmat(shmid, NULL, 0);
                if (team->shm_buffers[i] == (void *)-1) {
                    tl_error(team->super.super.context->lib,
                             "Child failed to attach to shmseg, errno: %s\n",
                             strerror(errno));
                    return UCC_ERR_NO_RESOURCE;
                }
            }
        }
        /* Need to wait for others to join to segment.
           Otherwise, they may immediately finish the team creation and may
           go to team destruction and segment will be removed from OS
           before everybody attached to it. */

        shmid = team->allgather_dst[ucc_ep_map_eval(team->base_groups[0].map, 0)];
        shmctl(shmid, IPC_STAT, &ds);
        team->init_sync = PTR_OFFSET(team->shm_buffers[0],
                          ds.shm_segsz - sizeof(uint32_t));
        ucc_atomic_add32(team->init_sync, 1);
        ucc_tl_shm_init_segs(team);
        ucc_free(team->allgather_dst);
    }

    if (*team->init_sync != UCC_TL_TEAM_SIZE(team)) {
        return UCC_INPROGRESS;
    }

    team->status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_team_get_scores(ucc_base_team_t *  tl_team,
                                        ucc_coll_score_t **score_p)
{
    ucc_tl_shm_team_t * team      = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_base_lib_t *    lib       = UCC_TL_TEAM_LIB(team);
    ucc_base_context_t *ctx       = UCC_TL_TEAM_CTX(team);
    size_t              data_size = team->arch_data_size;
    size_t              ctrl_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.ctrl_size;
    size_t              inline_size = ctrl_size - sizeof(ucc_tl_shm_ctrl_t);
    size_t              max_size    = ucc_max(inline_size, data_size);
    ucc_memory_type_t   mt          = UCC_MEMORY_TYPE_HOST;
    ucc_coll_score_t *  score;
    ucc_status_t        status;
    ucc_coll_score_team_info_t team_info;

    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TL_SHM_DEFAULT_SCORE;
    team_info.init                = NULL;
    team_info.num_mem_types       = 1;
    team_info.supported_mem_types = &mt;
    team_info.supported_colls     = UCC_TL_SHM_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "faild to alloc score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_BCAST, UCC_MEMORY_TYPE_HOST, 0, max_size,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_bcast_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_REDUCE, UCC_MEMORY_TYPE_HOST, 0, max_size,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_reduce_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_FANIN, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_fanin_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_FANOUT, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_fanout_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_barrier_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLREDUCE, UCC_MEMORY_TYPE_HOST, 0, max_size,
        UCC_TL_SHM_DEFAULT_SCORE, ucc_tl_shm_allreduce_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        goto err;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                tl_team, score);


        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }

    //TODO: check that collective range does not exceed data size

    *score_p = score;
    return UCC_OK;

err:
    ucc_coll_score_free(score);
    *score_p = NULL;
    return status;
}
