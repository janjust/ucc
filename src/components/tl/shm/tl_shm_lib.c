/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_shm_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_shm_lib_config_t *tl_shm_config =
        ucc_derived_of(config, ucc_tl_shm_lib_config_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_shm.super,
                              &tl_shm_config->super);
    memcpy(&self->cfg, tl_shm_config, sizeof(*tl_shm_config));

    if (self->cfg.ctrl_size < sizeof(ucc_tl_shm_ctrl_t)) {
        tl_warn(self, "ctrl_size cannot be smaller than %zd",
                sizeof(ucc_tl_shm_ctrl_t));
        self->cfg.ctrl_size = sizeof(ucc_tl_shm_ctrl_t);
    }
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_lib_t)
{
    tl_debug(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_shm_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_shm_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                     ucc_base_lib_attr_t * base_attr)
{
    ucc_tl_lib_attr_t *attr = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);
    attr->super.attr.thread_mode =
        UCC_THREAD_MULTIPLE; //TODO: run tm tests to make sure we support
    attr->super.attr.coll_types = UCC_TL_SHM_SUPPORTED_COLLS;
    attr->super.flags           = 0;
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE) {
        attr->super.min_team_size = lib->min_team_size;
    }

    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE) {
        attr->super.max_team_size = UCC_RANK_MAX;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_shm_get_lib_properties(ucc_base_lib_properties_t *prop)
{
    prop->default_team_size = 2;
    prop->min_team_size     = 2;
    prop->max_team_size     = UCC_RANK_MAX;
    return UCC_OK;
}
