/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"
#include "utils/ucc_dt_reduce.h"

ucc_status_t
ucc_tl_shm_reduce_read(ucc_tl_shm_team_t *team, ucc_tl_shm_seg_t *seg,
                       ucc_tl_shm_task_t *task, ucc_kn_tree_t *tree,
                       int is_inline, size_t count, ucc_datatype_t dt,
                       ucc_coll_args_t *args);

ucc_status_t ucc_tl_shm_reduce_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *     tl_team,
                                    ucc_coll_task_t **    task_h);
#endif
