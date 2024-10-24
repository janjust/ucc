/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef BCAST_H_
#define BCAST_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"

ucc_status_t ucc_tl_shm_bcast_write(ucc_tl_shm_team_t *team,
                                    ucc_tl_shm_seg_t * seg,
                                    ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    size_t data_size);

ucc_status_t ucc_tl_shm_bcast_read(ucc_tl_shm_team_t *team,
                                   ucc_tl_shm_seg_t * seg,
                                   ucc_tl_shm_task_t *task, ucc_kn_tree_t *tree,
                                   int is_inline, int *is_op_root,
                                   size_t data_size);

ucc_status_t ucc_tl_shm_bcast_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task_h);

void ucc_tl_shm_bcast_copy_out(ucc_tl_shm_task_t *task, size_t data_size);

ucc_status_t ucc_tl_shm_bcast_check_read_ready(ucc_tl_shm_task_t *task);

#endif
