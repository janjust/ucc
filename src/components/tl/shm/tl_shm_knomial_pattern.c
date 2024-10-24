/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm_knomial_pattern.h"

void ucc_tl_shm_kn_tree_init(ucc_rank_t      size,      /* group size */
                             ucc_rank_t      root,      /* root of bcast*/
                             ucc_rank_t      rank,      /* calling rank */
                             ucc_rank_t      radix,     /* kn radix */
                             ucc_coll_type_t coll_type, /* bcast/reduce */
                             ucc_kn_tree_t * tree_p /* output tree */)
{
    int         pos, i;
    ucc_rank_t  peer, vpeer, vparent;
    ucc_rank_t *peer_ptr;
    ucc_rank_t  dist       = 1;
    ucc_rank_t  vrank      = (rank - root + size) % size;
    int         n_children = 0, calc_parent = 0;

    radix = ucc_min(radix, size);

    if (coll_type == UCC_COLL_TYPE_BCAST ||
        coll_type == UCC_COLL_TYPE_FANOUT) {
        while (dist < size) {
            dist *= radix;
        }
    }

    while (CONDITION(dist, size, coll_type)) {
        if (vrank % dist == 0) {
            pos = (vrank / dist) % radix;
            if (pos == 0) {
                for (i = 1; i < radix; i++) {
                    vpeer = vrank + i * dist;
                    if (vpeer < size) {
                        peer     = (vpeer + root) % size;
                        peer_ptr = (ucc_rank_t *)PTR_OFFSET(
                            &tree_p->children[0],
                            sizeof(ucc_rank_t) * n_children);
                        *peer_ptr = peer;
                        n_children++;
                    } else if (coll_type == UCC_COLL_TYPE_REDUCE ||
                               coll_type == UCC_COLL_TYPE_FANIN) {
                        break;
                    }
                }
            } else if (!calc_parent) {
                vparent        = vrank - pos * dist;
                tree_p->parent = (vparent + root) % size;
                calc_parent    = 1;
            }
        }
        dist = (coll_type == UCC_COLL_TYPE_BCAST ||
                coll_type == UCC_COLL_TYPE_FANOUT) ? dist / radix
                                                   : dist * radix;
    }
    if (rank == root) {
        tree_p->parent = UCC_RANK_INVALID;
    }
    if (n_children == 0) {
        *tree_p->children = UCC_RANK_INVALID;
    }
    tree_p->n_children = n_children;
    tree_p->radix      = radix;
}
