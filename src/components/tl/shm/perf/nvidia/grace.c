/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

/* Tuning for Grace generic, i.e. Grace but when n_groups and group sizes don't match any of the below perf keys */
TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_RR,
                        /*bcast base tree only inline*/0,
                        /*bcast base tree radix inline */64,
                        /*bcast top tree radix inline */4,
                        /*bcast_alg large*/BCAST_WR,
                        /*bcast base tree only large*/0,
                        /*bcast base tree radix large */64,
                        /*bcast top tree radix large */4,
                        /*reduce base tree only inline*/0,
                        /*reduce base tree radix inline*/16,
                        /*reduce top tree radix inline*/8,
                        /*reduce base tree only large*/0,
                        /*reduce base tree radix large*/8,
                        /*reduce top tree radix large*/4,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups UNUSED for generic*/0,
                        /*data seg size*/8192,
                        /*group size UNUSED for generic*/0);

/* Tuning for Grace CG (single socket) */
TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_1_16,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/1,
                        /*bcast base tree radix inline */64,
                        /*bcast top tree radix inline */0,
                        /*bcast_alg large*/BCAST_WR,
                        /*bcast base tree only large*/1,
                        /*bcast base tree radix large */32,
                        /*bcast top tree radix large */0,
                        /*reduce base tree only inline*/1,
                        /*reduce base tree radix inline*/64,
                        /*reduce top tree radix inline*/0,
                        /*reduce base tree only large*/1,
                        /*reduce base tree radix large*/4,
                        /*reduce top tree radix large*/0,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/1,
                        /*data seg size*/8192,
                        /*group size*/16);


TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_1_32,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/1,
                        /*bcast base tree radix inline */512,
                        /*bcast top tree radix inline */0,
                        /*bcast_alg large*/BCAST_WR,
                        /*bcast base tree only large*/1,
                        /*bcast base tree radix large */32,
                        /*bcast top tree radix large */0,
                        /*reduce base tree only inline*/1,
                        /*reduce base tree radix inline*/64,
                        /*reduce top tree radix inline*/0,
                        /*reduce base tree only large*/1,
                        /*reduce base tree radix large*/8,
                        /*reduce top tree radix large*/0,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/1,
                        /*data seg size*/8192,
                        /*group size*/32);

TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_1_64,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_RR,
                        /*bcast base tree only inline*/1,
                        /*bcast base tree radix inline */512,
                        /*bcast top tree radix inline */0,
                        /*bcast_alg large*/BCAST_RR,
                        /*bcast base tree only large*/1,
                        /*bcast base tree radix large */512,
                        /*bcast top tree radix large */0,
                        /*reduce base tree only inline*/1,
                        /*reduce base tree radix inline*/8,
                        /*reduce top tree radix inline*/0,
                        /*reduce base tree only large*/1,
                        /*reduce base tree radix large*/8,
                        /*reduce top tree radix large*/0,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/1,
                        /*data seg size*/8192,
                        /*group size*/64);

TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_1_72,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/1,
                        /*bcast base tree radix inline */8,
                        /*bcast top tree radix inline */0,
                        /*bcast_alg large*/BCAST_RR,
                        /*bcast base tree only large*/1,
                        /*bcast base tree radix large */16,
                        /*bcast top tree radix large */0,
                        /*reduce base tree only inline*/1,
                        /*reduce base tree radix inline*/4,
                        /*reduce top tree radix inline*/0,
                        /*reduce base tree only large*/1,
                        /*reduce base tree radix large*/512,
                        /*reduce top tree radix large*/0,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/1,
                        /*data seg size*/8192,
                        /*group size*/72);


/* Tuning for Grace C2 (2 sockets) */
TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_2_16,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/0,
                        /*bcast base tree radix inline */64,
                        /*bcast top tree radix inline */8,
                        /*bcast_alg large*/BCAST_WR,
                        /*bcast base tree only large*/0,
                        /*bcast base tree radix large */16,
                        /*bcast top tree radix large */16,
                        /*reduce base tree only inline*/0,
                        /*reduce base tree radix inline*/16,
                        /*reduce top tree radix inline*/64,
                        /*reduce base tree only large*/0,
                        /*reduce base tree radix large*/4,
                        /*reduce top tree radix large*/32,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/2,
                        /*data seg size*/8192,
                        /*group sizes*/16, 16);

TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_2_32,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/0,
                        /*bcast base tree radix inline */4,
                        /*bcast top tree radix inline */4,
                        /*bcast_alg large*/BCAST_RR,
                        /*bcast base tree only large*/0,
                        /*bcast base tree radix large */16,
                        /*bcast top tree radix large */16,
                        /*reduce base tree only inline*/0,
                        /*reduce base tree radix inline*/4,
                        /*reduce top tree radix inline*/4,
                        /*reduce base tree only large*/0,
                        /*reduce base tree radix large*/4,
                        /*reduce top tree radix large*/32,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/2,
                        /*data seg size*/8192,
                        /*group sizes*/32, 32);

TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_2_64,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/0,
                        /*bcast base tree radix inline */64,
                        /*bcast top tree radix inline */4,
                        /*bcast_alg large*/BCAST_RR,
                        /*bcast base tree only large*/0,
                        /*bcast base tree radix large */64,
                        /*bcast top tree radix large */8,
                        /*reduce base tree only inline*/0,
                        /*reduce base tree radix inline*/32,
                        /*reduce top tree radix inline*/4,
                        /*reduce base tree only large*/0,
                        /*reduce base tree radix large*/8,
                        /*reduce top tree radix large*/32,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/2,
                        /*data seg size*/8192,
                        /*group sizes*/64, 64);

TL_SHM_PERF_KEY_DECLARE(/*name*/nvidia_grace_2_72,
                        /*vendor*/NVIDIA,
                        /*model*/GRACE,
                        /*bcast_alg inline*/BCAST_WR,
                        /*bcast base tree only inline*/0,
                        /*bcast base tree radix inline */64,
                        /*bcast top tree radix inline */4,
                        /*bcast_alg large*/BCAST_WR,
                        /*bcast base tree only large*/0,
                        /*bcast base tree radix large */64,
                        /*bcast top tree radix large */4,
                        /*reduce base tree only inline*/0,
                        /*reduce base tree radix inline*/16,
                        /*reduce top tree radix inline*/4,
                        /*reduce base tree only large*/0,
                        /*reduce base tree radix large*/4,
                        /*reduce top tree radix large*/4,
                        /*layout*/SEG_LAYOUT_CONTIG,
                        /*n_groups*/2,
                        /*data seg size*/8192,
                        /*group sizes*/72, 72);
