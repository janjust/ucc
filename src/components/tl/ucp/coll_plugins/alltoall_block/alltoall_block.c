/**
 * Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2009-2012, Oak Ridge National Laboratory. All rights reserved.
 *
 * See file LICENSE for terms.
 */

/* Blocked AllToAll Algorithm */
/* ========================== */
/* The AllToAll matrix consists of rows and columns. The number of rows and */
/* the number columns is equal. */
/* Each row contains the data of a single rank (that its rank ID is the */
/* same as the row index) that should be sent during the AllToAll collective */
/* operation to all other ranks (each rank should get the correct) segment */
/* of the data. Therefore the rows can be called SRs (Source Ranks). */
/* Each column of the AllToAll matrix contains the data of a single rank */
/* (that its rank ID is the same as the column index), that the rank will */
/* have in the end of the AllToAll collective operation. Therefore the column */
/* can be called DRs (Destination Ranks). */

/* The Blocked AllToAll algorithm process the AllToAll collective operation */
/* by dividing the AllToAll matrix into equal blocks, by getting as an */
/* input for the algorithm the amount of SRs and DRs that should be in every */
/* block. In case the AllToAll matrix cannot be divided into block of equal */
/* size, because the number of DRs or SRs cannot be divided without a */
/* reminder, the last DRs and SRs will be divided into blocks that are */
/* smaller than requested. The matrix that is formed after dividing the */
/* AllToAll matrix into blocks is called "Blocked AllToAll" matrix. */

/* Each block in the Blocked AllToAll matrix has a rank that is called */
/* Aggregation Send Rank (ASR) and this rank is responsible for collecting */
/* (aggregating) the data from all the SRs (the rows) of the block and then */
/* spread the data between all the DRs (the columns) of the block in a way */
/* that each DR get the data it should get from the SRs in the AllToAll */
/* collective operation */
/* Notice that when the ASR collects the data from the SRs it gets the data */
/* in "row-view" of the blocks row, but when the ASR wants to spread the */
/* data to the DRs, it spreads it as in "column-view". In order to optimize */
/* the process of spreading the data to the DRs, the ASR transposes the data */
/* it got from the SRs and only then distributes it to the DRs. Note that */
/* after the transpose operation, the data stored in the memory in a */
/* convenient form to distribute. It can be noticed that all the blocks can */
/* work in parallel.*/

/* Note: Each rank can calculate all the information regarding all the blocks,*/
/* and their parameters (like amount of SRs, amount of SRs, ASR etc.) */

/* Assignment of ASR to Block */
/* ===========================*/
/* The algorithm tries to assign different ASRs for all the blocks, but in */
/* case there are more blocks than ranks, then a single rank can serve as */
/* an ASR of several blocks. */
/* The assignment of ASR to block is done as follows: */
/* Each block within a single row of blocks in the Blocked AllToAll matrix */
/* will be assigned with an ASR that is chosen from the SRs of the block. */
/* This way, all the blocks within a single row in the Blocked AllToAll */
/* matrix share the same "pool" of possible ASRs. The first block (column */
/* index 0) of every row will get the first SR (smallest rank) and the */
/* following block get the next SR. In case there are more blocks in a row */
/* than SRs, than the assignment will begin again from the first SR. */
/* Note: this distribution guarantees that all the SRs that serve as */
/* ASRs will be ASRs of blocks that have the same amount of SRs. */

#include "config.h"
#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"
#include "coll_score/ucc_coll_score.h"
#include "components/tl/ucp/tl_ucp_sendrecv.h"

#define UCC_TLCP_UCP_ALLTOALL_BLOCK_SCORE 1

#define N_INSTANCES_CEIL(total_len, segment_len) ((total_len)+(segment_len)-1)/(segment_len)
#define N_INSTANCES_FLOOR(total_len, segment_len) (total_len)/(segment_len)
#define WRAPAROUND_DOWN(value, max_val) (value)%(max_val)
#define WRAPAROUND_UP(value, max_val) (value+max_val)%(max_val)

/* For details please refer to the code during init stage when buffers */
/* are allocated. */
#define ASR_RECV_BUF(_task_data, _i) PTR_OFFSET(((_task_data)->asr_memory->addr), ((_task_data)->asr_buf_len) * (_i))
#define ASR_SEND_BUF(_task_data, _i) PTR_OFFSET(((_task_data)->asr_memory->addr), ((_task_data)->asr_buf_len) * ((_i) + ((_task_data)->asr_cycles)))

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_alltoall_block;

typedef struct ucc_tlcp_ucp_alltoall_block_config {
    char *score_str;
    unsigned long alltoall_block_src_block_size;
    unsigned long alltoall_block_dst_block_size;
    unsigned long alltoall_block_max_send;
    unsigned long alltoall_block_max_recv;
} ucc_tlcp_ucp_alltoall_block_config_t;

typedef struct ucc_tlcp_ucp_alltoall_block_task_data {
    /* Holds the maximal number of outstanding send requests that a process */
    /* manages in all times */
    ucc_rank_t max_send;

    /* Holds the maximal number of outstanding receive requests that a */
    /* process manages in all times */
    ucc_rank_t max_recv;

    /* The size in bytes of each element of data that is being sent from */
    /* each rank to every other rank during the AllToAll operation */
    size_t data_size;

    /* The amount of elements (each element with size data_size) will be */
    /* sent from each rank to every other rank during the AllToAll operation*/
    size_t data_count;

    /* Holds the amount of rows in the Blocked AllToAll matrix */
    int n_src_blocks;

    /* Holds the amount of columns in the Blocked AllToAll matrix */
    int n_dst_blocks;

    /* Holds the row index of the block that contains this rank as SR */
    int src_block_id;

    /* Holds the column index of the block that contains this rank as DR */
    int dst_block_id;

    /* The amount of AllToAll matrix rows within a single block in the */
    /* Blocked AllToAll matrix which is the matrix that is formed after */
    /* dividing the AllToAll matrix into blocks. This is the required size */
    /* that the algorithm gets as an input. */
    /* This also can be thought as the number of SRs (Source Ranks) that will */
    /* send data to the ASR (Aggregation Send Rank) of the block. */
    int src_block_size;

    /* The amount of AllToAll matrix columns within a single block in the */
    /* Blocked AllToAll matrix which is the matrix that is formed after */
    /* dividing the AllToAll matrix into blocks. This is the required size */
    /* that the algorithm gets as an input. */
    /* This also can be thought as the number of DRs (Destination Ranks) */
    /* that the will receive the transposed data to their receive buffer */
    /* from the ASR */
    int dst_block_size;

    /* Holds the number of AllToAll matrix rows in the last row of the */
    /* Blocked AllToAll matrix. In case the rows of AllToAll matrix cannot */
    /* be divided to the number of blocks that are required, the last block */
    /* will contain less rows. */
    int last_src_block_size;

    /* Holds the number of AllToAll matrix columns in the last column of the */
    /* Blocked AllToAll matrix. In case the columns of AllToAll matrix */
    /* cannot be divided to the number of blocks that are required, the last */
    /* block will contain less columns. */
    int last_dst_block_size;

    /* Holds the number of rows in the block that contains this rank as SR */
    int my_src_block_size;

    /* Holds the number of columns in the block that contains this rank as */
    /* DR */
    int my_dst_block_size;

    /* The length (in bytes) of the buffer that each rank that serves as an */
    /* ASR (Send Aggregator Rank) will use for the aggregation phase. */
    /* For each time that the rank serves as an ASR, it will have different */
    /* buffer with this size */
    size_t asr_buf_len;

    /* Number of times that a rank has to serve as an ASR of a block */
    int asr_cycles;

    /* Information that the ranks saves in case it cannot proceed the */
    /* algorithm and that will be restored after the rank will continue */

    /* Current phase in which the algorithm is located for the current rank */
    int phase;

    /* The amount outstanding send request that currently the rank manages */
    int outstanding_sends;

    /* The amount outstanding receive request that currently the rank */
    /* manages */
    int outstanding_recvs;

    /* Holds how many times till now the rank has served as an ASR. */
    /* For each time the rank serves as a ASR, it uses a different buffer */
    /* so all the blocks under this rank responsibilty as an ASR could be */
    /* executed in parallel */
    int asr_cycle;

    int i_dst_block;
    int i;
    int count;
    ucc_mc_buffer_header_t *asr_memory;
} ucc_tlcp_ucp_alltoall_block_task_data_t;

enum {
    PHASE_00,
    PHASE_01,
    PHASE_02,
    PHASE_03,
    PHASE_04,
    PHASE_05,
    PHASE_06,
};

#define CONFIG(_lib) ((ucc_tlcp_ucp_alltoall_block_config_t*)((_lib)->tlcp_configs[ucc_tlcp_ucp_alltoall_block.id]))

#define CHECK_PHASE(_p) case _p: goto _p; break;

#define GOTO_PHASE(_phase) do {                                                \
    switch (_phase) {                                                          \
        CHECK_PHASE(PHASE_00);                                                 \
        CHECK_PHASE(PHASE_01);                                                 \
        CHECK_PHASE(PHASE_02);                                                 \
        CHECK_PHASE(PHASE_03);                                                 \
        CHECK_PHASE(PHASE_04);                                                 \
        CHECK_PHASE(PHASE_05);                                                 \
        CHECK_PHASE(PHASE_06);                                                 \
    };                                                                         \
} while(0)


#define SAVE_STATE(_phase) do {                                                \
    task_data->phase     = _phase;                                             \
    task_data->i         = i;                                                  \
    task_data->asr_cycle = asr_cycle;                                          \
} while(0)

#define RESTORE_STATE() do {                                                   \
    i_dst_block = task_data->i_dst_block;                                      \
    asr_cycle   = task_data->asr_cycle;                                        \
    count       = task_data->count;                                            \
    i           = task_data->i;                                                \
} while(0)

static ucc_config_field_t ucc_tlcp_ucp_alltoall_block_table[] = {
    {"TLCP_ALLTOALL_BLOCK_TUNE", "", "Collective score modifier",
     ucc_offsetof(ucc_tlcp_ucp_alltoall_block_config_t, score_str), UCC_CONFIG_TYPE_STRING},

    {"TLCP_ALLTOALL_BLOCK_SRC_BLOCK_SIZE", "2",
     "The amount of alltoall matrix rows within a single block",
     ucc_offsetof(ucc_tlcp_ucp_alltoall_block_config_t, alltoall_block_src_block_size),
     UCC_CONFIG_TYPE_ULUNITS},

    {"TLCP_ALLTOALL_BLOCK_DST_BLOCK_SIZE", "2",
     "The amount of alltoall matrix columns within a single block",
     ucc_offsetof(ucc_tlcp_ucp_alltoall_block_config_t, alltoall_block_dst_block_size),
     UCC_CONFIG_TYPE_ULUNITS},

    {"TLCP_ALLTOALL_BLOCK_MAX_SEND", "2",
     "Holds the maximal number of outstanding send requests that a process "
     "manages in all times",
     ucc_offsetof(ucc_tlcp_ucp_alltoall_block_config_t, alltoall_block_max_send),
     UCC_CONFIG_TYPE_ULUNITS},

    {"TLCP_ALLTOALL_BLOCK_MAX_RECV", "2",
     "Holds the maximal number of outstanding receive requests that a  process "
     "manages in all times",
     ucc_offsetof(ucc_tlcp_ucp_alltoall_block_config_t, alltoall_block_max_recv),
     UCC_CONFIG_TYPE_ULUNITS},

    {NULL}};

static ucs_config_global_list_entry_t ucc_tlcp_ucp_alltoall_block_cfg_entry =
{
    .name   = "TLCP_ALLTOALL_BLOCK",
    .prefix = "TL_UCP_",
    .table  = ucc_tlcp_ucp_alltoall_block_table,
    .size   = sizeof(ucc_tlcp_ucp_alltoall_block_config_t)
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_tlcp_ucp_alltoall_block_cfg_entry,
                                &ucc_config_global_list);

void ucc_tlcp_ucp_alltoall_block_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args       = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team       = TASK_TEAM(task);
    ucc_rank_t             trank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             tsize      = UCC_TL_TEAM_SIZE(team);
    size_t                 data_size  = ucc_dt_size(args->src.info.datatype);
    size_t                 data_count = args->src.info.count / tsize;
    ucc_memory_type_t      mt         = args->src.info.mem_type;
    ucc_tlcp_ucp_alltoall_block_task_data_t *task_data;
    ucc_rank_t i_dst_block, i, j, asr_rank, asr_cycle, n_ranks, dst, src, n_sends;
    void *buf;
    size_t count, len;
    int i_am_asr;

    task_data = (ucc_tlcp_ucp_alltoall_block_task_data_t*)&task->plugin_data;
    RESTORE_STATE();
    GOTO_PHASE(task_data->phase);
PHASE_00:
    /* 1. Send Aggregation Phase */
    /* During this phase, each rank serves as an ASR for the local block. */
    /* Loop through the destination direction */

    /* Going through all the ASRs */
    for (i_dst_block = 0; i_dst_block < task_data->n_dst_blocks; i_dst_block++) {
PHASE_01:
        /* The i-th cycle for the ASR */
        asr_cycle = N_INSTANCES_FLOOR(i_dst_block, task_data->my_src_block_size);

        /* Calculate the ASR of the block with row index */
        /* task_data->src_block_id and column index i_dst_block */
        asr_rank = (task_data->src_block_id * task_data->src_block_size) +
                   (i_dst_block % task_data->my_src_block_size);
        i_am_asr = (trank == asr_rank) ? 1 : 0;

        /* Number DRs in the current block */
        n_ranks = (i_dst_block != task_data->n_dst_blocks - 1) ?
            task_data->dst_block_size : task_data->last_dst_block_size;

        /* real data size to be send and receive */
        count = data_count * data_size * n_ranks;

        /* SR sends data to the ASR */
        buf = PTR_OFFSET(args->src.info.buffer, data_count * data_size * task_data->dst_block_size * i_dst_block);
        dst = asr_rank;
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buf, count, mt, dst, team, task),
                      task, out);

        /* aggregator receives */
        if (i_am_asr) {
            for (i = 0; i < task_data->my_src_block_size; i++) {
PHASE_02:
                buf = PTR_OFFSET(ASR_RECV_BUF(task_data, asr_cycle), data_count * data_size * task_data->dst_block_size * i);
                src = task_data->src_block_id * task_data->src_block_size + i;
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(buf, count, mt, src, team, task),
                              task, out);
            }
        }
    }
PHASE_03:
    /* Check that all the send/recv operation were finished before */
    /* continuing to the next phase of the operation. */
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(PHASE_03);
        return;
    }
    /* 2. Transpose The Data In the ASR Phase */
    /* The purpose of the phase is to make the send buffers contiguous */
    /* during the distribution phase */
    /* TODO: Potential optimization with an O(n) algorithm (in-shuffle problem) */
    for (asr_cycle = 0; asr_cycle < task_data->asr_cycles; asr_cycle++) {
        len = data_size * data_count;

        /* TODO: need to cut down cost on unnecessary memcpy() due to empty cycles */
        for (i = 0; i < task_data->my_src_block_size; i++) {
            for (j = 0; j < task_data->dst_block_size; j++) {
                memcpy(PTR_OFFSET(ASR_SEND_BUF(task_data, asr_cycle), len * (j * task_data->my_src_block_size + i)),
                       PTR_OFFSET(ASR_RECV_BUF(task_data, asr_cycle), len * (i * task_data->dst_block_size + j)),
                       len);
            }
        }
    }

    /* 3. Distribution Phase */

    /* Going through all the ASRs */
    for (i_dst_block = 0; i_dst_block < task_data->n_dst_blocks; i_dst_block++) {
        /* The i-th cycle for the ASR */
        asr_cycle = N_INSTANCES_FLOOR(i_dst_block, task_data->my_src_block_size);

        /* Calculate the ASR of the block with row index */
        /* task_data->src_block_id and column index i_dst_block */
        asr_rank = (task_data->src_block_id * task_data->src_block_size) +
                   (i_dst_block % task_data->my_src_block_size);
        i_am_asr = (trank == asr_rank) ? 1 : 0;

        /* ASR sends data to DR directly */
        if (i_am_asr) {
            i = 0;
PHASE_04:
            n_sends = (i_dst_block != task_data->n_dst_blocks - 1) ?
                task_data->dst_block_size : task_data->last_dst_block_size;
            /* The number of elements that the ASR will send to each DR is */
            /* the total amount of elements it got from all the SRs. */
            /* Multiplying it by the number of bytes within each element */
            /* results in the size in bytes of each transaction. */
            count = task_data->my_src_block_size * data_count * data_size;

            for (; i < n_sends; i++) {
                /* for the i-th cycle, use the i-th buffer */
                buf = PTR_OFFSET(ASR_SEND_BUF(task_data, asr_cycle), data_count * data_size * task_data->my_src_block_size * i);
                dst = (asr_cycle * task_data->my_src_block_size + trank % task_data->src_block_size) * task_data->dst_block_size + i;
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buf, count, mt, dst, team, task),
                              task, out);
            }
        }
        /* DR Receives data from ASR */
        if (task_data->dst_block_id == i_dst_block) {
            for (i = 0; i < task_data->n_src_blocks; i++) {
PHASE_05:
                /* Number of ranks that sent the ASR data from the DR */
                n_ranks = (i != task_data->n_src_blocks - 1) ?
                    task_data->src_block_size : task_data->last_src_block_size;

                /* Real data size to be received in bytes */
                count = data_count * data_size * n_ranks;
                buf = PTR_OFFSET(args->dst.info.buffer, data_count * data_size * task_data->src_block_size * i);
                src = (i != task_data->n_src_blocks - 1) ?
                    i * task_data->src_block_size + task_data->dst_block_id % task_data->src_block_size :
                    i * task_data->src_block_size + task_data->dst_block_id % task_data->last_src_block_size;
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(buf, count, mt, src, team, task),
                              task, out);
            }
        }
    }
PHASE_06:
    /* Check that all the send/recv operation were finished before finishing */
    /* the algorithm. */
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(PHASE_06);
        return;
    }

    task->super.status = UCC_OK;
out:
    return;
}

ucc_status_t ucc_tlcp_ucp_alltoall_block_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_tlcp_ucp_alltoall_block_task_data_t *task_data;

    task_data = (ucc_tlcp_ucp_alltoall_block_task_data_t*)&task->plugin_data;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task_data->phase               = PHASE_00;

    ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

    return UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_alltoall_block_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tlcp_ucp_alltoall_block_task_data_t *task_data;

    task_data = (ucc_tlcp_ucp_alltoall_block_task_data_t*)&task->plugin_data;
    ucc_mc_free(task_data->asr_memory);

    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tlcp_ucp_alltoall_block_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *team,
                                              ucc_coll_task_t **task_h)
{
    UCC_STATIC_ASSERT(sizeof(ucc_tlcp_ucp_alltoall_block_task_data_t) <=
                      UCC_TL_UCP_TASK_PLUGIN_MAX_DATA);
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_coll_args_t   *args    = &coll_args->args;
    ucc_tl_ucp_lib_t  *lib     = UCC_TL_UCP_TEAM_LIB(tl_team);
    ucc_rank_t         trank   = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         tsize   = UCC_TL_TEAM_SIZE(tl_team);
    ucc_tlcp_ucp_alltoall_block_task_data_t *task_data;
    size_t asr_memory_size;
    ucc_status_t status;
    ucc_tl_ucp_task_t *task;

    if (UCC_IS_INPLACE(coll_args->args) ||
        (coll_args->args.src.info.mem_type != UCC_MEMORY_TYPE_HOST)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (task == NULL) {
        return UCC_ERR_NO_MEMORY;
    }

    task_data = (ucc_tlcp_ucp_alltoall_block_task_data_t*)&task->plugin_data;
    memset(task_data, 0, sizeof(ucc_tlcp_ucp_alltoall_block_task_data_t));
    task_data->data_size           = ucc_dt_size(args->src.info.datatype);
    task_data->data_count          = args->src.info.count / tsize;
    task_data->src_block_size      = CONFIG(lib)->alltoall_block_src_block_size;
    task_data->dst_block_size      = CONFIG(lib)->alltoall_block_dst_block_size;
    task_data->max_send            = CONFIG(lib)->alltoall_block_max_send;
    task_data->max_recv            = CONFIG(lib)->alltoall_block_max_recv;
    task_data->n_src_blocks        = N_INSTANCES_CEIL(tsize, task_data->src_block_size);
    task_data->n_dst_blocks        = N_INSTANCES_CEIL(tsize, task_data->dst_block_size);
    task_data->src_block_id        = trank / task_data->src_block_size;
    task_data->dst_block_id        = trank / task_data->dst_block_size;
    task_data->last_src_block_size = (tsize % task_data->src_block_size != 0) ?
        tsize % task_data->src_block_size : task_data->src_block_size;
    task_data->last_dst_block_size = (tsize % task_data->dst_block_size != 0) ?
        tsize % task_data->dst_block_size : task_data->dst_block_size;
    task_data->my_src_block_size   = (task_data->src_block_id != task_data->n_src_blocks - 1) ?
        task_data->src_block_size : task_data->last_src_block_size;
    task_data->my_dst_block_size   = (task_data->dst_block_id != task_data->n_dst_blocks - 1) ?
        task_data->dst_block_size : task_data->last_dst_block_size;
    /* The amount of blocks that will have this rank as theirs ASR */
    task_data->asr_cycles          = N_INSTANCES_CEIL(task_data->n_dst_blocks, task_data->my_src_block_size);
    task_data->asr_buf_len         = task_data->data_size * task_data->data_count *
                                     task_data->dst_block_size * task_data->src_block_size;
    /* Each rank must allocated enough memory to hold the context */
    /* information of the AllToAll algorithm during the run. The context */
    /* must hold the following information: */
    /*      "ASR Memory": Memory that will be used for all the communication */
    /*      the rank will do as an ASR. Notice that a rank that serves as an */
    /*      ASR needs enough memory to store the data it receives from other */
    /*      ranks and because the rank has to transpose the data (which does */
    /*      not occur in place) it has to allocate another buffer for it,  */
    /*      thus allocate twice the size that was initially needed. */
    asr_memory_size = 2*task_data->asr_cycles * task_data->asr_buf_len;
    status = ucc_mc_alloc(&task_data->asr_memory, asr_memory_size,
                          UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(lib, "failed to allocate memory for asr buffer");
        return status;
    }

    task->super.finalize = ucc_tlcp_ucp_alltoall_block_finalize;
    task->super.post     = ucc_tlcp_ucp_alltoall_block_start;
    task->super.progress = ucc_tlcp_ucp_alltoall_block_progress;
    *task_h              = &task->super;

    return UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_alltoall_block_get_scores(ucc_base_team_t *tl_team,
                                                    ucc_coll_score_t **score_p)
{
    ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_lib_t  *lib  = UCC_TL_UCP_TEAM_LIB(team);
    ucc_memory_type_t  mt   = UCC_MEMORY_TYPE_HOST;

    ucc_coll_score_team_info_t  team_info;
    const char                 *score_str;
    ucc_coll_score_t           *score;
    ucc_status_t                status;

    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TLCP_UCP_ALLTOALL_BLOCK_SCORE;
    team_info.init                = NULL;
    team_info.num_mem_types       = 1;
    team_info.supported_mem_types = &mt;
    team_info.supported_colls     = UCC_COLL_TYPE_ALLTOALL;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score");
        return status;
    }
    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLTOALL,
                                      UCC_MEMORY_TYPE_HOST,
                                      0, 4096, UCC_TLCP_UCP_ALLTOALL_BLOCK_SCORE,
                                      ucc_tlcp_ucp_alltoall_block_init,
                                      tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range");
        return status;
    }
    score_str = CONFIG(lib)->score_str;
    if (strlen(score_str) > 0) {
        status = ucc_coll_score_update_from_str(score_str, &team_info,
                                                &team->super.super, score);
        if (status == UCC_ERR_INVALID_PARAM) {
            /* User provided incorrect input - try to proceed */
            status = UCC_OK;
        }
    }
    *score_p = score;
    return status;
}

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_alltoall_block = {
    .super.name   = "tl_ucp_alltoall_block",
    .super.score  = UCC_TLCP_UCP_ALLTOALL_BLOCK_SCORE,
    .config.table = ucc_tlcp_ucp_alltoall_block_table,
    .config.size  = sizeof(ucc_tlcp_ucp_alltoall_block_config_t),
    .get_scores   = ucc_tlcp_ucp_alltoall_block_get_scores
};
