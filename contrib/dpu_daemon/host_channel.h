/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef HOST_CHANNEL_H
#define HOST_CHANNEL_H

// #define _DEFAULT_SOURCE
#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <assert.h>
#include <pthread.h>

#include "server_ucc.h"
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#define MAX_THREADS         8
#define MAX_NUM_RANKS       128
#define MAX_RKEY_LEN        1024
#define IP_STRING_LEN       50
#define PORT_STRING_LEN     8
#define SUCCESS             0
#define ERROR               1
#define DEFAULT_PORT        12345

#define HOST_COMM_CHANNEL 0
#define HOST_DATA_CHANNEL 1

#define DPU_MIN(a,b) (((a)<(b))?(a):(b))
#define DPU_MAX(a,b) (((a)>(b))?(a):(b))

#ifdef NDEBUG
#define DPU_LOG(...)
#define CTX_LOG(...)
#else
#define DPU_LOG(_fmt, ...)                                  \
do {                                                        \
    fprintf(stderr, "%s:%d:%s(): " _fmt,                    \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__);   \
    fflush(stderr);                                                 \
} while (0)

#define CTX_LOG(_fmt, ...)                                          \
do {                                                                \
    fprintf(stderr, "[T%d] %s:%d:%s(): " _fmt,                       \
            ctx->idx, __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    fflush(stderr);                                                 \
} while (0)
#endif

typedef struct host_rkey_t {
    char    src_rkey_buf[MAX_RKEY_LEN];
    char    dst_rkey_buf[MAX_RKEY_LEN];
    size_t  src_rkey_len;
    size_t  dst_rkey_len;
    void   *src_buf;
    void   *dst_buf;
} host_rkey_t;

typedef struct buf_info_v_t {
    ucc_count_t counts[MAX_NUM_RANKS];
    ucc_count_t displs[MAX_NUM_RANKS];
} buf_info_v_t;

/* sync struct type
 * use it for counter, dtype, ar op, length */
typedef struct dpu_put_sync_t {
    host_rkey_t         rkeys;
    uint16_t            team_id;
    uint16_t            rail;
    uint16_t            dpu_per_node_cnt;
    uint16_t            create_new_team;
    uint16_t            num_ranks;
    ucc_rank_t          host_team_rank;
    ucc_rank_t          rank_list[MAX_NUM_RANKS];
    ucc_coll_args_t     coll_args;
    buf_info_v_t        src_v;
    buf_info_v_t        dst_v;
    volatile uint32_t   count_total;
    volatile uint32_t   coll_id;
} dpu_put_sync_t;

typedef struct dpu_get_sync_t {
    uint32_t  count_serviced;
    uint32_t  coll_id;
} dpu_get_sync_t;

typedef struct dpu_rkey_t {
    void    *rkey_addr;
    size_t  rkey_addr_len;
} dpu_rkey_t;

typedef struct dpu_mem_t {
    void *base;
    ucp_mem_h memh;
    dpu_rkey_t rkey;
} dpu_mem_t;

typedef struct dpu_mem_segs_t {
    dpu_mem_t sync;
    dpu_mem_t in;
    dpu_mem_t out;
} dpu_mem_segs_t;

typedef struct dpu_thread_sync_t {
    volatile int8_t todo[MAX_THREADS];
    volatile int8_t done[MAX_THREADS];
} dpu_thread_sync_t;

typedef enum dpu_ar_phase_t {
    WAIT,
    INIT,
    REDUCE,
    BCAST,
} dpu_ar_phase_t;

typedef enum dpu_buf_state_t {
    FREE,
    SENDRECV,
    REDUCING,
    IDLE,
} dpu_buf_state_t;

typedef struct dpu_buf_t {
    void              *buf;
    dpu_buf_state_t    state;
    ucs_status_ptr_t   ucp_req;
    size_t             count;
} dpu_buf_t;

typedef struct dpu_stage_t {
    dpu_buf_t accbuf;
    dpu_buf_t getbuf[2];
    
    dpu_ar_phase_t phase;
    int get_idx;
    int red_idx;
    int src_rank;
    int dst_rank;
    
    int done_get;
    int done_red;
    int done_put;
} dpu_stage_t;

typedef struct dpu_pipeline_t {
    size_t             buffer_size;
    size_t             num_buffers;

    dpu_stage_t        stages[1];
    size_t             my_count;
    size_t             my_offset;

    size_t             count_received;
    size_t             count_reduced;
    size_t             count_serviced;
} dpu_pipeline_t;

typedef struct dpu_hc_t {
    /* Comm or Data channel */
    int channel_type;

    /* TCP/IP stuff */
    char *hname;
    char *ip;
    int connfd, listenfd;
    uint16_t port;

    /* Local UCX stuff */
    ucp_context_h ucp_ctx;
    ucp_worker_h ucp_worker;
    ucp_worker_attr_t worker_attr;
    ucp_request_param_t req_param;
    dpu_mem_segs_t mem_segs;

    /* Remote UCX stuff */
    void *remote_addrs;
    void *rem_worker_addr;
    size_t rem_worker_addr_len;
    ucp_ep_h localhost_ep;
    uint64_t sync_addr;
    ucp_rkey_h src_rkey;
    ucp_rkey_h dst_rkey;

    /* pipeline buffer */
    dpu_pipeline_t  pipeline;

    /* remote eps */
    uint16_t world_team_id;
    uint32_t world_rank;
    uint32_t world_size;
    uint64_t team_rank;
    uint32_t team_size;
    ucp_ep_h *host_eps;
    ucp_ep_h *dpu_eps;
    host_rkey_t *host_rkeys;
    ucp_rkey_h *host_src_rkeys;
    ucp_rkey_h *host_dst_rkeys;

    /* Multi-rail support */
    int rail;
    int dpu_per_node_cnt;

    /* Global state */
    int job_id;
    int window_size;

    /* global visibility of collectives */
    dpu_put_sync_t *world_lsyncs;
} dpu_hc_t;

int dpu_hc_init(dpu_hc_t *dpu_hc);
int dpu_hc_accept_job(dpu_hc_t *hc);
int dpu_hc_connect_localhost_ep(dpu_hc_t *hc);
int dpu_hc_connect_remote_hosts(dpu_hc_t *hc, dpu_ucc_comm_t *comm);
int dpu_hc_reply(dpu_hc_t *hc, dpu_get_sync_t *coll_sync);
int dpu_hc_wait(dpu_hc_t *hc, unsigned int coll_id);
int dpu_hc_reset_pipeline(dpu_hc_t *hc);
int dpu_hc_reset_job(dpu_hc_t *dpu_hc);
int dpu_hc_finalize(dpu_hc_t *dpu_hc);

typedef struct thread_ctx_t {
    pthread_t       id;
    int             idx;
    int             nth;
    dpu_ucc_comm_t  *comm;
    dpu_hc_t        *hc;
    dpu_hc_t        *dc;
    dpu_get_sync_t  *coll_sync;
    dpu_thread_sync_t *thread_sync;
} thread_ctx_t;

/* thread accisble data - split reader/writer */
typedef struct thread_sync_t {
    volatile unsigned int todo[8];
    volatile unsigned int done[8];
    volatile dpu_buf_t *accbuf;
    volatile dpu_buf_t *getbuf;
} thread_sync_t;

ucs_status_t dpu_hc_issue_get(dpu_hc_t *hc, dpu_put_sync_t *sync, dpu_stage_t *stage, dpu_buf_t *getbuf, thread_ctx_t *ctx);
ucs_status_t dpu_hc_issue_put(dpu_hc_t *hc, dpu_put_sync_t *sync, dpu_stage_t *stage, dpu_buf_t *accbuf, thread_ctx_t *ctx);
ucs_status_t dpu_hc_local_reduce(dpu_hc_t *hc, dpu_put_sync_t *sync, thread_ctx_t *ctx, dpu_stage_t *stage, dpu_buf_t *accbuf, dpu_buf_t *getbuf);
ucs_status_t dpu_hc_progress_allreduce(dpu_hc_t *hc, dpu_put_sync_t *sync, thread_ctx_t *ctx);
ucs_status_t dpu_hc_issue_hangup(dpu_hc_t *dpu_hc, dpu_put_sync_t *sync, thread_ctx_t *ctx);
ucs_status_t dpu_send_init_completion(dpu_hc_t *hc);
ucs_status_t dpu_recv_world_team_id(dpu_hc_t *hc, dpu_ucc_comm_t *comm);

size_t dpu_ucc_dt_size(ucc_datatype_t dt);
int dpu_dc_create(thread_ctx_t *ctx, dpu_hc_t *hc, dpu_hc_t *dc);
int dpu_dc_reset(dpu_hc_t *dc);

void dpu_waitfor_comm_thread(thread_ctx_t *ctx, thread_sync_t *sync);
void dpu_signal_comm_thread(thread_ctx_t *ctx, thread_sync_t *sync);
void dpu_waitfor_comp_thread(thread_ctx_t *ctx, thread_sync_t *sync);
void dpu_signal_comp_thread(thread_ctx_t *ctx, thread_sync_t *sync);

ucs_status_t _dpu_request_wait(ucp_worker_h ucp_worker, ucs_status_ptr_t request);

ucc_rank_t dpu_get_world_rank(dpu_hc_t *hc,  int dpu_rank, int team_id, thread_ctx_t *ctx);
ucc_rank_t dpu_get_host_ep_rank(dpu_hc_t *hc,  int host_rank, int team_id, thread_ctx_t *ctx);
size_t dpu_ucc_dt_size(ucc_datatype_t dt);
    
#endif
