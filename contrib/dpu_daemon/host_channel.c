/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "host_channel.h"
#include <unistd.h>
#include <ucc/api/ucc.h>

#define UCC_DT_PREDEFINED_ID(_dt) ((_dt) >> UCC_DATATYPE_SHIFT)

#define UCC_DT_IS_PREDEFINED(_dt) \
    (((_dt) & UCC_DATATYPE_CLASS_MASK) == UCC_DATATYPE_PREDEFINED)

size_t ucc_dt_predefined_sizes[UCC_DT_PREDEFINED_LAST] = {
     [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]     = 1,
     [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]    = 1,
     [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)]    = 2,
     [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)]   = 2,
     [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)]  = 2,
     [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] = 2,
     [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)]    = 4,
     [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)]   = 4,
     [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)]  = 4,
     [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)]    = 8,
     [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)]   = 8,
     [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)]  = 8,
     [UCC_DT_PREDEFINED_ID(UCC_DT_INT128)]   = 16,
     [UCC_DT_PREDEFINED_ID(UCC_DT_UINT128)]  = 16,
};

size_t dpu_ucc_dt_size(ucc_datatype_t dt)
{
    if (UCC_DT_IS_PREDEFINED(dt)) {
        return ucc_dt_predefined_sizes[UCC_DT_PREDEFINED_ID(dt)];
    }
    return 0;
}

/* TODO: export ucc_mc.h */
ucc_status_t ucc_mc_reduce(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dtype,
                           ucc_reduction_op_t op, ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_reduce_multi(void *src1, void *src2, void *dst, size_t n_vectors,
                    size_t count, size_t stride, ucc_datatype_t dtype,
                    ucc_reduction_op_t op, ucc_memory_type_t mem_type);


static int _dpu_host_to_ip(dpu_hc_t *hc)
{
//     printf ("%s\n", __FUNCTION__);
    struct hostent *he;
    struct in_addr **addr_list;
    int i;

    hc->hname = calloc(1, 100 * sizeof(char));
    hc->ip = malloc(100 * sizeof(char));

    int ret = gethostname(hc->hname, 100);
    if (ret) {
        return 1;
    }

    if ( (he = gethostbyname( hc->hname ) ) == NULL)
    {
        // get the host info
        herror("gethostbyname");
        return 1;
    }

    addr_list = (struct in_addr **) he->h_addr_list;
    for(i = 0; addr_list[i] != NULL; i++)
    {
        //Return the first one;
        strcpy(hc->ip , inet_ntoa(*addr_list[i]) );
        return UCC_OK;
    }
    return UCC_ERR_NO_MESSAGE;
}

static int _dpu_listen(dpu_hc_t *hc)
{
    struct sockaddr_in serv_addr;

    if(_dpu_host_to_ip(hc)) {
        return UCC_ERR_NO_MESSAGE;
    }

    /* creates an UN-named socket inside the kernel and returns
     * an integer known as socket descriptor
     * This function takes domain/family as its first argument.
     * For Internet family of IPv4 addresses we use AF_INET
     */
    hc->listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (0 > hc->listenfd) {
        fprintf(stderr, "socket() failed (%s)\n", strerror(errno));
        goto err_ip;
    }
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(hc->port);

    /* The call to the function "bind()" assigns the details specified
     * in the structure 『serv_addr' to the socket created in the step above
     */
    if (0 > bind(hc->listenfd, (struct sockaddr*)&serv_addr,
                 sizeof(serv_addr))) {
        fprintf(stderr, "Failed to bind() (%s)\n", strerror(errno));
        goto err_sock;
    }

    /* The call to the function "listen()" with second argument as 10 specifies
     * maximum number of client connections that server will queue for this listening
     * socket.
     */
    if (0 > listen(hc->listenfd, 10)) {
        fprintf(stderr, "listen() failed (%s)\n", strerror(errno));
        goto err_sock;
    }

    return UCC_OK;
err_sock:
    close(hc->listenfd);
err_ip:
    free(hc->ip);
    free(hc->hname);
    return UCC_ERR_NO_MESSAGE;
}

static void _dpu_listen_cleanup(dpu_hc_t *hc)
{
    DPU_LOG("Cleaning up host channel\n");
    close(hc->listenfd);
    free(hc->ip);
    free(hc->hname);
}


ucc_status_t _dpu_req_test(ucs_status_ptr_t request)
{
    if (request == NULL) {
        return UCS_OK;
    }
    else if (UCS_PTR_IS_ERR(request)) {
        fprintf (stderr, "unable to complete UCX request\n");
        return UCS_PTR_STATUS(request);
    }
    else {
        return ucp_request_check_status(request);
    }
}

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf ("error handling callback was invoked with status %d (%s)\n",
            status, ucs_status_string(status));
}

static ucs_status_t _dpu_flush_host_eps(dpu_hc_t *hc)
{
    int i;
    ucp_request_param_t param = {};
    ucs_status_ptr_t request;

    for (i = 0; i < hc->world_size; i++) {
        request = ucp_ep_flush_nbx(hc->host_eps[i], &param);
        _dpu_request_wait(hc->ucp_worker, request);
    }
    return UCS_OK;
}

static ucs_status_t _dpu_worker_flush(dpu_hc_t *hc)
{
    ucp_request_param_t param = {};
    ucs_status_ptr_t request = ucp_worker_flush_nbx(hc->ucp_worker, &param);
    return _dpu_request_wait(hc->ucp_worker, request);
}

static int _dpu_ucx_init(dpu_hc_t *hc)
{
    int ret = UCC_OK;
    ucp_params_t ucp_params = {0};
    ucp_worker_params_t worker_params = {0};
    ucs_status_t status;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

    status = ucp_init(&ucp_params, NULL, &hc->ucp_ctx);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init(%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    /* FIXME: Although each thread owns its worker,
       multi-thread mode is necessary to prevent errors during finalize */
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create(hc->ucp_ctx, &worker_params, &hc->ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_cleanup;
    }

    hc->worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                                 UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    hc->worker_attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    status = ucp_worker_query (hc->ucp_worker, &hc->worker_attr);
    if (UCS_OK != status) {
        fprintf(stderr, "failed to ucp_worker_query (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_worker;
    }

    return ret;
err_worker:
    ucp_worker_destroy(hc->ucp_worker);
err_cleanup:
    ucp_cleanup(hc->ucp_ctx);
err:
    return ret;
}

static int _dpu_ucx_fini(dpu_hc_t *hc){
    ucp_worker_release_address(hc->ucp_worker, hc->worker_attr.address);
    ucp_worker_destroy(hc->ucp_worker);
    ucp_cleanup(hc->ucp_ctx);
}

static int _dpu_hc_buffer_alloc(dpu_hc_t *hc, dpu_mem_t *mem, size_t size)
{
    ucp_mem_map_params_t mem_params;
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;
    int ret = UCC_OK;

    memset(mem, 0, sizeof(*mem));
    mem->base = aligned_alloc(4096, size);
    if (mem->base == NULL) {
        fprintf(stderr, "failed to allocate %lu bytes base %p\n", size, mem->base);
        ret = UCC_ERR_NO_MEMORY;
        goto out;
    }

    memset(&mem_params, 0, sizeof(ucp_mem_map_params_t));
    mem_params.address = mem->base;
    mem_params.length = size;

    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_ADDRESS;

    status = ucp_mem_map(hc->ucp_ctx, &mem_params, &mem->memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_map (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_calloc;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;

    status = ucp_mem_query(mem->memh, &mem_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_query (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_map;
    }

    DPU_LOG("Requested to map base %p len %zu registered base %p len %zu\n",
            mem_params.address, mem_params.length, mem_attr.address, mem_attr.length);
    assert(mem_attr.length >= mem_params.length);
    assert(mem_attr.address == mem_params.address);

    status = ucp_rkey_pack(hc->ucp_ctx, mem->memh,
                           &mem->rkey.rkey_addr,
                           &mem->rkey.rkey_addr_len);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status));
        ret = UCC_ERR_NO_MESSAGE;
        goto err_map;
    }
    
    goto out;
err_map:
    ucp_mem_unmap(hc->ucp_ctx, mem->memh);
err_calloc:
    free(mem->base);
out:
    return ret;
}

static int _dpu_hc_buffer_free(dpu_hc_t *hc, dpu_mem_t *mem)
{
    ucp_rkey_buffer_release(mem->rkey.rkey_addr);
    ucp_mem_unmap(hc->ucp_ctx, mem->memh);
    free(mem->base);
}

static void _dpu_hc_reset_buf(dpu_buf_t *buf)
{
    buf->state = FREE;
    buf->count = 0;
    buf->bytes = 0;
    buf->ucp_req = NULL;
}

int dpu_hc_reset_pipeline(dpu_hc_t *hc)
{
    dpu_pipeline_t *pipe = &hc->pipeline;
    pipe->avail_buffs = pipe->num_buffers;
    pipe->my_count = pipe->my_offset = 0;
    pipe->get_idx = pipe->red_idx = 0;
    pipe->src_rank = pipe->dst_rank = hc->world_rank / hc->dpu_per_node_cnt;
    pipe->count_issued = pipe->count_received = 0;
    pipe->count_reduced = pipe->count_serviced = 0;

    _dpu_hc_reset_buf(&pipe->accbuf);
    for (int i=0; i<pipe->num_buffers; i++) {
        _dpu_hc_reset_buf(&pipe->getbuf[i]);
    }

    return 0;
}

static int _dpu_hc_init_pipeline(dpu_hc_t *hc)
{
    int i, ret;
    size_t buffer_size = hc->pipeline.buffer_size;
    size_t num_buffers = hc->pipeline.num_buffers;

    /* TODO: enforce limits */
    DPU_LOG("Init pipeline with %zu get buffers with size %zu\n",
            num_buffers, buffer_size);
    assert(buffer_size >= 4096);
    assert(num_buffers >= 2);
    assert(num_buffers <= 16);

    /* one extra for accbuf */
    ret = _dpu_hc_buffer_alloc(hc, &hc->mem_segs.in, buffer_size * (num_buffers + 1));
    if (ret) {
        goto out;
    }

    ret = _dpu_hc_buffer_alloc(hc, &hc->mem_segs.sync, sizeof(dpu_put_sync_t));
    if (ret) {
        goto err_put;
    }

    hc->pipeline.accbuf.buf = (char *)hc->mem_segs.in.base;
    for (i=0; i<num_buffers; i++) {
        hc->pipeline.getbuf[i].buf = (char *)hc->mem_segs.in.base + buffer_size * (i+1);
    }

    dpu_hc_reset_pipeline(hc);
    goto out;
err_put:
    _dpu_hc_buffer_free(hc, &hc->mem_segs.in);
out:
    return ret;
}

int dpu_hc_init(dpu_hc_t *hc)
{
    int ret = UCC_OK;
    // memset(hc, 0, sizeof(*hc));

    /* Start listening */
    ret = _dpu_listen(hc);
    if (ret) {
        goto err_ip;
    }
    goto out;

err_ip:
    _dpu_listen_cleanup(hc);
out:
    return ret;
}

static void dpu_coll_collect_host_addrs(dpu_ucc_comm_t *comm, void *addr, size_t addr_len, void *outbuf)
{
    ucs_status_t status;
    ucc_coll_req_h request;
    ucc_team_h team = comm->team;
    ucc_rank_t team_size = 0;
    UCC_CHECK(ucc_team_get_size(team, &team_size));
        
    ucc_coll_args_t coll = {
        .coll_type = UCC_COLL_TYPE_ALLGATHER,
        .src.info = {
            .buffer   = addr,
            .count    = addr_len,
            .datatype = UCC_DT_INT8,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
        .dst.info = {
            .buffer   = outbuf,
            .count    = addr_len * team_size,
            .datatype = UCC_DT_INT8,
            .mem_type = UCC_MEMORY_TYPE_HOST,
        },
    };

    DPU_LOG("Issue Allgather from ranks %d src %p dst %p bytes %lu\n",
            team_size, addr, outbuf, addr_len);
    UCC_CHECK(ucc_collective_init(&coll, &request, team));
    UCC_CHECK(ucc_collective_post(request));
    while (UCC_OK != ucc_collective_test(request)) {
        ucc_context_progress(comm->ctx);
    }
    UCC_CHECK(ucc_collective_finalize(request));
}

int dpu_hc_connect_localhost_ep(dpu_hc_t *hc)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    ep_params.field_mask    = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                              UCP_EP_PARAM_FIELD_ERR_HANDLER |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode		= UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb    = err_cb;
    ep_params.address = hc->rem_worker_addr;

    status = ucp_ep_create(hc->ucp_worker, &ep_params, &hc->localhost_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create endpoint on dpu to local host %d (%s)\n",
                status, ucs_status_string(status));
    }

    return status;
}

static ucs_status_t _dpu_create_remote_host_eps(dpu_hc_t *hc, dpu_ucc_comm_t *comm)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    int i;
    size_t rem_worker_addr_len = hc->rem_worker_addr_len;
    void *rem_worker_addr = hc->rem_worker_addr;

    DPU_LOG("Creating %u remote host eps, addr len %lu\n", hc->world_size, hc->rem_worker_addr_len);

    if (hc->channel_type == HOST_COMM_CHANNEL) {
        /* Allgather Host EP addresses */
        hc->remote_addrs = calloc(hc->world_size, rem_worker_addr_len);
        dpu_coll_collect_host_addrs(comm, rem_worker_addr, rem_worker_addr_len, hc->remote_addrs);
    } else {
        /* Connecting to existing workers,
           Reuse already collected EP addresses */
        assert(hc->remote_addrs != NULL);
    }

    ep_params.field_mask    = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                              UCP_EP_PARAM_FIELD_ERR_HANDLER |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode		= UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = err_cb;

    hc->host_eps = calloc(hc->world_size, sizeof(ucp_ep_h));
    /* Connect to all remote hosts */
    for (i = 0; i < hc->world_size; i++) {
        /* Skip Local Host for Comm channel, Already Connected */
        if (i == hc->world_rank && hc->channel_type == HOST_COMM_CHANNEL) {
            hc->host_eps[i] = hc->localhost_ep;
            continue;
        }

        ep_params.address = hc->remote_addrs + i * rem_worker_addr_len;
        status = ucp_ep_create(hc->ucp_worker, &ep_params, &hc->host_eps[i]);
        if (status != UCS_OK) {
            fprintf(stderr, "failed to create endpoint on dpu to host %d (%s)\n",
                    i, ucs_status_string(status));
            return UCC_ERR_NO_MESSAGE;
        }
    }

    hc->host_rkeys      = calloc(hc->world_size, sizeof(host_rkey_t));
    hc->host_src_rkeys  = calloc(hc->world_size, sizeof(ucp_rkey_h));
    hc->host_dst_rkeys  = calloc(hc->world_size, sizeof(ucp_rkey_h));
    hc->world_lsyncs    = calloc(hc->world_size, sizeof(dpu_put_sync_t));
    
    memset(&hc->req_param, 0, sizeof(hc->req_param));
    // hc->req_param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

    return UCC_OK;
}

static int _dpu_close_host_eps(dpu_hc_t *hc)
{
    ucp_request_param_t param;
    ucs_status_t status;
    ucs_status_ptr_t close_req;
    int ret = UCC_OK;
    int i;

    param.op_attr_mask  = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = UCP_EP_CLOSE_FLAG_FORCE;

    for (i = 0; i < hc->world_size; i++) {
        close_req = ucp_ep_close_nbx(hc->host_eps[i], &param);
        if (UCS_PTR_IS_PTR(close_req)) {
            do {
                ucp_worker_progress(hc->ucp_worker);
                status = ucp_request_check_status(close_req);
            } while (status == UCS_INPROGRESS);

            ucp_request_free(close_req);
        }
        else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
            fprintf(stderr, "failed to close ep %p\n", (void *)hc->host_eps[i]);
            ret = UCC_ERR_NO_MESSAGE;
        }
    }
    free(hc->host_eps);
    free(hc->host_rkeys);
    free(hc->host_src_rkeys);
    free(hc->host_dst_rkeys);
    return ret;
}

ucs_status_t _dpu_request_wait(ucp_worker_h ucp_worker, ucs_status_ptr_t request)
{
    ucs_status_t status;

    /* immediate completion */
    if (request == NULL) {
        return UCS_OK;
    }
    else if (UCS_PTR_IS_ERR(request)) {
        status = ucp_request_check_status(request);
        fprintf (stderr, "unable to complete UCX request (%s)\n", ucs_status_string(status));
        return UCS_PTR_STATUS(request);
    }
    else {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
    }

    return status;
}

int dpu_hc_accept_job(dpu_hc_t *hc)
{
    int ret;
    hc->job_id++;
    hc->channel_type = HOST_COMM_CHANNEL;

    /* init ucx ctx and worker */
    ret = _dpu_ucx_init(hc);
    if (ret) {
        goto err_ucx;
    }

    /* In the call to accept(), the server is put to sleep and when for an incoming
    * client request, the three way TCP handshake* is complete, the function accept()
    * wakes up and returns the socket descriptor representing the client socket.
    */
    DPU_LOG("Waiting for connection from Job Id %d at port %u\n", hc->job_id, hc->port);
    hc->connfd = accept(hc->listenfd, (struct sockaddr*)NULL, NULL);
    if (-1 == hc->connfd) {
        fprintf(stderr, "Error in accept (%s)!\n", strerror(errno));
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }
    DPU_LOG("Connection established from Job Id %d\n", hc->job_id);

    ret = send(hc->connfd, &hc->worker_attr.address_length, sizeof(size_t), 0);
    if (-1 == ret) {
        fprintf(stderr, "send worker_address_length failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = send(hc->connfd, hc->worker_attr.address, hc->worker_attr.address_length, 0);
    if (-1 == ret) {
        fprintf(stderr, "send worker_address failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = recv(hc->connfd, &hc->rem_worker_addr_len, sizeof(size_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv address_length failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    hc->rem_worker_addr = calloc(1, hc->rem_worker_addr_len);
    ret = recv(hc->connfd, hc->rem_worker_addr, hc->rem_worker_addr_len, MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv worker address failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    memset(&hc->pipeline, 0, sizeof(hc->pipeline));
    ret = recv(hc->connfd, &hc->pipeline.buffer_size, sizeof(size_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv pipeline buffer size failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = recv(hc->connfd, &hc->pipeline.num_buffers, sizeof(size_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv pipeline num buffers failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = _dpu_hc_init_pipeline(hc);
    if (ret) {
        fprintf(stderr, "init pipeline failed!\n");
        goto err;
    }

    ret = recv(hc->connfd, &hc->world_rank, sizeof(uint32_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv world rank failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = recv(hc->connfd, &hc->world_size, sizeof(uint32_t), MSG_WAITALL);
    if (-1 == ret) {
        fprintf(stderr, "recv world size failed!\n");
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = 0;
    goto out;

err:
    close(hc->connfd);
err_ucx:
    _dpu_ucx_fini(hc);
out:
    return ret;
}

int dpu_hc_connect_remote_hosts(dpu_hc_t *hc, dpu_ucc_comm_t *comm)
{
    int ret;

    if (ret = _dpu_create_remote_host_eps(hc, comm)) {
        fprintf(stderr, "_dpu_create_remote_host_eps failed!, %d\n", ret);
        ret = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    ret = _dpu_flush_host_eps(hc);
    if (ret) {
        fprintf(stderr, "ep flush failed!\n");
        goto err;
    }

err:
    return ret;
}

int dpu_dc_create(thread_ctx_t *ctx, dpu_hc_t *hc, dpu_hc_t *dc)
{
    int ret;

    assert(hc->channel_type == HOST_COMM_CHANNEL);
    memcpy(dc, hc, sizeof(dpu_hc_t));
    dc->channel_type = HOST_DATA_CHANNEL;

    CTX_LOG("Creating Data channel\n");
    ret = _dpu_ucx_init(dc);
    if (ret) {
        goto err_ucx;
    }

    ret = _dpu_hc_init_pipeline(dc);
    if (ret) {
        fprintf(stderr, "DC init pipeline failed!\n");
        goto err_ucx;
    }

    CTX_LOG("Created Data channel with rank %d size %d\n",
            dc->world_rank, dc->world_size);
    
    ret = 0;
    goto out;
err_ucx:
    _dpu_ucx_fini(dc);
out:
    return ret;
}

int dpu_hc_wait(dpu_hc_t *hc, uint32_t next_coll_id)
{
    dpu_put_sync_t *lsync = (dpu_put_sync_t*)hc->mem_segs.sync.base;
    ucp_request_param_t req_param = {0};
    ucp_tag_t req_tag = 0, tag_mask = 0;
    ucs_status_t status;

    ucs_status_ptr_t recv_req = ucp_tag_recv_nbx(hc->ucp_worker,
            lsync, sizeof(dpu_put_sync_t),
            req_tag, tag_mask, &req_param);
    status = _dpu_request_wait(hc->ucp_worker, recv_req);

    DPU_LOG("Got next coll id from host: %u was expecting %u\n", lsync->coll_id, next_coll_id);
    assert(lsync->coll_id == next_coll_id);

    host_rkey_t *rkeys = &lsync->rkeys;

    status = ucp_ep_rkey_unpack(hc->localhost_ep, (void*)rkeys->src_rkey_buf, &hc->src_rkey);

    status = ucp_ep_rkey_unpack(hc->localhost_ep, (void*)rkeys->dst_rkey_buf, &hc->dst_rkey);

    return 0;
}

int dpu_hc_reply(dpu_hc_t *hc, dpu_get_sync_t *coll_sync)
{
    ucs_status_t status;
    ucs_status_ptr_t request;
    ucp_tag_t req_tag = 0;

    DPU_LOG("Flushing host ep for coll_id: %d\n", coll_sync->coll_id);
    _dpu_worker_flush(hc);

    ucp_worker_fence(hc->ucp_worker);
    DPU_LOG("Notify host completed coll_id: %u, serviced: %u\n",
            coll_sync->coll_id, coll_sync->count_serviced);
    request = ucp_tag_send_nbx(hc->localhost_ep,
            coll_sync, sizeof(dpu_get_sync_t), req_tag, &hc->req_param);
    status = _dpu_request_wait(hc->ucp_worker, request);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to notify host of completion (%s)\n", ucs_status_string(status));
        return -1;
    }

    ucp_rkey_destroy(hc->src_rkey);
    ucp_rkey_destroy(hc->dst_rkey);
    dpu_hc_reset_pipeline(hc);
    return 0;
}

ucc_rank_t dpu_get_world_rank(dpu_hc_t *hc,  int dpu_rank, int team_id, thread_ctx_t *ctx)
{
    ucc_rank_t  world_rank;

    if (team_id == ctx->comm->world_team_id) {
        world_rank = dpu_rank;
    } else {
        world_rank = ctx->comm->dpu_team_ctx_ranks[team_id][dpu_rank];
    }

    return world_rank;
}

ucc_rank_t dpu_get_host_ep_rank(dpu_hc_t *hc,  int host_rank, int team_id, thread_ctx_t *ctx)
{
    /* find my world_rank of the remote process in dpu comm world then find
     * its ep_rank */
    ucc_rank_t ep_rank, world_rank;

    if (team_id == ctx->comm->world_team_id) {
        world_rank = host_rank;
    } else {
        world_rank = ctx->comm->host_team_ctx_ranks[team_id][host_rank];
    }

    ep_rank = world_rank * hc->dpu_per_node_cnt;
    return ep_rank;
}

#if 0
ucs_status_t dpu_hc_issue_get(dpu_hc_t *hc, dpu_put_sync_t *sync, dpu_buf_t *getbuf, thread_ctx_t *ctx)
{
    assert(getbuf->state == FREE && getbuf->ucp_req == NULL);
    dpu_pipeline_t *pipe = &hc->pipeline;
    uint32_t host_team_size = sync->num_ranks;

    ucc_datatype_t dtype = sync->coll_args.src.info.datatype;
    size_t dt_size = dpu_ucc_dt_size(dtype);
    size_t remaining_elems = pipe->my_count - pipe->count_received;
    size_t count = DPU_MIN(pipe->buffer_size/dt_size, remaining_elems);
    size_t get_offset = pipe->my_offset + pipe->count_received * dt_size;
    int src_rank = pipe->src_rank;
    int ep_src_rank = dpu_get_host_ep_rank(hc, src_rank, sync->team_id, ctx);

    assert(src_rank < host_team_size);

    if (0 == count) {
        return UCS_ERR_NO_RESOURCE;
    }

    size_t data_size = count * dt_size;
    void *src_addr = hc->host_rkeys[ep_src_rank].src_buf + get_offset;
    void *dst_addr = getbuf->buf;

    CTX_LOG("Issue Get from %d offset %lu src %p dst %p count %lu bytes %lu host_team_size: %d \n",
            ep_src_rank, get_offset, src_addr, dst_addr, count, data_size, host_team_size);
    
    ucp_worker_fence(hc->ucp_worker);
    getbuf->state = RECVING;
    getbuf->count = count;
    getbuf->bytes = data_size;
    getbuf->ucp_req =
            ucp_get_nbx(hc->host_eps[ep_src_rank], dst_addr, data_size,
            (uint64_t)src_addr, hc->host_src_rkeys[ep_src_rank], &hc->req_param);

    pipe->src_rank = (src_rank + 1) % host_team_size;
    return UCS_OK;
}

ucs_status_t dpu_hc_issue_put(dpu_hc_t *hc, dpu_put_sync_t *sync, dpu_buf_t *accbuf, thread_ctx_t *ctx)
{
    dpu_pipeline_t *pipe = &hc->pipeline;
    uint32_t host_team_size = sync->num_ranks;
    ucc_datatype_t dtype = sync->coll_args.src.info.datatype;
    size_t dt_size = dpu_ucc_dt_size(dtype);
    size_t count = accbuf->count;
    size_t put_offset = pipe->my_offset + pipe->count_serviced * dt_size;
    int dst_rank = pipe->dst_rank;
    int ep_dst_rank = dpu_get_host_ep_rank(hc, dst_rank, sync->team_id, ctx);

    assert(dst_rank < host_team_size);

    size_t data_size = count * dt_size;
    void *src_addr = accbuf->buf;
    void *dst_addr = hc->host_rkeys[ep_dst_rank].dst_buf + put_offset;

    CTX_LOG("Issue Put to %d offset %lu src %p dst %p count %lu bytes %lu host_team_size: %d\n",
            dst_rank, put_offset, src_addr, dst_addr, count, data_size, host_team_size);
    assert(count > 0 && dt_size == accbuf->bytes && accbuf->ucp_req == NULL);

    ucp_worker_fence(hc->ucp_worker);
    accbuf->state = SENDING;
    accbuf->ucp_req =
            ucp_put_nbx(hc->host_eps[ep_dst_rank], src_addr, data_size,
            (uint64_t)dst_addr, hc->host_dst_rkeys[ep_dst_rank], &hc->req_param);

    pipe->dst_rank = (dst_rank + 1) % host_team_size;
    return UCS_OK;
}

ucs_status_t dpu_hc_local_reduce(dpu_hc_t *hc, dpu_put_sync_t *sync, thread_ctx_t *ctx, dpu_buf_t *accbuf, dpu_buf_t *getbuf)
{
    assert(accbuf->state == IDLE && accbuf->ucp_req == NULL);
    assert(getbuf->state == IDLE && getbuf->ucp_req == NULL);
    assert(accbuf->count == getbuf->count);

    accbuf->state = REDUCING;
    getbuf->state = REDUCING;

    ucc_datatype_t dtype = sync->coll_args.src.info.datatype;
    ucc_reduction_op_t op = sync->coll_args.op;

    ucp_worker_fence(hc->ucp_worker);
    CTX_LOG("Issue AR accbuf %p getbuf %p count %lu\n", accbuf->buf, getbuf->buf, accbuf->count);
    ucc_mc_reduce_multi(accbuf->buf, getbuf->buf, accbuf->buf,
                        1, accbuf->count * 0, 0, dtype, op, UCC_MEMORY_TYPE_HOST);
    
    return UCS_OK;
}

ucs_status_t dpu_hc_progress_allreduce(dpu_hc_t *hc,
                    dpu_put_sync_t *sync,
                    thread_ctx_t *ctx)
{
    int i, j;
    ucc_status_t status;
    ucs_status_ptr_t request;
    dpu_pipeline_t *pp = &hc->pipeline;
    uint32_t host_team_size = sync->num_ranks;
    assert(host_team_size >= 1);
    int read_completed = 0;

    for (i=0; i<1; i++) {
        if (ucp_worker_progress(hc->ucp_worker)) {
            break;
        }
    }
    

    size_t num_buffers = pp->num_buffers;
    dpu_stage_t *stage = &pp->stages[0];
    dpu_buf_t *accbuf = &stage->accbuf;
    dpu_buf_t *getbuf = &stage->getbuf[stage->get_idx];
    dpu_buf_t *redbuf = &stage->getbuf[stage->red_idx];

    switch(stage->phase) {
        case INIT:
        if (accbuf->state == FREE) {
            stage->src_rank = stage->dst_rank = sync->host_team_rank;
            assert(stage->done_get == 0);
            dpu_hc_issue_get(hc, sync, stage, accbuf, ctx);
            /* Issue next Get from peer host immediately since
               read from own host only uses PCIe Link */
            stage->phase = REDUCE;
        }
        break;
        case REDUCE:
        read_completed = 0;
        if (getbuf->state == FREE && stage->done_get < host_team_size) {
            dpu_hc_issue_get(hc, sync, stage, getbuf, ctx);
            /* Advance Get buffer to next */
            stage->get_idx = (stage->get_idx + 1) % num_buffers;
        }
        /* Progress Read from own host if needed */
        if (accbuf->state == SENDRECV && accbuf->count > 0) {
            request = accbuf->ucp_req;
            if (_dpu_req_test(request) == UCS_OK) {
                if (request) ucp_request_free(request);
                accbuf->ucp_req = NULL;
                accbuf->state = IDLE;
                stage->done_get += 1;
                CTX_LOG("Received %ld bytes into accbuf, gets done %d\n",
                        accbuf->count, stage->done_get);
                read_completed++;
            }
        }
        /* Progress Read from remote host */
        if (getbuf->state == SENDRECV && getbuf->count > 0) {
            request = getbuf->ucp_req;
            if (_dpu_req_test(request) == UCS_OK) {
                if (request) ucp_request_free(request);
                getbuf->ucp_req = NULL;
                getbuf->state = IDLE;
                stage->done_get += 1;
                CTX_LOG("Received %ld bytes into getbuf[%d], gets done %d\n",
                        getbuf->count, stage->get_idx, stage->done_get);
                read_completed++;
            }
        }
        if (read_completed > 0) {
            if (stage->done_get == host_team_size) {
                pp->count_received += accbuf->count;
            }
        }
        if (redbuf->state == IDLE && accbuf->state == IDLE) {
            assert(stage->done_get > stage->done_red);
            getbuf->state = REDUCING;
            dpu_hc_local_reduce(hc, sync, ctx, stage, accbuf, redbuf);
        }
        if (redbuf->state == REDUCING && accbuf->state == REDUCING) {
            /* Blocking reduce will always be completed */
            redbuf->state = FREE;
            accbuf->state = IDLE;
            stage->done_red += 1;
            CTX_LOG("Reduced %ld bytes from getbuf[%d], reds done %d\n",
                    redbuf->count, stage->red_idx, stage->done_red);
            stage->red_idx = (stage->red_idx + 1) % num_buffers;

            if (stage->done_red == host_team_size-1) {
                /* Start broadcast */
                stage->phase = BCAST;
                pp->count_reduced += accbuf->count;
            }
        }
        break;
        case BCAST:
        if (accbuf->state == IDLE && accbuf->count > 0) {
            CTX_LOG("Broadcasting host_team_size %d done get %d red %d put %d\n",
                    host_team_size, stage->done_get, stage->done_red, stage->done_put);
            assert(stage->done_get == host_team_size);
            assert(stage->done_red == host_team_size-1);
            assert(stage->done_put < host_team_size);

            #if 0
            dpu_hc_issue_put(hc, sync, stage, accbuf, ctx);
            #else
            window_size = DPU_MIN(hc->window_size, (host_team_size-stage->done_put));
            /* Issue a window of RDMA writes */
            for (int j=0; j<window_size; j++) {
                dpu_hc_issue_put(hc, sync, stage, accbuf, ctx);
                request = accbuf->ucp_req;
                if (request) ucp_request_free(request);
                accbuf->ucp_req = NULL;
                stage->done_put++;
            }
            /* Wait for completion */
            _dpu_flush_host_eps(hc);
            #endif

        } else if (accbuf->state == SENDRECV && accbuf->count > 0) {
            request = accbuf->ucp_req;
            if (_dpu_req_test(request) == UCS_OK) {
                // if (request) ucp_request_free(request);
                // accbuf->ucp_req = NULL;
                // stage->done_put++;
                accbuf->state = IDLE;
                CTX_LOG("Sent %ld bytes from accbuf, puts done %d\n",
                        accbuf->count, stage->done_put);
                
                if (stage->done_put == host_team_size) {
                    /* Done with this stage */
                    pp->count_serviced += accbuf->count;
                    /* Reset this stage counters */
                    _dpu_hc_reset_stage(stage, hc);
                    /* Allow next stage to start */
                    stage->phase = INIT;
                }
            }
        } else {
            CTX_LOG("Impossible!! %p phase %d state %d\n",
                    stage, stage->phase, accbuf->state);
            assert(0);
        }
        break;
        default:
        break;
    }

}
#endif

ucs_status_t dpu_hc_progress_allreduce(dpu_hc_t *dc,
                    dpu_put_sync_t *sync,
                    thread_ctx_t *ctx)
{
    ucc_status_t status = UCS_OK;
    dpu_pipeline_t *pipe = &dc->pipeline;
    ucc_datatype_t dtype = sync->coll_args.src.info.datatype;
    size_t dt_size = dpu_ucc_dt_size(dtype);
    uint32_t host_team_size = sync->num_ranks;

    while (pipe->count_serviced < pipe->my_count) {

        /* Issue RDMA Read */
        if (pipe->count_received < pipe->my_count && pipe->avail_buffs) {
            size_t remaining_elems = pipe->my_count - pipe->count_received;
            size_t get_idx = pipe->get_idx % pipe->num_buffers;
            size_t count = DPU_MIN(pipe->buffer_size/dt_size, remaining_elems);
            size_t get_offset = pipe->my_offset + pipe->count_received * dt_size;
            size_t data_size = count * dt_size;
            
            int src_rank = pipe->src_rank;
            int ep_src_rank = dpu_get_host_ep_rank(dc, src_rank, sync->team_id, ctx);
            dpu_buf_t *getbuf = &pipe->getbuf[get_idx];
            void *src_addr = dc->host_rkeys[ep_src_rank].src_buf + get_offset;
            void *dst_addr = getbuf->buf;

            CTX_LOG("Issue Get from %d offset %lu src %p dst %p count %lu bytes %lu host_team_size: %d \n",
                    ep_src_rank, get_offset, src_addr, dst_addr, count, data_size, host_team_size);

            ucp_worker_fence(dc->ucp_worker);
            getbuf->state = RECVING;
            getbuf->count = count;
            getbuf->bytes = data_size;
            getbuf->ucp_req =
                    ucp_get_nbx(dc->host_eps[ep_src_rank], dst_addr, data_size,
                    (uint64_t)src_addr, dc->host_src_rkeys[ep_src_rank], &dc->req_param);

            pipe->src_rank = (src_rank + 1) % host_team_size;
            pipe->avail_buffs--;
            pipe->get_idx++;
            pipe->done_get++;
            if (pipe->done_get == host_team_size) {
                pipe->count_received += count;
                pipe->done_get = 0;
            }
        }

        /* Do Compute */
        if (pipe->count_reduced <= pipe->count_received && pipe->red_idx <= pipe->get_idx) {
            size_t red_idx = pipe->red_idx % pipe->num_buffers;
            dpu_buf_t *redbuf = &pipe->getbuf[red_idx];
            if (redbuf->state == RECVING) {
                ucs_status_ptr_t request = redbuf->ucp_req;
                if (_dpu_req_test(request) == UCS_OK) {
                    if (request) ucp_request_free(request);
                    redbuf->state = REDUCING;
                    redbuf->ucp_req = NULL;

                    CTX_LOG("Data received count %lu bytes %lu host_team_size: %d \n",
                            redbuf->count, redbuf->bytes, host_team_size);
                    pipe->avail_buffs++;
                    pipe->red_idx++;
                    pipe->done_red++;

                    if (pipe->done_red == host_team_size) {
                        pipe->count_reduced  += redbuf->count;
                        /* FIXME */
                        pipe->count_serviced += redbuf->count;
                        pipe->done_red = 0;
                    }
                }
            }
        }

        /* Issue RDMA Write */

        ucp_worker_progress(dc->ucp_worker);
    }

    _dpu_flush_host_eps(dc);

    return status;
}

ucs_status_t dpu_recv_world_team_id(dpu_hc_t *hc, dpu_ucc_comm_t *comm)
{
    ucs_status_t status;
    ucp_tag_t req_tag = 0;
    ucp_tag_t tag_mask = 0;
    ucs_status_ptr_t request;

    ucp_worker_fence(hc->ucp_worker);
    request = ucp_tag_recv_nbx(hc->ucp_worker,
            &comm->world_team_id, sizeof(uint16_t),
            req_tag, tag_mask, &hc->req_param);
    status = _dpu_request_wait(hc->ucp_worker, request);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to recv world team id (%s)\n", ucs_status_string(status));
        return status;
    }

    DPU_LOG("Recvd world team id %d\n", comm->world_team_id);
    comm->team_pool[comm->world_team_id] = comm->team;
    return UCS_OK;
}

ucs_status_t dpu_send_init_completion(dpu_hc_t *hc)
{
    ucs_status_t status;
    ucp_tag_t req_tag = 0;
    ucs_status_ptr_t request;

    dpu_get_sync_t coll_sync;
    coll_sync.coll_id = -1;
    coll_sync.count_serviced = -1;

    printf("# Accepted Job Id %d with rank %d size %d\n",
           hc->job_id, hc->world_rank, hc->world_size);
    _dpu_worker_flush(hc);

    ucp_worker_fence(hc->ucp_worker);
    request = ucp_tag_send_nbx(hc->localhost_ep,
            &coll_sync, sizeof(dpu_get_sync_t), req_tag, &hc->req_param);
    status = _dpu_request_wait(hc->ucp_worker, request);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to notify host of init completion (%s)\n", ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}

int dpu_dc_reset(dpu_hc_t *dc)
{
    _dpu_flush_host_eps(dc);
    _dpu_worker_flush(dc);
    _dpu_hc_buffer_free(dc, &dc->mem_segs.in);
    _dpu_hc_buffer_free(dc, &dc->mem_segs.sync);
    _dpu_close_host_eps(dc);
    _dpu_ucx_fini(dc);
    return UCC_OK;
}

int dpu_hc_reset_job(dpu_hc_t *hc)
{
    _dpu_flush_host_eps(hc);
    _dpu_worker_flush(hc);
    _dpu_hc_buffer_free(hc, &hc->mem_segs.in);
    _dpu_hc_buffer_free(hc, &hc->mem_segs.sync);
    _dpu_close_host_eps(hc);
    _dpu_ucx_fini(hc);
    printf("# Completed Job Id %d\n", hc->job_id);
    return UCC_OK;
}

/* TODO: call from atexit() */
int dpu_hc_finalize(dpu_hc_t *hc)
{
    printf("Finalizing DPU Server, Job Id %d\n", hc->job_id);
    _dpu_listen_cleanup(hc);
    return UCC_OK;
}
