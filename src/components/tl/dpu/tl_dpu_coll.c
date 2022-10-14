/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"
#include "tl_dpu_coll.h"

#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "../../../core/ucc_team.h"

ucc_status_t ucc_tl_dpu_req_test(ucs_status_ptr_t *req_p, ucp_worker_h worker)
{
    ucs_status_t status;
    ucs_status_ptr_t request = *req_p;
    if (request == NULL) {
        status = UCS_OK;
    }
    else if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
        fprintf (stderr, "unable to complete UCX request (%s)\n", ucs_status_string(status));
    }
    else {
        ucp_worker_progress(worker);
        status = ucp_request_check_status(request);
        if (UCS_OK == status) {
            ucp_request_free(request);
            *req_p = NULL;
        }
    }
    return ucs_status_to_ucc_status(status);
}

ucc_status_t ucc_tl_dpu_req_check(ucc_tl_dpu_team_t *team,
                                      ucs_status_ptr_t req) {
    if (UCS_PTR_IS_ERR(req)) {
        tl_error(team->super.super.context->lib,
                 "failed to send/recv msg");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_req_wait(ucp_worker_h ucp_worker, ucs_status_ptr_t request)
{
    ucs_status_t status;

    /* immediate completion */
    if (request == NULL) {
        return UCC_OK;
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

    return UCC_OK;
}

ucs_status_t ucc_tl_dpu_register_buf(
    ucp_context_h ucp_ctx,
    void *base, size_t size,
    ucc_tl_dpu_rkey_t *rkey)
{
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;
    ucp_mem_map_params_t mem_params = {
        .address = base,
        .length = size,
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS  |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
    };

    status = ucp_mem_map(ucp_ctx, &mem_params, &rkey->memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_map (%s)\n", ucs_status_string(status));
        goto out;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;

    status = ucp_mem_query(rkey->memh, &mem_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_query (%s)\n", ucs_status_string(status));
        goto err_map;
    }
    assert(mem_attr.length >= size);
    assert(mem_attr.address <= base);

    status = ucp_rkey_pack(ucp_ctx, rkey->memh, &rkey->rkey_buf, &rkey->rkey_buf_size);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status));
        goto err_map;
    }
    assert(rkey->rkey_buf_size < MAX_RKEY_LEN);

    goto out;
err_map:
    ucp_mem_unmap(ucp_ctx, rkey->memh);
out:
    return status;
}

ucc_status_t ucc_tl_dpu_deregister_buf(
    ucp_context_h ucp_ctx, ucc_tl_dpu_rkey_t *rkey)
{
    ucs_status_t status = UCS_OK;
    status = ucp_mem_unmap(ucp_ctx, rkey->memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_unmap (%s)\n", ucs_status_string(status));
        goto out;
    }
    ucp_rkey_buffer_release(rkey->rkey_buf);
out:
    return status;
}

static ucc_status_t ucc_tl_dpu_init_rkeys(ucc_tl_dpu_task_t *task)
{
    ucc_status_t status = UCC_OK;
    ucc_tl_dpu_context_t *ctx = UCC_TL_DPU_TEAM_CTX(task->team);

    void *src_buf, *dst_buf;
    size_t src_len, dst_len;

    if (task->args.coll_type == UCC_COLL_TYPE_ALLTOALLV) {
        src_buf = task->args.src.info_v.buffer;
        dst_buf = task->args.dst.info_v.buffer;

        src_len = dst_len = 0;
        for (int i=0; i<task->team->size; i++) {
            src_len += task->args.src.info_v.counts[i];
            dst_len += task->args.dst.info_v.counts[i];
        }

        src_len *= ucc_dt_size(task->args.src.info_v.datatype);
        dst_len *= ucc_dt_size(task->args.dst.info_v.datatype);
    } else {
        src_buf = task->args.src.info.buffer;
        dst_buf = task->args.dst.info.buffer;
        src_len = task->args.src.info.count * ucc_dt_size(task->args.src.info.datatype);
        dst_len = task->args.dst.info.count * ucc_dt_size(task->args.dst.info.datatype);
    }

    for (int rail = 0; rail < task->dpu_per_node_cnt; rail++) {
        status |= ucc_tl_dpu_register_buf(ctx->dpu_ctx_list[rail].ucp_context,
                src_buf, src_len, &task->dpu_task_list[rail].src_rkey);
        status |= ucc_tl_dpu_register_buf(ctx->dpu_ctx_list[rail].ucp_context,
                dst_buf, dst_len, &task->dpu_task_list[rail].dst_rkey);
    }
    return status;
}

static void ucc_tl_dpu_finalize_rkeys(ucc_tl_dpu_task_t *task)
{
    int rail;
    ucc_tl_dpu_context_t *ctx = UCC_TL_DPU_TEAM_CTX(task->team);

    for (rail = 0; rail < ctx->dpu_per_node_cnt; rail++) {
        ucc_tl_dpu_deregister_buf(ctx->dpu_ctx_list[rail].ucp_context,
                &task->dpu_task_list[rail].src_rkey);
        ucc_tl_dpu_deregister_buf(ctx->dpu_ctx_list[rail].ucp_context,
                &task->dpu_task_list[rail].dst_rkey);
        task->dpu_task_list[rail].status = UCC_TL_DPU_TASK_STATUS_FINALIZED; 
    }
}

static void ucc_tl_dpu_init_put(ucc_tl_dpu_context_t *ctx,
    ucc_tl_dpu_task_t *task, ucc_tl_dpu_team_t *team, int rail)
{
    ucc_tl_dpu_put_sync_t *put_sync = &task->dpu_task_list[rail].put_sync;
    ucc_tl_dpu_sub_task_t *dpu_task = &task->dpu_task_list[rail];

    memcpy(put_sync->rkeys.src_rkey, dpu_task->src_rkey.rkey_buf,
            dpu_task->src_rkey.rkey_buf_size);
    memcpy(put_sync->rkeys.dst_rkey, dpu_task->dst_rkey.rkey_buf,
            dpu_task->dst_rkey.rkey_buf_size);

    put_sync->rkeys.src_rkey_len = dpu_task->src_rkey.rkey_buf_size;
    put_sync->rkeys.dst_rkey_len = dpu_task->dst_rkey.rkey_buf_size;
    put_sync->rkeys.src_buf = task->args.src.info.buffer;
    put_sync->rkeys.dst_buf = task->args.dst.info.buffer;
}

static ucc_status_t ucc_tl_dpu_issue_send( ucc_tl_dpu_task_t *task,
    ucc_tl_dpu_context_t *ctx, ucc_tl_dpu_team_t *team, int rail)
{
    ucc_tl_dpu_sub_task_t *dpu_task = &task->dpu_task_list[rail];
    ucc_tl_dpu_task_req_t *task_reqs = &dpu_task->task_reqs;
    ucp_request_param_t req_param = {0};
    ucp_tag_t req_tag = 0;

    assert(dpu_task->status != UCC_TL_DPU_TASK_STATUS_POSTED);
    dpu_task->status = UCC_TL_DPU_TASK_STATUS_POSTED;
    ucc_tl_dpu_init_put(ctx, task, team, rail);
    assert(task_reqs->send_req == NULL);

    assert(task->dpu_per_node_cnt > 0);
    assert(dpu_task->put_sync.dpu_per_node_cnt > 0);
    ucp_worker_fence(ctx->dpu_ctx_list[rail].ucp_worker);

    task_reqs->send_req = ucp_tag_send_nbx(ctx->dpu_ctx_list[rail].ucp_ep,
        &dpu_task->put_sync, sizeof(dpu_task->put_sync), req_tag, &req_param);
    if (ucc_tl_dpu_req_check(team, task_reqs->send_req) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }

    tl_info(UCC_TL_TEAM_LIB(task->team), "Sent task to DPU: %p, coll type %d id %d count %u",
            task, dpu_task->put_sync.coll_args.coll_type, dpu_task->put_sync.coll_id,
            dpu_task->put_sync.count_total);
 
    ucc_tl_dpu_req_wait(ctx->dpu_ctx_list[rail].ucp_worker, task_reqs->send_req);
    task_reqs->send_req = NULL;
    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_issue_recv( ucc_tl_dpu_task_t *task,
    ucc_tl_dpu_context_t *ctx, ucc_tl_dpu_team_t *team, int rail)
{
    ucc_tl_dpu_sub_task_t *dpu_task = &task->dpu_task_list[rail];
    ucc_tl_dpu_task_req_t *task_reqs = &dpu_task->task_reqs;
    ucp_request_param_t req_param = {0};
    ucp_tag_t req_tag = 0, tag_mask = 0;

    assert(dpu_task->status == UCC_TL_DPU_TASK_STATUS_POSTED);
    assert(task_reqs->recv_req == NULL);
    ucp_worker_fence(ctx->dpu_ctx_list[rail].ucp_worker);

    task_reqs->recv_req = ucp_tag_recv_nbx(ctx->dpu_ctx_list[rail].ucp_worker, 
                &dpu_task->get_sync, sizeof(ucc_tl_dpu_get_sync_t),
                req_tag, tag_mask, &req_param);
    tl_info(UCC_TL_TEAM_LIB(task->team),
            "Posted recv to DPU task %p rail %d req %p, get sync %p, current coll id %d count %u",
            task, rail, task_reqs->recv_req, &dpu_task->get_sync,
            dpu_task->get_sync.coll_id, dpu_task->get_sync.count_serviced);
    return ucc_tl_dpu_req_check(team, task_reqs->recv_req);
}

static ucc_status_t ucc_tl_dpu_send_dpu_task(
    ucc_tl_dpu_task_t *task, ucc_tl_dpu_context_t *ctx)
{
    ucc_tl_dpu_team_t *team = task->team;
    ucc_tl_dpu_sub_task_t *sub_task = NULL;

    assert (task->status == UCC_TL_DPU_TASK_STATUS_INIT);
    for (int rail = 0; rail < task->dpu_per_node_cnt; rail++) {
        sub_task = &task->dpu_task_list[rail];
        assert(sub_task->status == UCC_TL_DPU_TASK_STATUS_INIT);
        tl_info(UCC_TL_TEAM_LIB(task->team), 
                "Put to DPU rail: %d coll task: %p, coll id %d, DPU count: %d", 
                rail, task, sub_task->put_sync.coll_id, task->dpu_per_node_cnt);
        ucc_tl_dpu_issue_send(task, ctx, team, rail);
        ucc_tl_dpu_issue_recv(task, ctx, team, rail);
        assert(sub_task->status == UCC_TL_DPU_TASK_STATUS_POSTED);
    }

    task->status = UCC_TL_DPU_TASK_STATUS_POSTED;
    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_check_progress(
    ucc_tl_dpu_task_t *task, ucc_tl_dpu_context_t *ctx)
{
    int i;
    ucc_tl_dpu_team_t *team = task->team;
    int rail = 0, rails_done = 0;
    ucc_tl_dpu_sub_task_t *sub_task = NULL;

    assert (task->status == UCC_TL_DPU_TASK_STATUS_POSTED);
    for (rail = 0; rail < task->dpu_per_node_cnt; rail++) {
        for (i = 0; i < UCC_TL_DPU_TC_POLL; i++) {
            if (ucp_worker_progress(ctx->dpu_ctx_list[rail].ucp_worker)) {
                break;
            }
        }
    }

    rails_done = 0;
    for (rail = 0; rail < task->dpu_per_node_cnt; rail++) {
        sub_task = &task->dpu_task_list[rail];
        ucc_tl_dpu_connect_t *dpu_connect = &ctx->dpu_ctx_list[rail];

        if (sub_task->status != UCC_TL_DPU_TASK_STATUS_DONE) {
            ucc_tl_dpu_task_req_t *task_reqs = &sub_task->task_reqs;
            ucc_tl_dpu_req_test(&task_reqs->recv_req, dpu_connect->ucp_worker);

            if (sub_task->get_sync.coll_id == sub_task->put_sync.coll_id &&
                sub_task->get_sync.count_serviced == sub_task->put_sync.count_total) {

                team->dpu_sync_list[rail].coll_id_completed = ++dpu_connect->coll_id_completed;
                assert(team->dpu_sync_list[rail].coll_id_completed == sub_task->get_sync.coll_id);
                sub_task->status = UCC_TL_DPU_TASK_STATUS_DONE;

                tl_info(UCC_TL_TEAM_LIB(task->team),
                    "Collective task %p coll id %d is marked DONE for rail: %d\n",
                    task, sub_task->put_sync.coll_id, rail);
                rails_done++;
            }
        } else {
            rails_done++;
        }
    }

    if (rails_done == task->dpu_per_node_cnt) {
        task->status = UCC_TL_DPU_TASK_STATUS_DONE;
        return UCC_OK;
    }

    return UCC_INPROGRESS;
}

static ucc_status_t ucc_tl_dpu_coll_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t *task = ucs_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_team_t *team = task->team;

    tl_info(UCC_TL_TEAM_LIB(task->team), "Collective start task %p coll type %d id %d",
            task, task->args.coll_type, task->dpu_task_list[0].put_sync.coll_id);

    coll_task->status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_DPU_TEAM_CORE_CTX(team)->pq, coll_task);
}

static void ucc_tl_dpu_coll_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t *task   = ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_context_t *ctx = UCC_TL_DPU_TEAM_CTX(task->team);
    ucc_status_t status;

    status = ucc_tl_dpu_check_progress(task, ctx);
    coll_task->status = status;
}

static ucc_status_t ucc_tl_dpu_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t *task = ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    tl_info(UCC_TL_TEAM_LIB(task->team),
            "finalizing task %p, task status %d, coll status %d, coll id %u",
            task, task->status, coll_task->status,
            task->dpu_task_list[0].get_sync.coll_id);

    assert(coll_task->status == UCC_OK);
    if(task->status == UCC_TL_DPU_TASK_STATUS_FINALIZED) {
        tl_warn(UCC_TL_TEAM_LIB(task->team),
                 "task %p already finalized, status %d, coll id %u",
                 task, task->status, task->dpu_task_list[0].get_sync.coll_id);
        return UCC_OK;
    }

    assert(task->status == UCC_TL_DPU_TASK_STATUS_DONE);
    ucc_tl_dpu_finalize_rkeys(task);
    task->status = UCC_TL_DPU_TASK_STATUS_FINALIZED;
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_allreduce_init(ucc_tl_dpu_task_t *task)
{
    ucc_coll_args_t      *coll_args = &task->args;
    ucc_tl_dpu_team_t    *team      = task->team;
    ucc_base_team_t      *base_team = &team->super.super;
    ucc_tl_dpu_put_sync_t *task_put_sync = NULL;

    if (!UCC_IS_INPLACE(*coll_args) && (coll_args->src.info.mem_type !=
                                        coll_args->dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_INPLACE(*coll_args)) {
        coll_args->src.info.buffer   = coll_args->dst.info.buffer;
        coll_args->dst.info.count    = coll_args->src.info.count;
        coll_args->dst.info.datatype = coll_args->src.info.datatype;
        coll_args->dst.info.mem_type = coll_args->src.info.mem_type;
    }

    assert(task->status == UCC_TL_DPU_TASK_STATUS_INIT);

    ucc_tl_dpu_init_rkeys(task);

    /* Set sync information for DPU */
    for (int rail = 0; rail < task->dpu_per_node_cnt; rail++) {

        task_put_sync = &task->dpu_task_list[rail].put_sync;
        memcpy(&task_put_sync->coll_args, coll_args, sizeof(ucc_coll_args_t));

        task_put_sync->count_total       = coll_args->src.info.count;
        task_put_sync->coll_id           = team->dpu_sync_list[rail].coll_id_issued;
        task_put_sync->team_id           = base_team->params.id;
        task_put_sync->create_new_team   = 0;
        task_put_sync->dpu_per_node_cnt  = task->dpu_per_node_cnt;
        task_put_sync->rail              = rail;
        task_put_sync->num_ranks         = team->size;
        task_put_sync->host_team_rank    = team->rank;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_alltoall_init(ucc_tl_dpu_task_t *task)
{
    ucc_coll_args_t     *coll_args = &task->args;
    ucc_tl_dpu_team_t   *team      = task->team;
    ucc_base_team_t     *base_team = &team->super.super;
    ucc_tl_dpu_put_sync_t *task_put_sync = NULL;

    if (!UCC_IS_INPLACE(task->args) && (task->args.src.info.mem_type !=
                                        task->args.dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_INPLACE(*coll_args)) {
        coll_args->src.info.buffer = coll_args->dst.info.buffer;
    }

    /* Set sync information for DPU */
    int rail = 0;
    task->dpu_per_node_cnt = 1;

    task_put_sync = &task->dpu_task_list[rail].put_sync;
    memcpy(&task_put_sync->coll_args, coll_args, sizeof(ucc_coll_args_t));

    task_put_sync->count_total      = coll_args->src.info.count;
    task_put_sync->coll_id          = team->dpu_sync_list[rail].coll_id_issued;
    task_put_sync->team_id          = base_team->params.id;
    task_put_sync->create_new_team  = 0;
    task_put_sync->dpu_per_node_cnt = task->dpu_per_node_cnt;
    task_put_sync->rail             = rail;

    ucc_tl_dpu_init_rkeys(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_alltoallv_init(ucc_tl_dpu_task_t *task)
{
    ucc_coll_args_t     *coll_args = &task->args;
    ucc_tl_dpu_team_t   *team      = task->team;
    ucc_base_team_t     *base_team = &team->super.super;
    ucc_tl_dpu_put_sync_t *task_put_sync = NULL;

    if (!UCC_IS_INPLACE(*coll_args) && (coll_args->src.info.mem_type !=
                                        coll_args->dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* TODO: is in_place supported? */
    if (UCC_IS_INPLACE(*coll_args)) {
        coll_args->src.info_v.buffer = coll_args->dst.info_v.buffer;
    }

    /* Set sync information for DPU */
    int rail = 0;
    task->dpu_per_node_cnt = 1;

    task_put_sync = &task->dpu_task_list[rail].put_sync;
    memcpy(&task_put_sync->coll_args, coll_args, sizeof(ucc_coll_args_t));

    task_put_sync->coll_id          = team->dpu_sync_list[rail].coll_id_issued;
    task_put_sync->team_id          = base_team->params.id;
    task_put_sync->create_new_team  = 0;
    task_put_sync->dpu_per_node_cnt = task->dpu_per_node_cnt;
    task_put_sync->rail             = rail;

    memcpy(&task_put_sync->src_v.counts, coll_args->src.info_v.counts,        team->size * sizeof(ucc_count_t));
    memcpy(&task_put_sync->src_v.displs, coll_args->src.info_v.displacements, team->size * sizeof(ucc_count_t));
    memcpy(&task_put_sync->dst_v.counts, coll_args->dst.info_v.counts,        team->size * sizeof(ucc_count_t));
    memcpy(&task_put_sync->dst_v.displs, coll_args->dst.info_v.displacements, team->size * sizeof(ucc_count_t));

    ucc_tl_dpu_init_rkeys(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_coll_init(ucc_base_coll_args_t      *coll_args,
                                         ucc_base_team_t    *team,
                                         ucc_coll_task_t    **task_h)
{
    ucc_tl_dpu_team_t    *tl_team = ucc_derived_of(team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_context_t *ctx     = UCC_TL_DPU_TEAM_CTX(tl_team);
    ucc_tl_dpu_task_t    *task;
    ucc_status_t          status;

    /* FIXME: unsupported collectives should be excluded by score */
    if (!(coll_args->args.coll_type & UCC_TL_DPU_SUPPORTED_COLLS) ||
        coll_args->args.dst.info.count < 2) {
        tl_info(team->context->lib, "coll type %d not supported", coll_args->args.coll_type);
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_mpool_get(&ctx->req_mp);

    ucc_coll_task_init(&task->super, coll_args, team);
    tl_info(UCC_TASK_LIB(task), "task %p initialized", task);

    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));

    assert(ctx->dpu_per_node_cnt > 0);

    /* Misc init stuff */
    task->team                       = tl_team;
    task->super.post                 = ucc_tl_dpu_coll_start;
    task->super.progress             = ucc_tl_dpu_coll_progress;
    task->super.finalize             = ucc_tl_dpu_coll_finalize;
    task->super.triggered_post       = ucc_triggered_post;
    task->status                     = UCC_TL_DPU_TASK_STATUS_INIT;
    task->dpu_per_node_cnt           = ctx->dpu_per_node_cnt;

    for (int rail = 0; rail < ctx->dpu_per_node_cnt; rail++) {
        task->dpu_task_list[rail].status = UCC_TL_DPU_TASK_STATUS_INIT;
        tl_team->dpu_sync_list[rail].coll_id_issued = ++(ctx->dpu_ctx_list[rail].coll_id_issued);
    }

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_dpu_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_dpu_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_dpu_alltoallv_init(task);
        break;
    default:
        tl_error(UCC_TASK_LIB(task), "coll type %d not supported", coll_args->args.coll_type);
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        ucc_mpool_put(task);
        return status;
    }

    tl_info(UCC_TASK_LIB(task), "init coll req %p coll id %d", 
            task, tl_team->dpu_sync_list[0].coll_id_issued);
    *task_h = &task->super;

    // sleep(30);
    status = ucc_tl_dpu_send_dpu_task(task, ctx);
    if (status != UCC_OK) {
        ucc_mpool_put(task);
    }
    return status;
}