#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <ucc/api/ucc.h>
#include <cuda_runtime.h>

#define MEMTYPE UCC_MEMORY_TYPE_CUDA

#define STR(x) #x
#define UCC_CHECK(_call)                                    \
  if (UCC_OK != (_call)) {                                  \
    fprintf(stderr, "*** UCC TEST FAIL: %s\n", STR(_call)); \
    exit(1);                                                \
  }
#define PTR_OFFSET(_ptr, _offset)                                              \
    ((void *)((ptrdiff_t)(_ptr) + (size_t)(_offset)))

struct test_config {
    int             num_ranks;
    ucc_coll_type_t coll_type;
    char            coll_type_name[32];
    size_t          msg_size;
} test_config_global;


ucc_coll_type_t str_to_coll_type(char *coll_type_str)
{
    if (!strcmp(coll_type_str, "allreduce")) {
        return UCC_COLL_TYPE_ALLREDUCE;
    } else if (!strcmp(coll_type_str, "allgather")) {
        return UCC_COLL_TYPE_ALLGATHER;
    } else {
        return UCC_COLL_TYPE_LAST;
    }
}

int process_args(int argc, char *argv[])
{
    int c;

    test_config_global.num_ranks = 2;
    test_config_global.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    test_config_global.msg_size  = 32;

    sprintf(test_config_global.coll_type_name, "%s", "allreduce");

    while ((c = getopt(argc, argv, "n:c:s:")) != -1) {
        switch (c) {
            case 'n':
                test_config_global.num_ranks = atoi(optarg);
                break;
            case 'c':
                test_config_global.coll_type = str_to_coll_type(optarg);
                strcpy(test_config_global.coll_type_name, optarg);
                break;
            case 's':
                test_config_global.msg_size = atoi(optarg);
                break;

        }
    }

    if (test_config_global.num_ranks < 2) {
        printf("min number of ranks is 2\n");
        exit(0);
    }
    if (test_config_global.coll_type == UCC_COLL_TYPE_LAST) {
        printf("collective operation %s is not supported\n", test_config_global.coll_type_name);
        exit(0);
    }
    printf("Multithtreaded CUDA test \n"
           "Number of ranks: %d\n"
           "Collective operation: %s\n"
           "Message size: %d\n",
           test_config_global.num_ranks,
           test_config_global.coll_type_name,
           test_config_global.msg_size);
    fflush(stdout);
}

typedef struct ucc_comm {
    ucc_lib_h     lib;
    ucc_context_h ctx;
    ucc_team_h    team;
    int           size;
    int           rank;
} ucc_comm_t;

typedef struct ucc_oob_shared_mem {
    int    iter;
    int    sbuf_ready;
    int    rbuf_ready;
    void  *sbuf;
    void  *rbuf;
    size_t size;
} ucc_oob_shared_mem_t;

typedef struct ucc_oob_coll_info {
    int                   rank;
    int                   size;
    ucc_oob_shared_mem_t *shmem;
} ucc_oob_coll_info_t;

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    ucc_oob_coll_info_t *info = (ucc_oob_coll_info_t*)coll_info;
    int                  rank = info->rank;

    info->shmem[rank].sbuf = sbuf;
    info->shmem[rank].rbuf = rbuf;
    info->shmem[rank].size = msglen;
    *req = info;

    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    ucc_oob_coll_info_t *info = (ucc_oob_coll_info_t*)req;
    int ready, r, rank;

    rank  = info->rank;
    info->shmem[rank].sbuf_ready = 1;

    if (rank == 0) {
        ready = 0;
        while (!ready) {
            ready = 1;
            for (r = 0; r < info->size; r++) {
                if (!info->shmem[r].sbuf_ready) {
                    ready = 0;
                    break;
                }
            }
        }
        for (r = 0; r < info->size; r++) {
            memcpy(PTR_OFFSET(info->shmem[rank].rbuf, r * info->shmem[rank].size),
                   info->shmem[r].sbuf,
                   info->shmem[rank].size);
        }
        for (r = 0; r < info->size; r++) {
            memcpy(info->shmem[r].rbuf,
                   info->shmem[rank].rbuf,
                   info->shmem[rank].size * info->size);
            info->shmem[r].rbuf_ready = 1;
        }
    } else {
        while(!info->shmem[rank].rbuf_ready);
    }

    return UCC_OK;
}

static ucc_status_t oob_allgather_free(void *req)
{
    volatile ucc_oob_coll_info_t *info = (ucc_oob_coll_info_t*)req;
    int ready, r, rank;

    rank = info->rank;

    if (rank == 0) {
        for (r = 0; r < info->size; r++) {
            info->shmem[r].sbuf_ready = 0;
        }
    } else {
        while(info->shmem[rank].sbuf_ready);
    }
    info->shmem[rank].rbuf_ready = 0;

    return UCC_OK;
}

static int create_ucc_comm(ucc_oob_coll_info_t *coll_info, ucc_comm_t *ucc_comm)
{
    ucc_lib_params_t lib_params;
    ucc_lib_config_h lib_config;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;
    ucc_team_params_t team_params;
    ucc_status_t status;

    ucc_comm->rank = coll_info->rank;
    ucc_comm->size = coll_info->size;

    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_MULTIPLE;

    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    UCC_CHECK(ucc_init(&lib_params, lib_config, &ucc_comm->lib));
    ucc_lib_config_release(lib_config);

    ctx_params.mask          = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test  = oob_allgather_test;
    ctx_params.oob.req_free  = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)coll_info;
    ctx_params.oob.n_oob_eps = ucc_comm->size;
    ctx_params.oob.oob_ep    = ucc_comm->rank;

    UCC_CHECK(ucc_context_config_read(ucc_comm->lib, NULL, &ctx_config));
    UCC_CHECK(ucc_context_create(ucc_comm->lib, &ctx_params, ctx_config,
                                 &ucc_comm->ctx));
    while (UCC_OK != ucc_context_progress(ucc_comm->ctx)) {
    }
    ucc_context_config_release(ctx_config);


    team_params.mask          = UCC_TEAM_PARAM_FIELD_OOB |
                                UCC_TEAM_PARAM_FIELD_EP |
                                UCC_TEAM_PARAM_FIELD_EP_RANGE;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void *)coll_info;
    team_params.oob.n_oob_eps = ucc_comm->size;
    team_params.oob.oob_ep    = ucc_comm->rank;
    team_params.ep            = ucc_comm->rank;
    team_params.ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;

    UCC_CHECK(ucc_team_create_post(&ucc_comm->ctx, 1, &team_params,
                                   &ucc_comm->team));
    while (UCC_INPROGRESS == (status = ucc_team_create_test(ucc_comm->team))) {
        UCC_CHECK(ucc_context_progress(ucc_comm->ctx));
    }
    if (UCC_OK != status) {
        fprintf(stderr, "failed to create ucc team\n");
        exit(1);
    }
    return 0;
}

void alloc_and_init_buf(int **data, ucc_memory_type_t mtype, int count, int rank)
{
    int i;

    if (mtype == UCC_MEMORY_TYPE_HOST) {
        *data = malloc(sizeof(int) * count);
        for (i = 0; i < count; i++) {
            (*data)[i] = rank + i;
        }
    } else if (mtype == UCC_MEMORY_TYPE_CUDA) {
        cudaMalloc((void**)data, sizeof(int) * count);
    } else {
        printf("memory type is not supported\n");
        exit(1);
    }
}


int check_and_free_buf(int *data, ucc_memory_type_t mtype, int count,
                       int comm_size, int comm_rank)
{
    int i;
    int current_value, expected_value;


    if (mtype == UCC_MEMORY_TYPE_HOST) {
        for (i = 0; i < count; i++) {
            current_value = data[i];
            expected_value = (comm_size - 1) * comm_size / 2 + comm_size * i;

            // Check if the current value matches the expected value
            if (current_value != expected_value) {
                printf("Error at rank %d: current value is %d, expected value is %d\n",
                    comm_rank, current_value, expected_value);
                return 1;
            }
        }
        printf("Rank %d passed the check\n", comm_rank);
        free(data);
    } else if (mtype == UCC_MEMORY_TYPE_CUDA) {
        cudaFree(data);
    } else {
        printf("memory type is not supported\n");
    }

    // All values passed the check
    return 0;
}

static int run_allreduce(ucc_comm_t *ucc_comm)
{
    int *data;
    int count = test_config_global.msg_size;
    ucc_coll_args_t args;
    ucc_coll_req_h req;

    alloc_and_init_buf(&data, UCC_MEMORY_TYPE_CUDA, count, ucc_comm->rank);
    args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    args.op = UCC_OP_SUM;
    args.dst.info.buffer = data;
    args.dst.info.count = count;
    args.dst.info.datatype = UCC_DT_INT32;
    args.dst.info.mem_type = MEMTYPE;
    args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;

    UCC_CHECK(ucc_collective_init(&args, &req, ucc_comm->team));
    UCC_CHECK(ucc_collective_post(req));
    while (UCC_OK != ucc_collective_test(req)) {
        UCC_CHECK(ucc_context_progress(ucc_comm->ctx));
    }
    UCC_CHECK(ucc_collective_finalize(req));

    check_and_free_buf(data, MEMTYPE, count,
                       ucc_comm->size, ucc_comm->rank);
    return 0;
}

static int run_allgather(ucc_comm_t *ucc_comm)
{
    int *dst_data, *src_data;
    int count = test_config_global.msg_size;
    ucc_coll_args_t args;
    ucc_coll_req_h req;
    int is_inplace = 0;

    alloc_and_init_buf(&dst_data, MEMTYPE, count * ucc_comm->size, ucc_comm->rank);
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    args.src.info.datatype = UCC_DT_INT32;
    args.src.info.mem_type = MEMTYPE;
    args.dst.info.datatype = UCC_DT_INT32;
    args.dst.info.mem_type = MEMTYPE;
    args.dst.info.buffer = dst_data;
    args.dst.info.count = count * ucc_comm->size;
    if (is_inplace) {
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    } else {
        args.flags = 0;
        alloc_and_init_buf(&src_data, MEMTYPE, count, ucc_comm->rank);
        args.src.info.buffer = src_data;
        args.src.info.count  = count;
    }

    UCC_CHECK(ucc_collective_init(&args, &req, ucc_comm->team));
    UCC_CHECK(ucc_collective_post(req));
    while (UCC_OK != ucc_collective_test(req)) {
        UCC_CHECK(ucc_context_progress(ucc_comm->ctx));
    }
    UCC_CHECK(ucc_collective_finalize(req));

    check_and_free_buf(dst_data, MEMTYPE, count,
                       ucc_comm->size, ucc_comm->rank);
    if (!is_inplace) {
        check_and_free_buf(src_data, MEMTYPE, count, ucc_comm->size,
                           ucc_comm->rank);
    }
}

static int destroy_ucc_comm(ucc_comm_t *ucc_comm)
{
    ucc_team_destroy(ucc_comm->team);
    ucc_context_destroy(ucc_comm->ctx);
    ucc_finalize(ucc_comm->lib);
    return 0;
}

void *start_thread(void *arg)
{
    ucc_oob_coll_info_t *coll_info = (ucc_oob_coll_info_t*)arg;
    ucc_comm_t ucc_comm;

    cudaSetDevice(coll_info->rank);
    // cudaSetDevice(0);
    create_ucc_comm(coll_info, &ucc_comm);

    switch (test_config_global.coll_type) {
        case UCC_COLL_TYPE_ALLREDUCE:
            run_allreduce(&ucc_comm);
            break;
        case UCC_COLL_TYPE_ALLGATHER:
            run_allgather(&ucc_comm);
            break;
        default:
            printf("unsupported collective operation %d\n", test_config_global.coll_type);
    }
    destroy_ucc_comm(&ucc_comm);
}

int main(int argc, char *argv[])
{
    int comm_size;
    ucc_oob_coll_info_t *coll_info;
    ucc_oob_shared_mem_t *shmem;
    pthread_t *ranks;
    int r;

    process_args(argc, argv);
    comm_size = test_config_global.num_ranks;
    ranks     = malloc(comm_size * sizeof(pthread_t));
    coll_info = malloc(comm_size * sizeof(ucc_oob_coll_info_t));
    shmem     = malloc(comm_size * sizeof(ucc_oob_shared_mem_t));

    for (r = 0; r < comm_size; r++) {
        coll_info[r].rank  = r;
        coll_info[r].size  = comm_size;
        coll_info[r].shmem = shmem;
        shmem->rbuf_ready = 0;
        shmem->sbuf_ready = 0;

        pthread_create(&ranks[r], NULL, start_thread, &coll_info[r]);
    }

    for (r = 0; r < comm_size; r++) {
        pthread_join(ranks[r], NULL);
    }
    free(ranks);
    free(coll_info);
    free(shmem);

    return 0;
}
