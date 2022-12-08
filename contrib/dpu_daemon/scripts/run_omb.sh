#!/bin/bash

BSZ=$((4))
ESZ=$((128 * 1024 * 1024))
MEMSZ=$((16 * 1024 * 1024 * 1024))
ITER=200
WARM=20

HOSTS=${1}
PPN=${2:-1}
NPROCS=$(($HOSTS*$PPN))

NBUF=4
BUFSZ=$((128*1024))

WORKSPACE="/global/scratch/users/sourav"
hostfile="$WORKSPACE/build-x86/hostfile.cpu"

UCX_DIR=$HPCX_UCX_DIR
UCC_DIR="$WORKSPACE/build-x86/ucc"
MPI_DIR=$HPCX_MPI_DIR
OMB_DIR=$HPCX_OSU_DIR

mcaopts+="--mca pml ucx --mca btl '^openib' "
mcaopts+="--mca opal_common_ucx_opal_mem_hooks 1 "
mcaopts+="--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 --mca coll_ucc_verbose 0  "

uccopts+="-x UCC_TL_DPU_TUNE=0-64:0 "
uccopts+="-x UCC_LOG_LEVEL=warn "
uccopts+="-x UCC_CL_BASIC_TLS=ucp,dpu "
uccopts+="-x UCC_TL_DPU_PIPELINE_BLOCK_SIZE=$BUFSZ "
uccopts+="-x UCC_TL_DPU_PIPELINE_BUFFERS=$NBUF "
uccopts+="-x UCC_TL_DPU_HOST_DPU_LIST=host_to_dpu.list "
uccopts+="-x UCX_NET_DEVICES=mlx5_4:1 "
uccopts+="-x UCX_TLS=self,rc_x "
uccopts+="-x UCX_MAX_RNDV_RAILS=1 "

rtopts+="-x LD_LIBRARY_PATH=$UCC_DIR/lib:$LD_LIBRARY_PATH  "


omb_exe="$OMB_DIR/osu_allreduce"
omb_opts+="-i $ITER -x $WARM -M $MEMSZ -m $BSZ:$ESZ -f "

cmd="${MPI_DIR}/bin/mpirun -np ${NPROCS} \
    --bind-to none \
    --map-by ppr:$PPN:node --hostfile ${hostfile} \
    ${mcaopts} ${uccopts} ${ncclopts} ${rtopts} \
    ${omb_exe} ${omb_opts} \
    "

echo $cmd
eval "$cmd"
