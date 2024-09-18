# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# $COPYRIGHT$
# Additional copyrights may follow

CHECK_TLCP_REQUIRED("alltoall_block")

AS_IF([test "$CHECKED_TLCP_REQUIRED" = "y"],
[
    tlcp_modules="${tlcp_modules}:alltoall_block"
    tlcp_alltoall_block_enabled=y
], [])

AM_CONDITIONAL([TLCP_ALLTOALL_BLOCK_ENABLED], [test "$tlcp_alltoall_block_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/ucp/coll_plugins/alltoall_block/Makefile])
