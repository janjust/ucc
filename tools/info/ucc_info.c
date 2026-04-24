/**
 * Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#include "ucc_info.h"
#include "core/ucc_global_opts.h"
#include "core/ucc_lib.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_log.h"
#include "utils/ucc_datastruct.h"
#include "components/tl/ucc_tl.h"
#include "components/cl/ucc_cl.h"
#include <getopt.h>
#include <stdlib.h>
#include <string.h>

static void usage()
{
    printf("Usage: ucc_info [options]\n");
    printf("At least one of the following options has to be set:\n");
    printf("  -v Show version information\n");
    printf("  -b Show build configuration\n");
    printf("  -c Show UCC configuration\n");
    printf("  -a Show also hidden configuration\n");
    printf("  -f Show fully decorated output\n");
    printf("  -s Show default components scores\n");
    printf("  -A Show collective algorithms and supported collectives\n");
    printf("  -h Show this help message\n");

    printf("\n");
}
extern ucc_list_link_t ucc_config_global_list;

static void print_algorithm_info(ucc_base_coll_alg_info_t *info)
{
    while (info->name) {
        printf("    %u : %16s : %s\n", info->id, info->name, info->desc);
        info++;
    }
}

static void print_default_algorithm_info(const char *component,
                                         const char *component_name,
                                         ucc_coll_type_t coll_type)
{
    printf("      %16s : supported by default %s/%s %s implementation "
           "(not a selectable algorithm id)\n", "default", component,
           component_name, ucc_coll_type_str(coll_type));
}

static void print_component_algs(ucc_base_coll_alg_info_t **alg_info,
                                 uint64_t supported_colls,
                                 const char *component,
                                 const char *component_name)
{
    int have_algs = 0;
    int i;

    for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
        if (alg_info[i] || (supported_colls & UCC_BIT(i))) {
            have_algs = 1;
            break;
        }
    }
    if (have_algs) {
        printf("%s/%s algorithms:\n", component, component_name);
        for (i = 0; i < UCC_COLL_TYPE_NUM; i++) {
            ucc_coll_type_t coll_type = (ucc_coll_type_t)UCC_BIT(i);

            if (alg_info[i] || (supported_colls & coll_type)) {
                printf("  %s\n", ucc_coll_type_str(coll_type));
            }

            if (alg_info[i]) {
                print_algorithm_info(alg_info[i]);
            } else if (supported_colls & coll_type) {
                print_default_algorithm_info(component, component_name,
                                             coll_type);
            }
        }
        printf("\n");
    }
}

static uint64_t get_cl_supported_colls(const ucc_lib_info_t *lib,
                                       const ucc_cl_iface_t *cl)
{
    int i;

    for (i = 0; i < lib->n_cl_libs_opened; i++) {
        if (lib->cl_libs[i]->iface == cl) {
            return lib->cl_attrs[i].super.attr.coll_types;
        }
    }
    return 0;
}

static uint64_t get_tl_supported_colls(const ucc_lib_info_t *lib,
                                       ucc_tl_iface_t *tl)
{
    ucc_tl_lib_attr_t attr;
    ucc_status_t      status;
    int               i;

    memset(&attr, 0, sizeof(attr));
    for (i = 0; i < lib->n_tl_libs_opened; i++) {
        if (lib->tl_libs[i]->iface == tl) {
            status = tl->lib.get_attr(&lib->tl_libs[i]->super, &attr.super);
            return (UCC_OK == status) ? attr.super.attr.coll_types : 0;
        }
    }

    status = tl->lib.get_attr(NULL, &attr.super);
    return (UCC_OK == status) ? attr.super.attr.coll_types : 0;
}

int main(int argc, char **argv)
{
    ucc_global_config_t *cfg = &ucc_global_config;
    ucc_config_print_flags_t print_flags;
    unsigned                 print_opts;
    int                      c, show_scores, show_algs;
    ucc_lib_h                lib;
    ucc_lib_config_h         config;
    ucc_lib_params_t         params;
    ucc_status_t             status;
    ucc_tl_iface_t *         tl;
    ucc_cl_iface_t *         cl;
    ucc_lib_info_t *         lib_info;

    print_flags = (ucc_config_print_flags_t)0;
    print_opts  = 0;
    show_scores = 0;
    show_algs   = 0;
    while ((c = getopt(argc, argv, "vbcafhsA")) != -1) {
        switch (c) {
        case 'f':
            print_flags |= (ucc_config_print_flags_t)(UCC_CONFIG_PRINT_CONFIG |
                                                      UCC_CONFIG_PRINT_HEADER |
                                                      UCC_CONFIG_PRINT_DOC);
            break;
        case 'c':
            print_flags |= (ucc_config_print_flags_t)UCC_CONFIG_PRINT_CONFIG;
            break;
        case 'a':
            print_flags |= (ucc_config_print_flags_t)UCC_CONFIG_PRINT_HIDDEN;
            break;
        case 'v':
            print_opts |= PRINT_VERSION;
            break;
        case 'b':
            print_opts |= PRINT_BUILD_CONFIG;
            break;
        case 's':
            show_scores = 1;
            break;
        case 'A':
            show_algs = 1;
            break;
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return -1;
        }
    }

    if ((print_opts == 0) && (print_flags == 0) && (!show_scores) &&
        (!show_algs)) {
        usage();
        return -2;
    }

    /* need to call ucc_init to force loading of dynamic
       ucc components */
    params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCC_THREAD_SINGLE;
    if (UCC_OK != ucc_lib_config_read(NULL, NULL, &config)) {
        return 0;
    }

    status = ucc_init(&params, config, &lib);
    ucc_lib_config_release(config);
    if (UCC_OK != status) {
        return 0;
    }

    if (print_opts & PRINT_VERSION) {
        print_version();
    }

    if (print_opts & PRINT_BUILD_CONFIG) {
        print_build_config();
    }

    if (print_flags & UCC_CONFIG_PRINT_CONFIG) {
        ucc_config_parser_print_all_opts(stdout, "UCC_", print_flags,
                                         &ucc_config_global_list);
    }
    lib_info = (ucc_lib_info_t *)lib;
    if (show_scores) {
        if (cfg->cl_framework.n_components) {
            printf("Default CLs scores:");
            for (c = 0; c < cfg->cl_framework.n_components; c++) {
                printf(" %s=%d", cfg->cl_framework.components[c]->name,
                       cfg->cl_framework.components[c]->score);
            }
            printf("\n");
        }
        if (cfg->tl_framework.n_components) {
            printf("Default TLs scores:");
            for (c = 0; c < cfg->tl_framework.n_components; c++) {
                printf(" %s=%d", cfg->tl_framework.components[c]->name,
                       cfg->tl_framework.components[c]->score);
            }
            printf("\n");
        }
    }
    if (show_algs) {
        for (c = 0; c < cfg->cl_framework.n_components; c++) {
            cl =
                ucc_derived_of(cfg->cl_framework.components[c], ucc_cl_iface_t);
            print_component_algs(cl->alg_info,
                                 get_cl_supported_colls(lib_info, cl),
                                 "cl", cl->super.name);
        }

        for (c = 0; c < cfg->tl_framework.n_components; c++) {
            tl =
                ucc_derived_of(cfg->tl_framework.components[c], ucc_tl_iface_t);
            print_component_algs(tl->alg_info,
                                 get_tl_supported_colls(lib_info, tl),
                                 "tl", tl->super.name);
        }
    }
    ucc_finalize(lib);
    return 0;
}
