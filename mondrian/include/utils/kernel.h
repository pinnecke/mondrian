//
// Created by gabriel on 06.03.17.
//

#ifndef MONDRIAN_KERNEL_H
#define MONDRIAN_KERNEL_H

#include <printf.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>

#include <utils/sconfig.h>
#include "log.h"

#define MND_VERSION "0.009"

typedef u_int64_t u64;

typedef u_int64_t GlobalAttributeId;

typedef u_int64_t GlobalTableId;

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

#define return_ok() { return MND_OK; }

#define return_failure(error_code) { warning("Function call returned with failure: " #error_code); return error_code; }

static char string_buffer[5 * 2048];

#define format_string(out_ref, str, ...)      \
{                                             \
    sprintf(string_buffer, str, __VA_ARGS__); \
    *out_ref = malloc(strlen(string_buffer)); \
    strcpy(*out_ref, string_buffer);          \
}

static inline void *monitored_malloc(size_t size)
{
    void *ptr = malloc(size);

    //debugf("Allocate %zu bytes at address %p", size, ptr);
//    if_warning(if (ptr == NULL) warningf("Unable to allocate %zu bytes in host memory ", size);)

    return ptr;
}

#ifdef ZERO_MEMORY
static void *__mnd_malloc(size_t size) {
        void *host_ptr = NULL;
        if ((host_ptr = monitored_malloc(size)) != NULL) {
            //debugf("Zero memory at address %p, len %zu", host_ptr, size);
            memset(host_ptr, 0, size);
        }
        return host_ptr;
    }

    #define mnd_malloc(size)  __mnd_malloc(size)
#else
    #define mnd_malloc(size)  malloc_with_debug(size)
#endif




#endif
