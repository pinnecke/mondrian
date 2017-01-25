#ifndef GRIDSTORE_ERROR_H
#define GRIDSTORE_ERROR_H

#include <stdio.h>

enum pan_error {
    PE_FAILED            = 0,
    PE_FALSE             = 0,
    PE_NOP               = 0,
    PE_SUCCESS           = 1,
    PE_TRUE              = 1,
    PE_ILLEGAL_ARG,
    PE_ILLEGAL_OPP,
    PE_HOST_MALLOC_FAILED,
    PE_HOST_REALLOC_FAILED,
    PE_ALREADY_FREED,
    PE_ILLEGAL_STATE,
    PE_UNEQUALS,
    PE_EQUALS,
    PE_NO_ELEMENT,
    PE_NO_COMPARATOR_DEFINED
};

/*static __thread*/ enum pan_error pan_last_err;

void gsPrintError(FILE *stream, enum pan_error error);

#endif
