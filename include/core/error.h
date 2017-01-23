#ifndef GRIDSTORE_ERROR_H
#define GRIDSTORE_ERROR_H

#include <stdio.h>

enum gsEnumErrors {
    gsFailed            = 0,
    gsFalse             = 0,
    GS_NO_OPERATION       = 0,
    GS_SUCCESS           = 1,
    gsTrue              = 1,
    GS_ILLEGAL_ARGUMENT,
    gsIllegalOperation,
    gsHostMallocFailed,
    gsHostReallocFailed,
    gsAlreadyFreed,
    GS_ILLEGAL_STATE,
    gsUnequals,
    gsEquals,
    gsInternalError,
    GS_NO_ELEMENT
};

typedef enum gsEnumErrors gsError_t;

/*static __thread*/ gsError_t gsLastError;

void gsPrintError(FILE *stream, gsError_t error);

#endif
