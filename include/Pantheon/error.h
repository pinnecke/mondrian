#pragma once

#include <stdio.h>

namespace Pantheon {

    enum class ErrorType
    {
        Failed = 0,
        False = 0,
        NoOperation = 0,
        Success = 1,
        True = 1,
        IllegalArgument,
        IllegalOperation,
        HostMallocFailed,
        HostReallocFailed,
        AlreadyFreed,
        IllegalState,
        Unequals,
        Equals,
        NoSuchElement
    };

    namespace DiagnosticService {

        void PrintError(FILE *stream, enum ErrorType error);
        ErrorType GetLastError();
        void SetLastError(ErrorType error);

    }
}

