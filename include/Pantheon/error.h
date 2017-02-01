#pragma once

#include <stdio.h>

namespace Pantheon {

    enum class PRESULT
    {
        Failed = 0,
        False = 0,
        NoOperation = 0,
        OK = 1,
        True = 1,
        IllegalArgument,
        IllegalOperation,
        HostMallocFailed,
        HostReallocFailed,
        AlreadyFreed,
        IllegalState,
        Unequals,
        Equals,
        NoSuchElement,
        OutOfBounds
    };

    namespace DiagnosticService {

        void PrintError(FILE *stream, enum PRESULT error);
        PRESULT GetLastError();
        void SetLastError(PRESULT error);

    }
}

