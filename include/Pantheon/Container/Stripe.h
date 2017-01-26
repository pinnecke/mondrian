#include <Pantheon/error.h>
#include <Pantheon/stddef.h>

#ifndef GRIDSTORE_VECTOR_H
#define GRIDSTORE_VECTOR_H

namespace Pantheon {

    struct field {
        struct {
            BYTE IsNull : 1;
        } flags;
    };

// A Stripe is a generalization of a vector data structure and suitable for both n-nary storage model (NSM) and
// decomposed storage model (DSM)
    struct Stripe
    {
        void *Data;
        QWORD Capacity;
        QWORD Size;

        ErrorType Create(Stripe *Stripe, QWORD Capacity, QWORD ElementSize);

        ErrorType Dispose(Stripe *Stripe);

        const BYTE *At(const Stripe *Stripe, QWORD Index);

        void *Front(const Stripe *Stripe);

        void *Back(const Stripe *Stripe);

        ErrorType IsEmpty(const Stripe *Stripe);

        ErrorType GetNumberOfElements(retval QWORD *number, const Stripe *Stripe);

        ErrorType ReserveElements(Stripe *Stripe, QWORD NumberOfElements);

        ErrorType GetCapacity(retval QWORD *capacity, const Stripe *Stripe);

        ErrorType ShrinkToFit(Stripe *Stripe);

        ErrorType Clear(Stripe *Stripe);

        ErrorType PushBack(Stripe *Stripe, const BYTE *data);

        ErrorType Remove(Stripe *Stripe, QWORD index);

        ErrorType Swap(Stripe *Stripe, QWORD PositionLeft, QWORD PositionRight);
    };



}

#endif //GRIDSTORE_VECTOR_H
