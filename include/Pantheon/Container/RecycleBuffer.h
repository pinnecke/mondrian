#pragma once

#include <Pantheon/stdinc.h>
#include <Pantheon/stddef.h>

namespace Pantheon
{
    namespace Container
    {
        struct RecycleBuffer
        {
            struct Slot {
                QWORD Index;
                Slot *next;
            };

            BYTE *Base;
            QWORD Capacity;
            QWORD ElementSize;
            QWORD NumberOfFreedElements;
            QWORD NumberOfElementsInUse;
            Slot *FreeList, *InUseList;

            static ErrorType Create(retval RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize);

            static ErrorType GetSlot(retval QWORD *Slot, RecycleBuffer *Buffer);

            static ErrorType RemoveSlot(RecycleBuffer *Buffer, QWORD Slot);

            static ErrorType PutData(RecycleBuffer *Buffer, QWORD Slot, const BYTE *Data);

            static ErrorType GetData(retval BYTE **data, const RecycleBuffer *Buffer, QWORD Slot);

            static ErrorType Dispose(RecycleBuffer *Buffer);

        private:

            static ErrorType InitBuffer(RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize);

            static ErrorType InitFreeList(RecycleBuffer *Buffer, QWORD Capacity);

            static bool IsFreeListFilled(RecycleBuffer *Buffer);

            static void ExpandFreeList(RecycleBuffer *Buffer);

            static QWORD PopFromFreeList(RecycleBuffer *Buffer);

            static bool IsSlotFreed(const RecycleBuffer *Buffer, QWORD Slot_id);
        };
    }
}

