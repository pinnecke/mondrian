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

            static PRESULT Create(retval RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize);

            static PRESULT GetSlot(retval QWORD *Slot, RecycleBuffer *Buffer);

            static PRESULT RemoveSlot(RecycleBuffer *Buffer, QWORD Slot);

            static PRESULT PutData(RecycleBuffer *Buffer, QWORD Slot, const BYTE *Data);

            static PRESULT GetData(retval BYTE **data, const RecycleBuffer *Buffer, QWORD Slot);

            static PRESULT Dispose(RecycleBuffer *Buffer);

        private:

            static PRESULT InitBuffer(RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize);

            static PRESULT InitFreeList(RecycleBuffer *Buffer, QWORD Capacity);

            static bool IsFreeListFilled(RecycleBuffer *Buffer);

            static void ExpandFreeList(RecycleBuffer *Buffer);

            static QWORD PopFromFreeList(RecycleBuffer *Buffer);

            static bool IsSlotFreed(const RecycleBuffer *Buffer, QWORD Slot_id);
        };
    }
}

