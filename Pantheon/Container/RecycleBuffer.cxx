#include <Pantheon/Container/RecycleBuffer.h>

namespace Pantheon {
    namespace Container {

        PRESULT RecycleBuffer::Create(RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize) {
            if (Capacity == 0 || Capacity == 0)
                return PRESULT::IllegalArgument;
            if ((InitBuffer(Buffer, Capacity, ElementSize) != PRESULT::OK) ||
                (InitFreeList(Buffer, Capacity) != PRESULT::OK))
                return PRESULT::Failed;
            return PRESULT::OK;
        }

        enum PRESULT RecycleBuffer::GetSlot(QWORD *Slot, RecycleBuffer *Buffer) {
            if ((Slot == nullptr) || (Buffer == nullptr))
                return PRESULT::IllegalArgument;

            if (!IsFreeListFilled(Buffer))
                ExpandFreeList(Buffer);

            *Slot = PopFromFreeList(Buffer);
            return PRESULT::OK;
        }

        enum PRESULT RecycleBuffer::RemoveSlot(RecycleBuffer *Buffer, QWORD Slot) {
            assert (Buffer != nullptr);
            assert (Buffer->Base != nullptr);
            assert (Buffer->InUseList != nullptr);

            auto root = Buffer->InUseList;
            if (root->Index == Slot) {
                Buffer->InUseList = Buffer->InUseList->next;
                root->next = Buffer->FreeList;
                Buffer->FreeList = root;
                Buffer->NumberOfFreedElements++;
                Buffer->NumberOfElementsInUse--;
                return PRESULT::OK;
            }
            for (auto prev = root, it = root->next; it != nullptr; prev = it, it = it->next) {
                if (it->Index == Slot) {
                    prev->next = it->next;
                    it->next = Buffer->FreeList;
                    Buffer->FreeList = it;
                    Buffer->NumberOfFreedElements++;
                    Buffer->NumberOfElementsInUse--;
                    return PRESULT::OK;
                }
            }
            return PRESULT::NoSuchElement;
        }

        static inline BYTE *GetOffset(const RecycleBuffer *buffer, QWORD slot) {
            return buffer->Base + (slot * buffer->ElementSize);
        }

        PRESULT RecycleBuffer::PutData(RecycleBuffer *Buffer, QWORD Slot, const BYTE *Data) {
            if ((Buffer == nullptr) || (Slot >= Buffer->Capacity) || (Data == nullptr))
                return PRESULT::IllegalArgument;
            assert (!RecycleBuffer::IsSlotFreed(Buffer, Slot));
            auto offset = GetOffset(Buffer, Slot);
            auto size = Buffer->ElementSize;
            memset(offset, 0, size);
            memcpy(offset, Data, size);
            return PRESULT::OK;
        }

        PRESULT RecycleBuffer::GetData(retval BYTE **data, const RecycleBuffer *Buffer, QWORD Slot) {
            if ((Buffer == nullptr) || (Slot >= Buffer->Capacity)) {
                return PRESULT::IllegalArgument;
            }
            assert(!(IsSlotFreed(Buffer, Slot)));
            *data = GetOffset(Buffer, Slot);
            return PRESULT::OK;
        }

        enum PRESULT RecycleBuffer::Dispose(struct RecycleBuffer *Buffer) {
            // TODO
            return PRESULT::OK;
        }

        static inline PRESULT fill(RecycleBuffer::Slot **begin, QWORD beginValue, QWORD endValue) {
            using Slot = RecycleBuffer::Slot;

            Slot *head, *it;
            if ((head = (Slot *) malloc(sizeof(Slot)), MSG_HOST_MALLOC) == nullptr)
                return PRESULT::HostMallocFailed;

            head->Index = endValue - 1;
            head->next = nullptr;

            for (auto handlerId = head->Index; handlerId > beginValue; handlerId--) {
                if ((it = (Slot *) malloc(sizeof(Slot))) == NULL)
                    return PRESULT::HostMallocFailed;
                it->Index = handlerId - 1;
                it->next = head;
                head = it;
            }

            *begin = head;
            return PRESULT::OK;
        }

        static inline RecycleBuffer::Slot *tail(RecycleBuffer::Slot *begin) {
            while (begin->next != NULL) {
                begin = begin->next;
            }
            return begin;
        }

        bool RecycleBuffer::IsSlotFreed(const RecycleBuffer *Buffer, QWORD Slot) {
            for (auto it = Buffer->FreeList; it != nullptr; it = it->next) {
                if (it->Index == Slot)
                    return true;
            }
            return false;
        }

        QWORD RecycleBuffer::PopFromFreeList(RecycleBuffer *Buffer) {
            assert (Buffer != nullptr);
            assert (Buffer->Base != nullptr);
            assert (Buffer->FreeList != nullptr);
            auto element = Buffer->FreeList;
            Buffer->FreeList = element->next;
            QWORD result = element->Index;
            element->next = Buffer->InUseList;
            Buffer->InUseList = element;
            Buffer->NumberOfFreedElements--;
            Buffer->NumberOfElementsInUse++;
            return result;
        }

        bool RecycleBuffer::IsFreeListFilled(RecycleBuffer *buffer) {
            assert (buffer != nullptr);
            assert (buffer->FreeList != nullptr || buffer->InUseList != nullptr);
            return (buffer->FreeList != nullptr);
        }

        PRESULT RecycleBuffer::InitFreeList(RecycleBuffer *Buffer, QWORD Capacity) {
            assert (Buffer != nullptr);
            assert (Capacity > 0);
            fill(&Buffer->FreeList, 0, Capacity);
            Buffer->NumberOfFreedElements = Capacity;
            Buffer->NumberOfElementsInUse = 0;
            return PRESULT::OK;
        }

        void RecycleBuffer::ExpandFreeList(RecycleBuffer *Buffer) {
            QWORD capacityOld = Buffer->Capacity;
            QWORD capacityNew = capacityOld * Config::StaticConfig::RecycleBufferGrowFactor;
            Buffer->Capacity = capacityNew;
            REQUIRE_NON_NULL(Buffer->Base = (BYTE *) realloc(Buffer->Base, Buffer->ElementSize * capacityNew),
                             MSG_HOST_REALLOC)
            Buffer->NumberOfFreedElements += (capacityNew - capacityOld);
            fill(&Buffer->FreeList, capacityOld, capacityNew);
        }

        enum PRESULT RecycleBuffer::InitBuffer(RecycleBuffer *Buffer, QWORD Capacity, QWORD ElementSize) {
            assert (Capacity != 0);
            assert (ElementSize != 0);

            QWORD size = ElementSize * Capacity;
            if ((Buffer->Base = (BYTE *) malloc(size)) == NULL)
                return PRESULT::Failed;
            memset(Buffer->Base, 0, size);
            Buffer->Capacity = Capacity;
            Buffer->ElementSize = ElementSize;
            return PRESULT::OK;
        }
    }
}
