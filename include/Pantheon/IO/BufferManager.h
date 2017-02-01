#pragma once

#include <Pantheon/error.h>
#include <Pantheon/stddef.h>
#include <Pantheon/Container/ArrayList.h>
#include <functional>
#include <Pantheon/Container/RecycleBuffer.h>

using namespace std;

namespace Pantheon
{
    namespace IO {

        struct BufferManager
        {
            struct Slot
            {
                QWORD qwSlotId;
                QWORD qwRecordId;
                QWORD qwRecordSizeInByte;
                struct {
                    BYTE isPinned : 1;
                    BYTE isFree : 1;
                };

                Slot(QWORD qwSlotId) : qwSlotId(qwSlotId), qwRecordId(0), qwRecordSizeInByte(0), isPinned(false),
                                       isFree(true) {}
            };

            struct Statistics
            {
                QWORD BytesInUse;
                QWORD BytesReserved;
                QWORD NumberOfSlots;
            };

            struct SourceBinding
            {
                function<PRESULT (retval QWORD *qwRecordId, BYTE *pData, QWORD qwSize)> fnInsertRecord;
                function<PRESULT (QWORD qwRecordId, BYTE *pData, QWORD qwSize)> fnUpdateRecord;
                function<PRESULT (retval QWORD qwRecordId)> fnDeleteRecord;
                function<PRESULT (retval QWORD *qwSize, QWORD qwRecordId)> fnGetSizeOfRecord;
                function<PRESULT (const BYTE *pData, QWORD qwRecordId)> fnReadRecord;
                function<PRESULT (retval QWORD *qwRecordId, QWORD qwRequiredSize)> fnNextFreeRecordId;
            };

            Container::ArrayList alSlots;
            function<Slot*()> fnNextVictim;
            SourceBinding sbSourceBinding;

            static PRESULT Create(retval BufferManager *pBuffer, QWORD qwNumberOfInMemorySlots,
                                  function<Slot*(const BufferManager*)> fnNextVictim,
                                  SourceBinding sbSourceBinding);

            static PRESULT Dispose(BufferManager *pBuffer);

            static PRESULT GetStatistics(retval Statistics *pStatistics, const BufferManager *pBufferManager);

            static PRESULT Insert(retval Slot *pSlot, const BufferManager *pBuffer, const BYTE *pData, QWORD qwSize);

            static PRESULT Update(Slot *pSlot, const BufferManager *buffer, const BYTE *pData, QWORD qwSize);

            static PRESULT Delete(Slot *pSlot, const BufferManager *buffer);

            static const BYTE *Read(const Slot *pSlot, const BufferManager *buffer);

            static PRESULT GetNumberOfSlots(retval QWORD *qwNumberOfSlots, const BufferManager *buffer);

            static PRESULT ListSlots(retval Slot *pSlots, const BufferManager *buffer);

        };

    }
}