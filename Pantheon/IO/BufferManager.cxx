#include <Pantheon/IO/BufferManager.h>

namespace Pantheon
{
    namespace IO
    {
        PRESULT BufferManager::Create(retval BufferManager *pBuffer, QWORD qwNumberOfInMemorySlots,
                                      function<Slot*(const BufferManager*)> fnNextVictim,
                                      SourceBinding sbSourceBinding)
        {
            using namespace Container;
            PRESULT presult;

            if (pBuffer == NULL || qwNumberOfInMemorySlots < 1)
                return PRESULT::IllegalArgument;

            pBuffer->fnNextVictim = fnNextVictim;
            pBuffer->sbSourceBinding = sbSourceBinding;

            if ((presult = ArrayList::Create(&pBuffer->alSlots, qwNumberOfInMemorySlots,
                                        sizeof(BufferManager::Slot),
                                        Config::StaticConfig::BufferManagerGrowFactor)) != PRESULT::OK)
                return presult;

            for (QWORD slotId = 0; slotId < qwNumberOfInMemorySlots; slotId++) {
                Slot slot(slotId);
                if ((presult = ArrayList::Insert(nullptr, &pBuffer->alSlots, (const BYTE *) &slot)) != PRESULT::OK)
                    return presult;
            }

            return PRESULT::OK;
        }

        PRESULT BufferManager::GetStatistics(retval Statistics *pStatistics, const BufferManager *pBufferManager)
        {
            if ((pStatistics == nullptr) || (pBufferManager == nullptr))
                return PRESULT::IllegalArgument;
            auto numberOfElements = pBufferManager->alSlots.qwNumberOfElements;
            auto capacityOfElements = pBufferManager->alSlots.qwCapacityOfElements;
            auto slotContentSize = pBufferManager->alSlots.qwElementSizeInByte;
            pStatistics->BytesInUse = numberOfElements * slotContentSize;
            pStatistics->BytesReserved = (capacityOfElements - numberOfElements) * slotContentSize;
            return PRESULT::OK;
        }

        PRESULT BufferManager::Insert(retval Slot *pSlot, const BufferManager *pBuffer, const BYTE *pData, QWORD qwSize)
        {
            using namespace Container;
            PRESULT presult;
            QWORD nextSlotId;

            if ((pSlot == nullptr) || (pBuffer == nullptr) || (pData == nullptr) || (qwSize < 1))
                return PRESULT::IllegalArgument;



            return PRESULT::OK;
        }

        PRESULT BufferManager::Update(Slot *pSlot, const BufferManager *buffer, const BYTE *pData, QWORD qwSize)
        {

        }

        PRESULT BufferManager::Delete(Slot *pSlot, const BufferManager *buffer)
        {

        }

        const BYTE *BufferManager::Read(const Slot *pSlot, const BufferManager *buffer)
        {

        }

        PRESULT BufferManager::GetNumberOfSlots(retval QWORD *qwNumberOfSlots, const BufferManager *buffer)
        {

        }

        PRESULT BufferManager::ListSlots(retval Slot *pSlots, const BufferManager *buffer)
        {

        }
    }
}

