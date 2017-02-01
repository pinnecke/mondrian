#include <Pantheon/Container/ArrayList.h>

namespace Pantheon
{
    namespace Container
    {
        PRESULT ArrayList::Create(retval ArrayList *pArrayList, QWORD qwCapacity, QWORD qwElementSize, float fGrowFactor)
        {
            if ((pArrayList == nullptr) || (qwCapacity == 0) || (qwElementSize == 0) || (fGrowFactor <= 1.00001f))
                return PRESULT::IllegalArgument;
            pArrayList->fGrowFactor = fGrowFactor;
            pArrayList->qwCapacityOfElements = qwCapacity;
            pArrayList->qwElementSizeInByte = qwElementSize;
            pArrayList->qwNumberOfElements = 0;
            if ((pArrayList->pData = (BYTE *) malloc(qwElementSize * qwCapacity)) == nullptr)
                return PRESULT::HostMallocFailed;
            return PRESULT::OK;
        }

        PRESULT ArrayList::Dispose(retval ArrayList *pArrayList)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr))
                return PRESULT::IllegalArgument;
            free (pArrayList->pData);
            pArrayList->pData = nullptr;
            return PRESULT::OK;
        }

        const BYTE *ArrayList::GetFront(const ArrayList *pArrayList)
        {
            return GetDataAt(pArrayList, 0);
        }

        const BYTE *ArrayList::GetBack(const ArrayList *pArrayList)
        {
            if (pArrayList == nullptr) {
                DiagnosticService::SetLastError(PRESULT::IllegalArgument);
                return nullptr;
            }
            return GetDataAt(pArrayList, pArrayList->qwNumberOfElements - 1);
        }

        PRESULT ArrayList::Iterator(retval Iterator *pIterator, const ArrayList *pArrayList)
        {
            if ((pIterator == nullptr) || (pArrayList == nullptr) || (pArrayList->pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            pIterator->pBegin = GetFront(pArrayList);
            pIterator->pEnd = GetBack(pArrayList);
            pIterator->qwStepSize = pArrayList->qwElementSizeInByte;
            return PRESULT::OK;
        }

        PRESULT ArrayList::IsEmpty(const ArrayList *pArrayList)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            return (pArrayList->qwNumberOfElements == 0)? PRESULT::True : PRESULT::False;
        }

        PRESULT ArrayList::Reserve(ArrayList *pArrayList, QWORD qwCapacity)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            if (qwCapacity < pArrayList->qwCapacityOfElements) {
                return PRESULT::NoOperation;
            }
            BYTE *ptr = (BYTE *) realloc(pArrayList->pData, pArrayList->qwElementSizeInByte * qwCapacity);
            if (ptr == NULL) {
                return PRESULT::HostReallocFailed;
            } else {
                pArrayList->pData = ptr;
                pArrayList->qwCapacityOfElements = qwCapacity;
            }

            return PRESULT::OK;
        }

        PRESULT ArrayList::FitToContent(ArrayList *pArrayList)
        {
            if ((pArrayList == nullptr) || pArrayList->pData == nullptr) {
                return PRESULT::IllegalArgument;
            }
            if (pArrayList->qwNumberOfElements == pArrayList->qwCapacityOfElements) {
                return PRESULT::NoOperation;
            }
            BYTE *ptr = (BYTE *) realloc(pArrayList->pData, pArrayList->qwElementSizeInByte * pArrayList->qwNumberOfElements);
            if (ptr == NULL) {
                return PRESULT::HostReallocFailed;
            } else {
                pArrayList->pData = ptr;
                pArrayList->qwCapacityOfElements = pArrayList->qwNumberOfElements;
            }

            return PRESULT::OK;
        }

        PRESULT ArrayList::Clear(ArrayList *pArrayList)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            pArrayList->qwNumberOfElements = 0;
            return PRESULT::OK;
        }

#define GET_OFFSET_OF_ELEMENT(index) \
        { pArrayList->pData + (index * pArrayList->qwElementSizeInByte) }


        PRESULT ArrayList::Insert(retval nullable QWORD *pqwElementIndex, ArrayList *pArrayList, const BYTE *pData)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr) || (pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            if ((pArrayList->qwNumberOfElements + 1) == pArrayList->qwCapacityOfElements) {
                PRESULT result;
                QWORD newCapacity = pArrayList->qwCapacityOfElements * pArrayList->fGrowFactor;
                if ((result = Reserve(pArrayList, newCapacity)) != PRESULT::OK) {
                    return result;
                }
            }
            memcpy(GET_OFFSET_OF_ELEMENT(pArrayList->qwNumberOfElements), pData, pArrayList->qwElementSizeInByte);
            if (pqwElementIndex != nullptr) {
                *pqwElementIndex = pArrayList->qwNumberOfElements;
            }
            pArrayList->qwNumberOfElements++;
            return PRESULT::OK;
        }

        PRESULT ArrayList::Update(ArrayList *pArrayList, QWORD qwElementIndex, const BYTE *pData)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr) || (pData == nullptr)) {
                return PRESULT::IllegalArgument;
            }
            if (qwElementIndex >= pArrayList->qwNumberOfElements) {
                return PRESULT::OutOfBounds;
            }
            memcpy(GET_OFFSET_OF_ELEMENT(qwElementIndex), pData, pArrayList->qwElementSizeInByte);
            return PRESULT::OK;
        }

        const BYTE *ArrayList::GetDataAt(const ArrayList *pArrayList, QWORD qwElementIndex)
        {
            if ((pArrayList == nullptr) || (pArrayList->pData == nullptr)) {
                DiagnosticService::SetLastError(PRESULT::IllegalArgument);
                return nullptr;
            }
            if (pArrayList->qwNumberOfElements == 0) {
                DiagnosticService::SetLastError(PRESULT::NoSuchElement);
                return nullptr;
            }
            if (qwElementIndex >= pArrayList->qwNumberOfElements)
            {
                DiagnosticService::SetLastError(PRESULT::OutOfBounds);
                return nullptr;
            }
            return pArrayList->pData + (qwElementIndex * pArrayList->qwElementSizeInByte);
        }

        PRESULT ArrayList::Remove(ArrayList *pArrayList, QWORD qwElementIndex)
        {
            // TODO
            return PRESULT::Failed;
        }

        PRESULT ArrayList::Swap(ArrayList *pArrayList, QWORD qwElementIndex1, QWORD qwElementIndex2)
        {
            // TODO
            return PRESULT::Failed;
        }
    }
}