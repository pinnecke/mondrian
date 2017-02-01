#pragma once

#include <Pantheon/stddef.h>

namespace Pantheon
{
    namespace Container
    {
        struct ArrayList
        {
            BYTE *pData;
            QWORD qwNumberOfElements;
            QWORD qwCapacityOfElements;
            QWORD qwElementSizeInByte;
            float fGrowFactor;

            struct Iterator
            {
                const BYTE *pBegin;
                const BYTE *pEnd;
                QWORD qwStepSize;
            };

            static PRESULT Create(retval ArrayList *pArrayList, QWORD qwCapacity, QWORD qwElementSize, float fGrowFactor);
            static PRESULT Dispose(retval ArrayList *pArrayList);

            static const BYTE *GetFront(const ArrayList *pArrayList);
            static const BYTE *GetBack(const ArrayList *pArrayList);
            static PRESULT Iterator(retval Iterator *pIterator, const ArrayList *pArrayList);
            static PRESULT IsEmpty(const ArrayList *pArrayList);

            static PRESULT Reserve(ArrayList *pArrayList, QWORD qwCapacity);
            static PRESULT FitToContent(ArrayList *pArrayList);
            static PRESULT Clear(ArrayList *pArrayList);

            static PRESULT Insert(retval nullable QWORD *pqwElementIndex, ArrayList *pArrayList, const BYTE *pData);
            static PRESULT Update(ArrayList *pArrayList, QWORD qwElementIndex, const BYTE *pData);
            static const BYTE *GetDataAt(const ArrayList *pArrayList, QWORD qwElementIndex);
            static PRESULT Remove(ArrayList *pArrayList, QWORD qwElementIndex);

            static PRESULT Swap(ArrayList *pArrayList, QWORD qwElementIndex1, QWORD qwElementIndex2);
        };
    }
}