#include <Pantheon/Container/Queue.h>

using namespace Pantheon;
using namespace Pantheon::Container;

ErrorType Queue::Create(Queue *Queue, size_t Capacity, float GrowFactor, QWORD ElementSize)
{
    if (Queue == NULL || Capacity == 0 || GrowFactor <= 1 || ElementSize == 0)
        return ErrorType::IllegalArgument;
    if ((Queue->Base = (BYTE *) malloc(ElementSize * Capacity)) == NULL)
        return ErrorType::HostMallocFailed;
    Queue->Front = Queue->Back = Queue->Base;
    Queue->ElementSize = ElementSize;
    Queue->Capacity = Capacity;
    Queue->GrowFactor = GrowFactor;
    return ErrorType::Success;
}

ErrorType Queue::Dispose(Queue *Queue)
{
    if (Queue == nullptr)
        return ErrorType::IllegalArgument;
    if (Queue->Front == nullptr)
        return ErrorType::AlreadyFreed;
    free (Queue->Front);
    Queue->Front = NULL;
    Queue->Capacity = Queue->ElementSize = 0;
    return ErrorType::Success;
}

enum ErrorType Queue::Enqueue(Queue *Queue, const BYTE *data)
{
    if (Queue == nullptr || data == nullptr)
        return ErrorType::IllegalArgument;
    size_t currentSize = (Queue->Back - Queue->Base) / Queue->ElementSize;
    size_t currentFrontOffset = (Queue->Front - Queue->Base);
    size_t currentFrontPosition = currentFrontOffset / Queue->ElementSize;

    if (currentFrontPosition > Queue->Capacity / 2) {
        size_t distanceBackFront = Queue->Back - Queue->Front;
        memmove(Queue->Base, Queue->Front, distanceBackFront);
        Queue->Front = Queue->Base;
        Queue->Back = Queue->Front + distanceBackFront;
    }

    if (currentSize + 1 >= Queue->Capacity) {
        size_t newCapacity = Queue->Capacity * Queue->GrowFactor;
        if ((Queue->Base = (BYTE *) realloc(Queue->Base, Queue->ElementSize * newCapacity)) == NULL)
            return ErrorType::HostReallocFailed;
        Queue->Back = Queue->Base + (currentSize * Queue->ElementSize);
    }

    memcpy(Queue->Back, data, Queue->ElementSize);
    Queue->Back += Queue->ElementSize;
    Queue->Front = Queue->Base + currentFrontOffset;
    return ErrorType::Success;
}

const BYTE *Queue::GetNewest(const Queue *Queue)
{
    if (Queue == nullptr) {
        DiagnosticService::SetLastError(ErrorType::IllegalArgument);
        return nullptr;
    }
    return Queue->Back;
}

const BYTE *Queue::GetOldest(const Queue *Queue)
{
    if (Queue == nullptr) {
        DiagnosticService::SetLastError(ErrorType::IllegalArgument);
        return nullptr;
    }
    return Queue->Front;
}

const BYTE *Queue::Deqeue(Queue *Queue)
{
    if (Queue == nullptr) {
        DiagnosticService::SetLastError(ErrorType::IllegalArgument);
        return nullptr;
    }
    if (Queue->Front >= Queue->Back) {
        DiagnosticService::SetLastError(ErrorType::IllegalOperation);
        return nullptr;
    }
    const BYTE *result = Queue->Front;
    Queue->Front += Queue->ElementSize;
    return result;
}

enum ErrorType Queue::IsEmpty(const Queue *Queue)
{
    return ((Queue == nullptr) || (Queue->Front == Queue->Back)) ? ErrorType::True : ErrorType::False;
}

QWORD Queue::GetNumberOfElements(const Queue *Queue)
{
    if (Queue == nullptr) {
        DiagnosticService::SetLastError(ErrorType::IllegalArgument);
        return 0;
    }
    return ((Queue->Front == Queue->Back) / Queue->ElementSize);
}