#ifndef GRIDSTORE_QUEUE_H
#define GRIDSTORE_QUEUE_H

#include <Pantheon/stdinc.h>
#include <Pantheon/error.h>
#include <Pantheon/stddef.h>

namespace Pantheon
{
    namespace Container
    {
        struct Queue
        {
            BYTE *Base, *Front, *Back;
            QWORD Capacity, ElementSize;
            float GrowFactor;

            static PRESULT Create(retval Queue *Queue, size_t Capacity, float GrowFactor, QWORD ElementSize);

            static PRESULT Dispose(Queue *Queue);

            static PRESULT Enqueue(Queue *Queue, const BYTE *data);

            static const BYTE *GetNewest(const Queue *queue);

            static const BYTE *GetOldest(const Queue *queue);

            static const BYTE *Deqeue(Queue *queue);

            static PRESULT IsEmpty(const Queue *queue);

            QWORD GetNumberOfElements(const Queue *queue);
        };
    }
}

#endif //GRIDSTORE_QUEUE_H
