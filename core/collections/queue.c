#include <core/collections/queue.h>

gsError_t queue_create(struct queue *out, size_t capacity, float grow_factor, size_t element_size)
{
    if (out == NULL || capacity == 0 || grow_factor <= 1 || element_size == 0)
        return GS_ILLEGAL_ARGUMENT;
    if ((out->base = malloc(element_size * capacity)) == NULL)
        return gsHostMallocFailed;
    out->front = out->back = out->base;
    out->element_size = element_size;
    out->capacity = capacity;
    out->grow_factor = grow_factor;
    return GS_SUCCESS;
}

gsError_t queue_dispose(struct queue *out)
{
    if (out == NULL)
        return GS_ILLEGAL_ARGUMENT;
    if (out->front == NULL)
        return gsAlreadyFreed;
    free (out->front);
    out->front = NULL;
    out->capacity = out->element_size = 0;
    return GS_SUCCESS;
}

gsError_t queue_add(struct queue *out, const void *data)
{
    if (out == NULL || data == NULL)
        return GS_ILLEGAL_ARGUMENT;
    size_t currentSize = (out->back - out->base) / out->element_size;
    size_t currentFrontOffset = (out->front - out->base);
    size_t currentFrontPosition = currentFrontOffset / out->element_size;

    if (currentFrontPosition > out->capacity / 2) {
        size_t distanceBackFront = out->back - out->front;
        memmove(out->base, out->front, distanceBackFront);
        out->front = out->base;
        out->back = out->front + distanceBackFront;
    }

    if (currentSize + 1 >= out->capacity) {
        size_t newCapacity = out->capacity * out->grow_factor;
        if ((out->base = realloc(out->base, out->element_size * newCapacity)) == NULL)
            return gsHostReallocFailed;
        out->back = out->base + (currentSize * out->element_size);
    }

    memcpy(out->back, data, out->element_size);
    out->back += out->element_size;
    out->front = out->base + currentFrontOffset;
    return GS_SUCCESS;
}

const void *queue_newest(const struct queue *queue)
{
    if (queue == NULL) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    return queue->back;
}

const void *queue_oldest(const struct queue *queue)
{
    if (queue == NULL) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    return queue->front;
}

const void *queue_pop(struct queue *queue)
{
    if (queue == NULL) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    if (queue->front >= queue->back) {
        gsLastError = gsIllegalOperation;
        return NULL;
    }
    const void *result = queue->front;
    queue->front += queue->element_size;
    return result;
}

gsError_t queue_empty(const struct queue *queue)
{
    if (queue == NULL) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    return (queue->front == queue->back) ? gsTrue : gsFalse;
}

size_t queue_num_elements(const struct queue *queue)
{
    if (queue == NULL) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    return ((queue->front == queue->back) / queue->element_size);
}