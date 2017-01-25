#include <core/collections/queue.h>

enum pan_error queue_create(struct queue *out, size_t capacity, float grow_factor, size_t element_size)
{
    if (out == NULL || capacity == 0 || grow_factor <= 1 || element_size == 0)
        return PE_ILLEGAL_ARG;
    if ((out->base = malloc(element_size * capacity)) == NULL)
        return PE_HOST_MALLOC_FAILED;
    out->front = out->back = out->base;
    out->element_size = element_size;
    out->capacity = capacity;
    out->grow_factor = grow_factor;
    return PE_SUCCESS;
}

enum pan_error queue_dispose(struct queue *out)
{
    if (out == NULL)
        return PE_ILLEGAL_ARG;
    if (out->front == NULL)
        return PE_ALREADY_FREED;
    free (out->front);
    out->front = NULL;
    out->capacity = out->element_size = 0;
    return PE_SUCCESS;
}

enum pan_error queue_add(struct queue *out, const void *data)
{
    if (out == NULL || data == NULL)
        return PE_ILLEGAL_ARG;
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
            return PE_HOST_REALLOC_FAILED;
        out->back = out->base + (currentSize * out->element_size);
    }

    memcpy(out->back, data, out->element_size);
    out->back += out->element_size;
    out->front = out->base + currentFrontOffset;
    return PE_SUCCESS;
}

const void *queue_newest(const struct queue *queue)
{
    if (queue == NULL) {
        pan_last_err = PE_ILLEGAL_ARG;
        return NULL;
    }
    return queue->back;
}

const void *queue_oldest(const struct queue *queue)
{
    if (queue == NULL) {
        pan_last_err = PE_ILLEGAL_ARG;
        return NULL;
    }
    return queue->front;
}

const void *queue_pop(struct queue *queue)
{
    if (queue == NULL) {
        pan_last_err = PE_ILLEGAL_ARG;
        return NULL;
    }
    if (queue->front >= queue->back) {
        pan_last_err = PE_ILLEGAL_OPP;
        return NULL;
    }
    const void *result = queue->front;
    queue->front += queue->element_size;
    return result;
}

enum pan_error queue_empty(const struct queue *queue)
{
    return ((queue == NULL) || (queue->front == queue->back)) ? PE_TRUE : PE_FALSE;
}

size_t queue_num_elements(const struct queue *queue)
{
    if (queue == NULL) {
        pan_last_err = PE_ILLEGAL_ARG;
        return 0;
    }
    return ((queue->front == queue->back) / queue->element_size);
}