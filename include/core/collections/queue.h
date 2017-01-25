#ifndef GRIDSTORE_QUEUE_H
#define GRIDSTORE_QUEUE_H

#include <core/stdinc.h>
#include <core/error.h>

struct queue {
    void *base, *front, *back;
    size_t capacity, element_size;
    float grow_factor;
};

enum pan_error queue_create(struct queue *out, size_t capacity, float grow_factor, size_t element_size);

enum pan_error queue_dispose(struct queue *out);

enum pan_error queue_add(struct queue *out, const void *data);

const void *queue_newest(const struct queue *queue);

const void *queue_oldest(const struct queue *queue);

const void *queue_pop(struct queue *queue);

enum pan_error queue_empty(const struct queue *queue);

size_t queue_num_elements(const struct queue *queue);

#endif //GRIDSTORE_QUEUE_H
