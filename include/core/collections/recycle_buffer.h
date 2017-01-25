#ifndef GRIDSTORE_RECYCLE_BUFFER_H
#define GRIDSTORE_RECYCLE_BUFFER_H

#include <core/stdinc.h>
#include <core/stddef.h>

struct recycle_buffer_slot
{
    u64 idx;
    struct recycle_buffer_slot *next;
};

struct recycle_buffer
{
    void *base;
    u64 capacity;
    u64 element_size;
    u64 num_elements_freed;
    u64 num_elements_in_use;
    struct recycle_buffer_slot *free_list, *in_use_list;
};

enum pan_error recycle_buffer_create(struct recycle_buffer *out, u64 capacity, u64 element_size);

enum pan_error recycle_buffer_get_slot(u64 *out, struct recycle_buffer *buffer);

enum pan_error recycle_buffer_remove_slot(struct recycle_buffer *buffer, u64 slot);

enum pan_error recycle_buffer_put_data(struct recycle_buffer *buffer, u64 slot, const void *data);

const void *recycle_buffer_get_data(const struct recycle_buffer *buffer, u64 slot);

void recycle_buffer_get_data_batch(void **out, const struct recycle_buffer *buffer, u64 *slots, u64 num_slots);

enum pan_error recycle_buffer_get_num_slots_in_use(u64 *out, const struct recycle_buffer *buffer);

enum pan_error recycle_buffer_free(struct recycle_buffer *buffer);

#endif //GRIDSTORE_RECYCLE_BUFFER_H
