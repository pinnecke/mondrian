#include <core/collections/recycle_buffer.h>

#define GROW_FACT 1.4f

enum pan_error recycle_buffer_create(struct recycle_buffer *out, u64 capacity, u64 element_size)
{
    enum pan_error init_buffer(struct recycle_buffer *, u64, u64);
    enum pan_error init_free_list(struct recycle_buffer *, u64);

    if (out == NULL || capacity == 0 || element_size == 0)
        return PE_ILLEGAL_ARG;
    if ((init_buffer(out, capacity, element_size) != PE_SUCCESS) || (init_free_list(out, capacity) != PE_SUCCESS))
        return PE_FAILED;
    return PE_SUCCESS;
}

enum pan_error recycle_buffer_get_slot(u64 *out, struct recycle_buffer *buffer)
{
    bool free_list_filled(struct recycle_buffer *);
    void expand_free_list(struct recycle_buffer *);
    u64 pop_free_list(struct recycle_buffer *);

    if ((out == NULL) || (buffer == NULL))
        return PE_ILLEGAL_ARG;

    if (!free_list_filled(buffer)) {
        expand_free_list(buffer);
    }

    *out = pop_free_list(buffer);
    return PE_SUCCESS;
}

enum pan_error recycle_buffer_remove_slot(struct recycle_buffer *buffer, u64 slot)
{
    assert (buffer != NULL);
    assert (buffer->base != NULL);
    assert (buffer->in_use_list != NULL);

    struct recycle_buffer_slot *root = buffer->in_use_list;
    if (root->idx == slot) {
        buffer->in_use_list = buffer->in_use_list->next;
        root->next = buffer->free_list;
        buffer->free_list = root;
        buffer->num_elements_freed++;
        buffer->num_elements_in_use--;
        return PE_SUCCESS;
    }
    for (struct recycle_buffer_slot *prev = root, *it = root->next; it != NULL; prev = it, it = it->next) {
        if (it->idx == slot) {
            prev->next = it->next;
            it->next = buffer->free_list;
            buffer->free_list = it;
            buffer->num_elements_freed++;
            buffer->num_elements_in_use--;
            return PE_SUCCESS;
        }
    }

    return PE_NO_ELEMENT;
}

static inline void *get_offset_slot_data(const struct recycle_buffer *buffer, u64 slot)
{
    return buffer->base + (slot * buffer->element_size);
}

enum pan_error recycle_buffer_put_data(struct recycle_buffer *buffer, u64 slot, const void *data)
{
    bool is_freed_slot(const struct recycle_buffer *, u64);

    if ((buffer == NULL) || (slot >= buffer->capacity) || (data == NULL))
        return PE_ILLEGAL_ARG;

    assert (!is_freed_slot(buffer, slot));

    void *offset = get_offset_slot_data(buffer, slot);
    u64 size = buffer->element_size;
    memset(offset, 0, size);
    memcpy(offset, data, size);
    return PE_SUCCESS;
}

enum pan_error recycle_buffer_get_num_slots_in_use(u64 *out, const struct recycle_buffer *buffer)
{
    if ((out == NULL) || (buffer == NULL))
        return PE_ILLEGAL_ARG;
    *out = buffer->num_elements_in_use;
    return PE_SUCCESS;
}

const void *recycle_buffer_get_data(const struct recycle_buffer *buffer, u64 slot)
{
    bool is_freed_slot(const struct recycle_buffer *, u64);

    if ((buffer == NULL) || (slot >= buffer->capacity)) {
        pan_last_err = PE_ILLEGAL_ARG;
        return NULL;
    }

    assert(!(is_freed_slot(buffer, slot)));

    return get_offset_slot_data(buffer, slot);
}

enum pan_error recycle_buffer_free(struct recycle_buffer *buffer)
{
    // TODO
    return PE_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline enum pan_error fill(struct recycle_buffer_slot **begin, u64 begin_value, u64 end_value)
{
    struct recycle_buffer_slot *head, *it;
    if ((head = malloc(sizeof (struct recycle_buffer_slot)), MSG_HOST_MALLOC) == NULL)
        return PE_HOST_MALLOC_FAILED;

    head->idx = end_value - 1;
    head->next = NULL;

    for (u64 handlerId = head->idx; handlerId > begin_value; handlerId--) {
        if ((it = malloc(sizeof (struct recycle_buffer_slot))) == NULL)
            return PE_HOST_MALLOC_FAILED;
        it->idx = handlerId - 1;
        it->next = head;
        head = it;
    }

    *begin = head;
    return PE_SUCCESS;
}

static inline struct recycle_buffer_slot *tail(struct recycle_buffer_slot *begin)
{
    while (begin->next != NULL) {
        begin = begin->next;
    }
    return begin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_freed_slot(const struct recycle_buffer *buffer, u64 slot_id)
{
    for (struct recycle_buffer_slot *slot = buffer->free_list; slot != NULL; slot = slot->next) {
        if (slot->idx == slot_id)
            return true;
    }
    return false;
}

u64 pop_free_list(struct recycle_buffer *buffer)
{
    assert (buffer != NULL);
    assert (buffer->base != NULL);
    assert (buffer->free_list != NULL);
    struct recycle_buffer_slot *element = buffer->free_list;
    buffer->free_list = element->next;
    u64 result = element->idx;
    element->next = buffer->in_use_list;
    buffer->in_use_list = element;
    buffer->num_elements_freed--;
    buffer->num_elements_in_use++;
    return result;
}

bool free_list_filled(struct recycle_buffer *eventGroup)
{
    assert (eventGroup != NULL);
    assert (eventGroup->free_list != NULL || eventGroup->in_use_list != NULL);
    return (eventGroup->free_list != NULL);
}

enum pan_error init_free_list(struct recycle_buffer *buffer, u64 capacity)
{
    assert (buffer != NULL);
    assert (capacity > 0);
    fill(&buffer->free_list, 0, capacity);
    buffer->num_elements_freed = capacity;
    buffer->num_elements_in_use = 0;
    return PE_SUCCESS;
}

void expand_free_list(struct recycle_buffer *buffer)
{
    assert (buffer != NULL);

    u64 cap_old = buffer->capacity;
    u64 cap_new = cap_old * GROW_FACT;
    buffer->capacity = cap_new;
    REQUIRE_NON_NULL(buffer->base = realloc(buffer->base, buffer->element_size * cap_new),
                     MSG_HOST_REALLOC)
    buffer->num_elements_freed += (cap_new - cap_old);
    fill(&buffer->free_list, cap_old, cap_new);
}

enum pan_error init_buffer(struct recycle_buffer *group, u64 capacity, u64 element_size)
{
    assert (group != NULL);
    assert (capacity != 0);
    assert (element_size != 0);

    u64 size = element_size * capacity;
    if ((group->base = malloc(size)) == NULL)
        return PE_FAILED;
    memset(group->base, 0, size);
    group->capacity = capacity;
    group->element_size = element_size;
    return PE_SUCCESS;
}


