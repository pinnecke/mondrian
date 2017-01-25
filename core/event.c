#include <core/event.h>
#include <core/stddef.h>
#include <core/collections/queue.h>
#include <core/collections/recycle_buffer.h>

#define GROW_FACT 1.4f

struct func_ptr
{
    void (*func)(enum event_type, void *);
};

struct event
{
    enum event_type type;
    void *args;
};

struct global_sub
{
    enum event_type event;
    u64 local_id, global_id;
};

// The i-th entry in the following array is the i-th type defined in enum 'gsEvent_t'. For each entry, a list of
// registered type handler is stored
struct recycle_buffer event_register[_ENUM_EVENT_MAX];

struct recycle_buffer global_register;
struct queue posted_events;
bool is_initialized;

static inline void init_register()
{
    if (!is_initialized) {
        is_initialized = true;

        struct config_info config;
        REQUIRE_SUCCESS(config_get(&config), MSG_CONFIG_LOAD)
        u64 capacity = config.init_cap_event_groups;

        for (u64 event_type = 0; event_type < _ENUM_EVENT_MAX; event_type++) {
            struct recycle_buffer *buffer = event_register + event_type;
            REQUIRE_SUCCESS(recycle_buffer_create(buffer, capacity, sizeof(struct func_ptr)),
                            MSG_EVENT_SYSTEM_INIT_FAILED)
        }

        REQUIRE_SUCCESS(recycle_buffer_create(&global_register, config.max_sub_id, sizeof(struct global_sub)),
                        MSG_EVENT_SYSTEM_INIT_FAILED);

        REQUIRE_EQUAL(queue_create(&posted_events, 200, GROW_FACT, sizeof(struct event)), PE_SUCCESS, MSG_EVENT_QUEUE)
    }
}

static inline void install_callback(struct recycle_buffer *group, uint64_t id, void (*func)(enum event_type, void *))
{
    assert (group != NULL);
    assert (func != NULL);
    assert (id < group->capacity);
    struct func_ptr data = {
        .func = func
    };
    recycle_buffer_put_data(group, id, &data);
}

static inline enum pan_error create_global_id(u64 *out, enum event_type event, uint64_t local_id)
{
    assert (out != NULL);
    assert (event >= 0);
    assert (event < _ENUM_EVENT_MAX);

    u64 global_id;
    enum pan_error result;

    if ((result = recycle_buffer_get_slot(&global_id, &global_register)) != PE_SUCCESS) {
        return result;
    }

    struct global_sub data = {
        .event = event,
        .global_id = global_id,
        .local_id = local_id
    };

    if ((result =recycle_buffer_put_data(&global_register, global_id, &data)) != PE_SUCCESS) {
        return result;
    }

    *out = global_id;

    return PE_SUCCESS;
}

static inline struct global_sub * get_local_id(u64 global_id)
{
    return recycle_buffer_get_data(&global_register, global_id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

u64 events_subscribe(enum event_type type, void (*callback)(enum event_type, void *))
{
    REQUIRE_NON_NULL(callback, MSG_NULL_POINTER)
    REQUIRE_IN_RANGE(type, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    init_register();

    struct recycle_buffer *eventGroup = event_register + type;
    u64 localId;
    REQUIRE_SUCCESS(recycle_buffer_get_slot(&localId, eventGroup), GS_MSG_EVENT_HANDLER_REGISTRATION_FAILED);
    install_callback(eventGroup, localId, callback);

    u64 global_id;
    REQUIRE_SUCCESS(create_global_id(&global_id, type, localId), GS_MSG_EVENT_HANDLER_REGISTRATION_FAILED);
    return global_id;
}

static inline void remove_global_sub(u64 global_id) {
    REQUIRE_SUCCESS(recycle_buffer_remove_slot(&global_register, global_id),
                    MSG_REMOVE_EVENT_HANLDER_FAILED);
}

enum pan_error events_unsubscribe(u64 subscriber_id) {
    if(is_initialized != true)
        return PE_ILLEGAL_STATE;

    struct global_sub *local_id = get_local_id(subscriber_id);
    REQUIRE_IN_RANGE(local_id->event, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    struct recycle_buffer *group = event_register + local_id->event;
    if (recycle_buffer_remove_slot(group, local_id->local_id) != PE_SUCCESS)
        return PE_NO_ELEMENT;
    remove_global_sub(subscriber_id);
    return PE_SUCCESS;
}

enum pan_error events_post(enum event_type type, void *args) {
    if(!is_initialized)
        return PE_ILLEGAL_STATE;

    if((type >= _ENUM_EVENT_MAX) || (args == NULL))
        return PE_ILLEGAL_ARG;

    struct event data = {
        .type = type,
        .args = args
    };

    return queue_add(&posted_events, &data);
}

enum pan_error events_process() {
    if (queue_empty(&posted_events) == PE_TRUE)
        return PE_NOP;
    const struct event *data = queue_pop(&posted_events);
    const struct recycle_buffer *group = event_register + data->type;
    struct recycle_buffer_slot *it = group->in_use_list;
    while (it != NULL) {
        const struct func_ptr *callback;
        REQUIRE_NON_NULL((callback = recycle_buffer_get_data(group, it->idx)), MSG_EVENT_PROCESSING_FAILED);
        callback->func(data->type, data->args);
        it = it->next;
    }
    return PE_SUCCESS;
}