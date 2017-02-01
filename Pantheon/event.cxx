#include <Pantheon/event.h>
#include <Pantheon/stddef.h>
#include <Pantheon/Container/Queue.h>
#include <Pantheon/Container/RecycleBuffer.h>

#define GROW_FACT 1.4f

using namespace Pantheon;

struct func_ptr
{
    void (*func)(enum event_type, void *);

    func_ptr(void (*f)(enum event_type, void *) = nullptr): func(f) { }
};

struct event
{
    enum event_type type;
    void *args;
};

struct global_sub
{
    enum event_type event;
    QWORD local_id, global_id;
};

// The i-th entry in the following array is the i-th type defined in enum 'gsEvent_t'. For each entry, a list of
// registered type handler is stored
Container::RecycleBuffer event_register[_ENUM_EVENT_MAX];
Container::RecycleBuffer global_register;

Container::Queue posted_events;
bool is_initialized;

static inline void init_register()
{
    using Container::RecycleBuffer;
    using Container::Queue;

    if (!is_initialized) {
        is_initialized = true;

        QWORD capacity = Config::StaticConfig::InitialEventGroupCapacity;

        for (QWORD event_type = 0; event_type < _ENUM_EVENT_MAX; event_type++) {
            RecycleBuffer *buffer = event_register + event_type;
            REQUIRE_SUCCESS(RecycleBuffer::Create(buffer, capacity, sizeof(struct func_ptr)),
                            MSG_EVENT_SYSTEM_INIT_FAILED)
        }

        REQUIRE_SUCCESS(RecycleBuffer::Create(&global_register, Config::StaticConfig::InitialSubscriberCapacity,
                                              sizeof(struct global_sub)),
                        MSG_EVENT_SYSTEM_INIT_FAILED);

        REQUIRE_EQUAL(Queue::Create(&posted_events, 200, GROW_FACT, sizeof(struct event)), PRESULT::OK, MSG_EVENT_QUEUE)
    }
}

static inline void install_callback(Container::RecycleBuffer *group, uint64_t id, void (*func)(enum event_type, void *))
{
    assert (group != NULL);
    assert (func != NULL);
    assert (id < group->Capacity);
    func_ptr data { func };
    Container::RecycleBuffer::PutData(group, id, (const BYTE *) &data);
}

static inline enum PRESULT create_global_id(QWORD *id, enum event_type event, uint64_t local_id)
{
    assert (id != NULL);
    assert (event >= 0);
    assert (event < _ENUM_EVENT_MAX);

    QWORD global_id;
    enum PRESULT result;

    if ((result = Container::RecycleBuffer::GetSlot(&global_id, &global_register)) != PRESULT::OK) {
        return result;
    }

    struct global_sub data = {
        .event = event,
        .global_id = global_id,
        .local_id = local_id
    };

    if ((result = Container::RecycleBuffer::PutData(&global_register, global_id, (const BYTE *) &data)) != PRESULT::OK) {
        return result;
    }

    *id = global_id;

    return PRESULT::OK;
}

static inline struct global_sub * get_local_id(QWORD global_id)
{
    BYTE *result;
    REQUIRE_SUCCESS(Container::RecycleBuffer::GetData(&result, &global_register, global_id), MSG_NO_ELEMENT);
    return (global_sub *) result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

QWORD Pantheon::events_subscribe(enum event_type type, void (*callback)(enum event_type, void *))
{
    REQUIRE_NON_NULL(callback, MSG_NULL_POINTER)
    REQUIRE_IN_RANGE(type, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    init_register();

    Container::RecycleBuffer *eventGroup = event_register + type;
    QWORD localId;
    REQUIRE_SUCCESS(Container::RecycleBuffer::GetSlot(&localId, eventGroup), GS_MSG_EVENT_HANDLER_REGISTRATION_FAILED);
    install_callback(eventGroup, localId, callback);

    QWORD global_id;
    REQUIRE_SUCCESS(create_global_id(&global_id, type, localId), GS_MSG_EVENT_HANDLER_REGISTRATION_FAILED);
    return global_id;
}

static inline void remove_global_sub(QWORD global_id) {
    REQUIRE_SUCCESS(Container::RecycleBuffer::RemoveSlot(&global_register, global_id),
                    MSG_REMOVE_EVENT_HANLDER_FAILED);
}

enum PRESULT Pantheon::events_unsubscribe(QWORD subscriber_id) {
    if(is_initialized != true)
        return PRESULT::IllegalState;

    struct global_sub *local_id = get_local_id(subscriber_id);
    REQUIRE_IN_RANGE(local_id->event, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    Container::RecycleBuffer *group = event_register + local_id->event;
    if (Container::RecycleBuffer::RemoveSlot(group, local_id->local_id) != PRESULT::OK)
        return PRESULT::NoSuchElement;
    remove_global_sub(subscriber_id);
    return PRESULT::OK;
}

enum PRESULT Pantheon::events_post(enum event_type type, void *args) {
    if(!is_initialized)
        return PRESULT::IllegalState;

    if((type >= _ENUM_EVENT_MAX) || (args == NULL))
        return PRESULT::IllegalArgument;

    struct event data = {
        .type = type,
        .args = args
    };

    return Container::Queue::Enqueue(&posted_events, (BYTE *) &data);
}

enum PRESULT Pantheon::events_process() {
    if (Container::Queue::IsEmpty(&posted_events) == PRESULT::True)
        return PRESULT::NoOperation;
    const struct event *data = (const event *) Container::Queue::Deqeue(&posted_events);
    const Container::RecycleBuffer *group = event_register + data->type;
    Container::RecycleBuffer::Slot *it = group->InUseList;
    while (it != NULL) {
        func_ptr *callback;
        REQUIRE_SUCCESS(Container::RecycleBuffer::GetData((BYTE **) &callback, group, it->Index),
                        MSG_EVENT_PROCESSING_FAILED);
        callback->func(data->type, data->args);
        it = it->next;
    }
    return PRESULT::OK;
}