#include <core/event.h>
#include <core/stddef.h>
#include <core/collections/queue.h>

#define GROW_FACT 1.4f

struct node
{
    u64 subscriber_id;
    struct node *next;
};

struct func_ptr
{
    void (*func)(enum event_type, void *);
};

struct event_group
{
    struct func_ptr *callbacks;
    size_t subscribers;
    struct node *free_list, *in_use_list;
};

// The i-th entry in the following array is the i-th type defined in enum 'gsEvent_t'. For each entry, a list of
// registered type handler is stored
struct event_group event_register[_ENUM_EVENT_MAX];

struct global_sub
{
    enum event_type event;
    u64 local_id, global_id;
};

struct
{
    struct global_sub *list;
    u64 next_pos;
    u64 size;
    u64 capacity;
} global_reg;

struct event
{
    enum event_type type;
    void *args;
};

struct queue posted_events;

bool is_initialized;

static inline u64 pop_free_list(struct event_group *eventGroup)
{
    assert (eventGroup != NULL);
    assert (eventGroup->callbacks != NULL);
    assert (eventGroup->free_list != NULL);
    struct node *element = eventGroup->free_list;
    eventGroup->free_list = element->next;
    u64 result = element->subscriber_id;
    element->next = eventGroup->in_use_list;
    eventGroup->in_use_list = element;
    return result;
}

static inline bool free_list_filled(struct event_group *eventGroup)
{
    assert (eventGroup != NULL);
    assert (eventGroup->free_list != NULL || eventGroup->in_use_list != NULL);
    return (eventGroup->free_list != NULL);
}

static inline void fill_expand(struct node **begin, u64 begin_value, u64 end_value)
{
    struct node *head, *it;
    REQUIRE_NON_NULL(head = malloc(sizeof (struct node)), MSG_HOST_MALLOC)

    head->subscriber_id = end_value - 1;
    head->next = NULL;

    for (u64 handlerId = head->subscriber_id; handlerId > begin_value; handlerId--) {
        REQUIRE_NON_NULL(it = malloc(sizeof (struct node)), MSG_HOST_MALLOC)
        it->subscriber_id = handlerId - 1;
        it->next = head;
        head = it;
    }

    *begin = head;
}

static inline void init_free_list(struct event_group *group, u64 capacity)
{
    assert (group != NULL);
    assert (capacity > 0);
    fill_expand(&group->free_list, 0, capacity);
}

static inline struct node *tail(struct node *begin)
{
    while (begin->next != NULL) {
        begin = begin->next;
    }
    return begin;
}

static inline void expand_free_list(struct event_group *group)
{
    assert (group != NULL);

    u64 cap_old = group->subscribers;
    u64 cap_new = cap_old * GROW_FACT;
    struct func_ptr *callbacks = group->callbacks;
    group->subscribers = cap_new;
    REQUIRE_NON_NULL(group->callbacks = realloc(callbacks, sizeof(struct func_ptr) * cap_new),
                         MSG_HOST_REALLOC)

    fill_expand(&group->free_list, cap_old, cap_new);
}

static inline u64 next_local_subscriber_id(struct event_group *eventGroup)
{
    assert (eventGroup != NULL);
    if (!free_list_filled(eventGroup)) {
        expand_free_list(eventGroup);
    }
    return pop_free_list(eventGroup);
}

static inline void init_group(struct event_group *group, u64 capacity)
{
    assert (group != NULL);
    REQUIRE_NON_NULL(group->callbacks = malloc(sizeof (struct func_ptr) * capacity), MSG_HOST_MALLOC)
    group->subscribers = capacity;
}

static inline void init_register()
{
    if (!is_initialized) {
        is_initialized = true;

        struct config_info config;
        REQUIRE_SUCCESS(config_get(&config), MSG_CONFIG_LOAD)
        u64 capacity = config.init_cap_event_groups;

        for (u64 event_type = 0; event_type < _ENUM_EVENT_MAX; event_type++) {
            struct event_group *group = event_register + event_type;
            init_group(group, capacity);
            init_free_list(group, capacity);
        }

        global_reg.next_pos = global_reg.size = 0;
        global_reg.capacity = config.max_sub_id;
        REQUIRE_NON_NULL(global_reg.list = malloc(sizeof(struct global_sub) * config.max_sub_id), MSG_HOST_MALLOC)

        REQUIRE_EQUAL(queue_create(&posted_events, 200, GROW_FACT, sizeof(struct event)), GS_SUCCESS, MSG_EVENT_QUEUE)
    }
}

static inline void install_callback(struct event_group *group, uint64_t id, void (*func)(enum event_type, void *))
{
    assert (group != NULL);
    assert (func != NULL);
    assert (id < group->subscribers);
    group->callbacks[id].func = func;
}

int comp_by_global_id(const void *a, const void *b)
{
    u64 lhs = ((const struct global_sub *) a)->global_id;
    u64 rhs = ((const struct global_sub *) b)->global_id;
    return (lhs > rhs ? 1 : (lhs < rhs ? -1 : 0));
}

static inline u64 create_global_id(enum event_type event, uint64_t local_id)
{
    assert (event >= 0);
    assert (event < _ENUM_EVENT_MAX);

    struct global_sub *base = global_reg.list;

    if(global_reg.size == global_reg.capacity) {
        u64 new_max_id = global_reg.capacity * GROW_FACT;
        REQUIRE_NON_NULL(base = realloc(base, sizeof(struct global_sub) * new_max_id), MSG_HOST_REALLOC)
    }
    u64 global_id = global_reg.next_pos++;
    struct global_sub *localIdentifier = global_reg.list + global_reg.size++;
    localIdentifier->event = event;
    localIdentifier->local_id = local_id;
    localIdentifier->global_id = global_id;

    REQUIRE_EQUAL(mergesort(base, global_reg.size, sizeof(struct global_sub), comp_by_global_id), 0, MSG_MERGESORT)
    return global_id;
}

static inline const struct global_sub * get_local_id(u64 global_id)
{
    void *needle;
    struct global_sub key = { .global_id = global_id };

    REQUIRE_NON_NULL(needle = bsearch(&key, global_reg.list, global_reg.size, sizeof(struct global_sub),
                                      comp_by_global_id),
                     MSG_GLOBAL_SUB_UNKNOWN);

    return (struct global_sub *) needle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

u64 events_subscribe(enum event_type type, void (*callback)(enum event_type event, void *args))
{
    REQUIRE_NON_NULL(callback, MSG_NULL_POINTER)
    REQUIRE_IN_RANGE(type, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    init_register();

    struct event_group *eventGroup = event_register + type;
    u64 localId = next_local_subscriber_id(eventGroup);
    install_callback(eventGroup, localId, callback);

    return create_global_id(type, localId);
}

static inline gsError_t remove_local_sub(struct event_group *group, u64 local_id)
{
    assert (group != NULL);
    assert (group->callbacks != NULL);
    assert (group->in_use_list != NULL);

    struct node *root = group->in_use_list;
    if (root->subscriber_id == local_id) {
        group->in_use_list = group->in_use_list->next;
        root->next = group->free_list;
        group->free_list = root;
        return GS_SUCCESS;
    }
    for (struct node *prev = root, *it = root->next; it != NULL; prev = it, it = it->next) {
        if (it->subscriber_id == local_id) {
            prev->next = it->next;
            it->next = group->free_list;
            group->free_list = it;
            return GS_SUCCESS;
        }
    }

    return GS_NO_ELEMENT;
}

static inline void remove_global_sub(u64 global_id) {
    u64 cap_old = global_reg.capacity;
    u64 cap_new = global_reg.capacity - 1;

    struct global_sub *list_new = malloc(sizeof(struct global_sub) * cap_new);
    memcpy(list_new, global_reg.list, sizeof(struct global_sub) * global_id++);
    memcpy(list_new + global_id, global_reg.list + global_id, sizeof(struct global_sub) * (cap_old - global_id));
    free (global_reg.list);
    global_reg.list = list_new;
    global_reg.capacity = cap_new;
    global_reg.size--;
    global_reg.next_pos--;
}

gsError_t events_unsubscribe(u64 subscriber_id) {
    if(is_initialized != true)
        return GS_ILLEGAL_STATE;

    struct global_sub *local_id = get_local_id(subscriber_id);
    REQUIRE_IN_RANGE(local_id->event, 0, _ENUM_EVENT_MAX, MSG_ENUM_BOUNDS)

    struct event_group *group = event_register + local_id->event;
    if (remove_local_sub(group, local_id->local_id) != GS_SUCCESS)
        return GS_NO_ELEMENT;
    remove_global_sub(subscriber_id);
    return GS_SUCCESS;
}

gsError_t events_post(enum event_type type, void *args) {
    if(!is_initialized)
        return GS_ILLEGAL_STATE;

    if((type >= _ENUM_EVENT_MAX) || (args == NULL))
        return GS_ILLEGAL_ARGUMENT;

    struct event data = {
        .type = type,
        .args = args
    };
    queue_add(&posted_events, &data);
}

gsError_t events_process() {
    if (queue_empty(&posted_events) == gsTrue)
        return GS_NO_OPERATION;
    const struct event *data = queue_pop(&posted_events);
    const struct event_group *group = event_register + data->type;
    struct node *it = group->in_use_list;
    while (it != NULL) {
        group->callbacks[it->subscriber_id].func(data->type, data->args);
        it = it->next;
    }
    return GS_SUCCESS;
}