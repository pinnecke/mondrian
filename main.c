#include <printf.h>
#include <core/attribute.h>
#include <core/event.h>

void handle_event(enum event_type type, void *data) {
    printf("Received: %d\n", type);
}

void handle_event2(enum event_type type, void *data) {
    printf("Received: %d\n", type);
}

void handle_event3(enum event_type type, void *data) {
    printf("Received: %d\n", type);
}

int main() {

    for (int i = 0; i < 11; i++) {
        const gsAttribute_t *a1 = gsCreateAttribute("Test", GS_DATA_TYP_INT8, (i + 1) % 2 == 0 ? 1 : (i+1), GS_ATTR_FLAG_PRIMARY);
        printf("name: %s, type:%d, length: %zu, flags: %d, aid: %zu\n", a1->name, a1->dataType, a1->len, a1->flags, a1->attribute_id);
    }

    printf("Hey");

    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    events_unsubscribe(0);

    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));

    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    events_unsubscribe(0);

    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX, handle_event));

    return 0;
}