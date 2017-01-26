#include <printf.h>
#include <Pantheon/Attribute.h>
#include <Pantheon/event.h>
#include <Pantheon/Container/RecycleBuffer.h>

using namespace Pantheon;

void handle_event(event_type type, void *data) {
    printf("Received: %d\n", type);
}

void handle_event2(event_type type, void *data) {
    printf("Received: %d\n", type);
}

void handle_event3(event_type type, void *data) {
    printf("Received: %d\n", type);
}

int comp(void *lhs, void *rhs)
{
    int a = *((int *) lhs);
    int b = *((int *) rhs);
    return (a > b ? 1 : (a < b ? -1 : 0));
}

int main() {

   /* for (int i = 0; i < 11; i++) {
        const gsAttribute_t *a1 = gsCreateAttribute("Test", GS_DATA_TYP_INT8, (i + 1) % 2 == 0 ? 1 : (i+1), GS_ATTR_FLAG_PRIMARY);
        printf("name: %s, type:%d, length: %zu, flags: %d, aid: %zu\n", a1->name, a1->dataType, a1->len, a1->flags, a1->attribute_id);
    }*/

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

    printf("Next should be zero: %d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    events_unsubscribe(8);
    events_unsubscribe(6);
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));
    printf("%d\n", events_subscribe(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, handle_event));

    int dummy;
    events_post(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, &dummy);
    events_post(GS_EVENT_ATTRIBUTES_HEAP_REALLOC, &dummy);
    events_process();
    events_process();
    events_process();

    printf("--------------\n");


    return 0;
}