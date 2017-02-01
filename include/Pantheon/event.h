#include <Pantheon/error.h>
#include <stdint.h>

#ifndef GRIDSTORE_EVENT_H
#define GRIDSTORE_EVENT_H

namespace Pantheon {

    enum event_type {
        /// Raised when heap size for attribute object management was readjusted
        /// \see Event handler argument pointer <code>arg</code> is of type <code>gsAttributesHeapReallocArgs_t</code>
                GS_EVENT_ATTRIBUTES_HEAP_REALLOC,

        // new events here...

        /* Add new events above this comment, but do not assign enums to a certain value (see below) */

        // *IMPORTANT*: the enum constant '_ENUM_EVENT_MAX' must be the last item in 'gsEvent_t'. It is used to determine
        // the number of events defined in 'gsEvent_t' (see implementation file for usage). Consequentially, none of the
        // elements in this enum is allowed to be assigned to a certain number manually, i.e., statements like
        // 'GS_EVENT_ATTRIBUTES_HEAP_REALLOC = <value>' are forbidden here.
                _ENUM_EVENT_MAX
    };

    struct subscriber {
        enum event_type type;

        void (*callback)(enum event_type event, void *args);
    };

    typedef struct gsStructAttributesHeapReallocArgs {
        size_t oldSize, newSize;
    } gsAttributesHeapReallocArgs_t;

    uint64_t events_subscribe(enum event_type type, void (*callback)(enum event_type event, void *args));

    PRESULT events_unsubscribe(uint64_t subscriber_id);

    PRESULT events_post(enum event_type type, void *args);

    PRESULT events_process();

}

#endif //GRIDSTORE_EVENT_H
