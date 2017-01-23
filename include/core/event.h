#include <core/error.h>
#include <stdint.h>

#ifndef GRIDSTORE_EVENT_H
#define GRIDSTORE_EVENT_H

enum event_type {
    /// Raised when heap size for attribute object management was readjusted
    /// \see Event handler argument pointer <code>arg</code> is of type <code>gsAttributesHeapReallocArgs_t</code>
    GS_EVENT_ATTRIBUTES_HEAP_REALLOC,
    GS_EVENT_ATTRIBUTES_HEAP_REALLOC_TESTXXX,

    // new events here...

    /* Add new events above this comment, but do not assign enums to a certain value (see below) */

    // *IMPORTANT*: the enum constant '_ENUM_EVENT_MAX' must be the last item in 'gsEvent_t'. It is used to determine
    // the number of events defined in 'gsEvent_t' (see implementation file for usage). Consequentially, none of the
    // elements in this enum is allowed to be assigned to a certain number manually, i.e., statements like
    // 'GS_EVENT_ATTRIBUTES_HEAP_REALLOC = <value>' are forbidden here.
    _ENUM_EVENT_MAX
};

struct subscriber {
    enum event_type eventType;
    void (*callback)(enum event_type event, void *args);
};


typedef struct gsStructAttributesHeapReallocArgs {
    size_t oldSize, newSize;
} gsAttributesHeapReallocArgs_t;

/// Installs the event handler \p callback that is invoked when the event \p event was raised.
///
/// Events can be submitted (post) with the function <code>gsPostEvent(gsEvent_t, void *)</code>. Several
/// receiver can be notified about the happening of events. By calling the function
/// <code>gsSubscribeEvent(gsEvent_t, void ()(gsEvent_t event, void *args))</code>, a single interested receiver
/// can submit interest on multiple events by invoking this function multiple times on multiple event types. If an
/// event is raised, the event is added to an event queue. The event queue is read if the function
/// <code>gsProcessEvents()</code> is invoked, and interested receivers are notified (see documentation of
/// <code>gsProcessEvents</code> for more details). The function \p callback to handle the requested event is invoked
/// with the corresponding event type and an void pointer. The event type is used to distinguish between multiple
/// event since the same callback can be registered for multiple events. The void pointer points to an object
/// of a certain (pre-defined) event-depending type. This objects contains further event arguments (e.g., timestamp
/// if available). To which type \p *args must be casted depends on the chosen event type.
///
/// \see <code>gsEvent_t</code> documentation for more information on event types and casting targets for \p *args
///
/// \param type An event of type <code>gsEvent_t</code> starting with <code>GS_EVENT_</code> prefix.
/// \param callback The function which is invoked if \p event is read during <code>gsProcessEvents()</code> execution
/// \return An unique identifier that can be used to remove interest for certain events by calling
///         <code>gsRemoveEventHandler(eventHandlerId)</code>
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
///
uint64_t events_subscribe(enum event_type type, void (*callback)(enum event_type event, void *args));

/// Removes the interests for a certain event for which
/// <code>gsSubscribeEvent(gsEvent_t, void ()(gsEvent_t event, void *args))</code> was previously called.
///
/// An interested receiver can submit interest for one or more events by calling
/// <code>gsSubscribeEvent(gsEvent_t, void ()(gsEvent_t event, void *args))</code> on the according event types.
/// If an interest does no longer exists, this interest can be removed by removing the event handler. The event
/// handler is identified by an unique id which was returned by
/// <code>gsSubscribeEvent(gsEvent_t, void ()(gsEvent_t event, void *args))</code>.
///
/// \param subscriber_id the unique identifier that identifies the event handler to be removed.
/// \return <code>gsSuccess</code> if the event handler associated with \p id is known and active,
///         <code>gsIllegalState</code> if no event was ever subscribed before, and <code>gsNoSuchElement</code> if
///         \p id is unknown.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
gsError_t events_unsubscribe(uint64_t subscriber_id);

gsError_t events_post(enum event_type type, void *args);

///
/// \note This function is a co-routine.
/// \return
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
gsError_t events_process();

#endif //GRIDSTORE_EVENT_H
