#ifndef GRIDSTORE_CONFIG_H
#define GRIDSTORE_CONFIG_H

#include <stdbool.h>
#include "error.h"

/* Enables or disables the sharing of attributes */
#ifndef OPT_SHARE_ATTRIBUTES
#define OPT_SHARE_ATTRIBUTES true
#endif

/* The number of attributes that can be managed in the heap */
#ifndef OPT_GC_ATTRIBUTE_HEAPSIZE
#define OPT_GC_ATTRIBUTE_HEAPSIZE     10
#endif

/* The initial Capacity (in number of objects) of the per-type group list of installed type handlers.
 * Used in: Pantheon/type.c */
#ifndef OPT_EVENT_GROUP_INIT_LIST_CAPACITY
#define OPT_EVENT_GROUP_INIT_LIST_CAPACITY     10
#endif

/* The maximum number of handlers to be registered during runtime */
/* Used in: Pantheon/type.c */
#ifndef OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER
#define OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER     1024
#endif

#ifndef OPT_RECYCLE_BUFFER_GROW_FACTOR
#define OPT_RECYCLE_BUFFER_GROW_FACTOR     1.4f
#endif

namespace Pantheon
{
namespace Config
{
    struct StaticConfig
    {
        static const bool ShareAttributes;
        static const size_t GcAttributeHeapSize;
        static const size_t InitialEventGroupCapacity;
        static const size_t InitialSubscriberCapacity;
        static const float RecycleBufferGrowFactor;
    };
}
}

#endif //GRIDSTORE_CONFIG_H
