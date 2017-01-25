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

/* The initial capacity (in number of objects) of the per-type group list of installed type handlers.
 * Used in: core/type.c */
#ifndef OPT_EVENT_GROUP_INIT_LIST_CAPACITY
#define OPT_EVENT_GROUP_INIT_LIST_CAPACITY     10
#endif

/* The maximum number of handlers to be registered during runtime */
/* Used in: core/type.c */
#ifndef OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER
#define OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER     1024
#endif

struct config_info {
    bool shareAttributes;
    size_t gcAttributeHeapSize;
    size_t init_cap_event_groups;
    size_t max_sub_id;
};

enum pan_error config_get(struct config_info *info);

#endif //GRIDSTORE_CONFIG_H
