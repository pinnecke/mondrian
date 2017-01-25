#include <core/config.h>

enum pan_error config_get(struct config_info *info) {
    if (info == NULL)
        return PE_ILLEGAL_ARG;

    info->shareAttributes = OPT_SHARE_ATTRIBUTES;
    info->gcAttributeHeapSize = OPT_GC_ATTRIBUTE_HEAPSIZE;
    info->init_cap_event_groups = OPT_EVENT_GROUP_INIT_LIST_CAPACITY;
    info->max_sub_id = OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER;

    return PE_SUCCESS;
}