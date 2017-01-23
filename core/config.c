#include <core/config.h>

gsError_t config_get(struct config_info *info) {
    if (info == NULL)
        return GS_ILLEGAL_ARGUMENT;

    info->shareAttributes = OPT_SHARE_ATTRIBUTES;
    info->gcAttributeHeapSize = OPT_GC_ATTRIBUTE_HEAPSIZE;
    info->init_cap_event_groups = OPT_EVENT_GROUP_INIT_LIST_CAPACITY;
    info->max_sub_id = OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER;

    return GS_SUCCESS;
}