#include <Pantheon/config.h>

namespace Pantheon
{
namespace Config
{
    const bool StaticConfig::ShareAttributes = OPT_SHARE_ATTRIBUTES;
    const size_t StaticConfig::GcAttributeHeapSize = OPT_GC_ATTRIBUTE_HEAPSIZE;
    const size_t StaticConfig::InitialEventGroupCapacity = OPT_EVENT_GROUP_INIT_LIST_CAPACITY;
    const size_t StaticConfig::InitialSubscriberCapacity = OPT_EVENT_MAXIMUM_GLOBAL_IDENTIFIER;
    const float StaticConfig::RecycleBufferGrowFactor = OPT_RECYCLE_BUFFER_GROW_FACTOR;
}
}

