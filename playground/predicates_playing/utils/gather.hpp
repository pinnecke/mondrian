#pragma once

namespace mondrian
{
    namespace utils
    {
#define POINTER_GATHER(dest, src, idx, num)                         \
        {                                                           \
            assert (dest != nullptr);                               \
            assert (src != nullptr);                                \
            __builtin_prefetch(dest, PREFETCH_RW_FOR_WRITE, 0);     \
            __builtin_prefetch(src, PREFETCH_RW_FOR_READ, 0);       \
            __builtin_prefetch(idx, PREFETCH_RW_FOR_READ, 0);       \
            size_t count = num;                                     \
            while (count--) {                                       \
                *(dest++) = *(src + *(idx++));                      \
            }                                                       \
        }
    }
}
