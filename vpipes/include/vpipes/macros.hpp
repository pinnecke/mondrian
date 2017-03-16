#define MIN(x,y) (x < y ? x : y)

#define PREFETCH_RW_FOR_WRITE                   1
#define PREFETCH_RW_FOR_READ                    0
#define PREFETCH_LOCALITY_REMOVE_FROM_CACHE     0
#define PREFETCH_LOCALITY_KEEP_IN_CACHES_LOW    1
#define PREFETCH_LOCALITY_KEEP_IN_CACHES_NORMAL 2
#define PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH   3

#ifdef NDEBUG
#define debug_create_variable(type, name, value) \
    type name = value;
#define debug_exec(code)                         \
    { code }

#else
#define debug_create_variable(type, name, value) \
    ;
#define debug_exec(code)                         \
    { }

#endif