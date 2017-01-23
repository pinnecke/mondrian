#include <core/attribute.h>
#include <stdbool.h>
#include <core/config.h>
#include <stdlib.h>
#include <core/strtable.h>
#include <assert.h>
#include <string.h>
#include <core/mtb.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Structs
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct structHeapEntry {
    gsAttribute_t *attribute;
    size_t shared_counter;
} heapEntry_t;

struct structPositionListEntry {
    struct structPositionListEntry *next;
    size_t position;
} typedef positionListEntry_t;

struct structHeap {
    heapEntry_t *pool;
    size_t num_in_use;
    positionListEntry_t *inUseList, *freeList;
    bool is_initialized;
} heap;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Implementation
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const gsAttribute_t *gsCreateAttribute(const char *columnName, gsDataType_t dataType, size_t length, uint8_t flags) {
    gsError_t initAttributesManager();
    heapEntry_t *findAttributeInHeap(const gsAttribute_t *needle);
    heapEntry_t *createHeapEntry(const gsAttribute_t *entry);

    initAttributesManager();

    if (columnName == NULL || length == 0) {
        gsLastError = GS_ILLEGAL_ARGUMENT;
        return NULL;
    }
    heapEntry_t *heapEntry;
    gsAttribute_t needle = {
            .name = strdup(columnName),
            .dataType = dataType,
            .flags = flags,
            .len = length
    };

    if (((heapEntry = findAttributeInHeap(&needle)) == NULL) && ((heapEntry = createHeapEntry(&needle)) == NULL))  {
        fprintf(stderr, GS_MSG_OOM_ATTRIBUTES_HEAP);
        exit (EXIT_FAILURE);
    }

    return heapEntry->attribute;
}

gsError_t gsCompareAttributes(const gsAttribute_t *lhs, const gsAttribute_t *rhs){
    gsError_t initAttributesManager();

    if (lhs == NULL || rhs == NULL)
        return GS_ILLEGAL_ARGUMENT;
    initAttributesManager();
    return ((strcmp(lhs->name, rhs->name) == 0) && (lhs->flags == rhs->flags) && (lhs->dataType == rhs->dataType) &&
            (lhs->len == rhs->len))? gsEquals : gsUnequals; /* attribute_id does not participate for comparison */
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Helper
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

gsError_t initAttributesManager() {
    if (heap.is_initialized)
        return GS_NO_OPERATION;
    heap.is_initialized = true;
    heap.inUseList = NULL;
    heap.num_in_use = 0;

    struct config_info info;
    config_get(&info);
    heap.pool = malloc(info.gcAttributeHeapSize * sizeof(heapEntry_t));

    positionListEntry_t *cursor;
    cursor = malloc(sizeof(positionListEntry_t));
    cursor->position = 0;
    cursor->next = NULL;
    heap.freeList = cursor;
    for (size_t idx = 1; idx < info.gcAttributeHeapSize; idx++) {
        positionListEntry_t *entry = malloc(sizeof(positionListEntry_t));
        entry->position = idx;
        cursor->next = entry;
        cursor = entry;
        cursor->next = NULL;
    }

    return GS_SUCCESS;
}

gsError_t gsShutdownAttributesManager() {
    // TODO: ...
    return GS_SUCCESS;
}

heapEntry_t *findAttributeInHeapNaiveImpl(const gsAttribute_t *needle) {
    assert (needle != NULL);

    positionListEntry_t *it = heap.inUseList;
    while (it != NULL) {
        heapEntry_t *heapEntry = &heap.pool[it->position];
        assert (heapEntry->attribute != NULL);
        if (gsCompareAttributes(needle, heapEntry->attribute) == gsEquals) {
            return heapEntry;
        }
        it = it->next;
    }
    return NULL;
}

heapEntry_t *findAttributeInHeap(const gsAttribute_t *needle) {
    assert(needle != NULL);

    struct config_info config;
    REQUIRE_SUCCESS(config_get(&config), MSG_CONFIG_LOAD);

    // TODO:                         Improve this by more efficient algorithm
    return config.shareAttributes ? findAttributeInHeapNaiveImpl(needle) : NULL;
}

heapEntry_t *createHeapEntry(const gsAttribute_t *entry) {
    if (heap.freeList == NULL) {
        return NULL;
    }
    positionListEntry_t *heapEntry = heap.freeList;
    heap.freeList = heap.freeList->next;
    heapEntry->next = heap.inUseList;
    heap.inUseList = heapEntry;
    heapEntry_t *poolEntry = &heap.pool[heapEntry->position];
    size_t typeSize = sizeof(gsAttribute_t);
    poolEntry->attribute = malloc(typeSize);
    memcpy(poolEntry->attribute, entry, typeSize);
    poolEntry->attribute->name = strdup(entry->name);
    poolEntry->shared_counter = 1;
    poolEntry->attribute->attribute_id = heapEntry->position;
    return poolEntry;
}

/// Prints an attribute in a human-readable form to a stream.
///
/// The content of the attribute \p attr is printed to stream \p stream with the following format:<br/>
/// <code>Attribute(type=[data type], length=[length], name=<[column name], flags=[verbose form of flags])</code>
///
/// \param stream stream to which the attribute should be printed, e.g., <code>stdout</code>
/// \param attribute the attribute that should be printed. The attribute pointer must be valid.
/// \return <code>gsSuccess</code> if success, <code>gsIllegalArgument</code> if \p stream or \p attr are illegal. The
/// function returns <code>gsIllegalState</code> if called before first call of <code>gsInitAttributesManager</code>.
///
/// \note <code>gsInitAttributesManager</code>() must be called before first call to this function.
gsError_t gsPrintAttribute(FILE *stream, const gsAttribute_t *attribute){
    return GS_SUCCESS;
}

/// Notifies the system that the given attribute is no longer needed.
//
/// The system is requested to free up memory for the given attribute object. Since multiple callers might share the
/// same pointer to the attribute object \p attr, the actual deleting of the given attribute object might be done at
/// a later time. However, after calling this function the caller should treat its attribute object as freed and
/// should no longer use the pointer to \p attr.
///
/// \param attribute A valid pointer to an attribute object
/// \return <code>gsSuccess</code> if success, <code>gsIllegalArgument</code> if \p is non-valid attribute. The
/// function returns <code>gsIllegalState</code> if called before first call of <code>gsInitAttributesManager</code>.
///
/// \note <code>gsInitAttributesManager</code>() must be called before first call to this function.
gsError_t gsDisposeAttribute(gsAttribute_t *attribute){
    return GS_SUCCESS;
}

/// Receives information to the attribute heap.
///
/// \param info a pointer to an <code>gsAttributeHeapInfo_t</code> object in which the info will be stored
/// \returns  <code>gsIllegalState</code> if called before first call of <code>gsInitAttributesManager</code>,
///           <code>gsSuccess</code> otherwise.
///
/// \note <code>gsInitAttributesManager</code>() must be called before first call to this function.
gsError_t gsAttributeHeapInfo(gsAttributeHeapInfo_t *info){
    return GS_SUCCESS;
}

/// Requests to free memory for attributes object that are no longer in use.
///
/// If <code>gsCreateAttribute</code> is called more than one times with same parameters, subsequent callers receive
/// a shared pointer to the first created attribute object. A call to <code>gsDisposeAttribute</code> requests to
/// remove these objects but none of these request actually starts a memory deallocation. Using this function,
/// the system is notified to free allocated memory for attribute objects that are no longer needed.
///
/// \return <code>gsSuccess</code> if garbage collection removes at least one object, otherwise <code>gsNoOperation</code>.
/// The function returns <code>gsIllegalState</code> if called before first call of <code>gsInitAttributesManager</code>.
///
/// \note <code>gsInitAttributesManager</code>() must be called before first call to this function.
gsError_t gsExecAttributesGarbageCollection(){
    return GS_SUCCESS;
}
