
// - managed host memory with custom garbage collector to share attributes in order to save space.
// - attribute heap statically allocates certain amount of memory; heap capacity can be configured at compile time.
// - memory deallication is done over lazy garbage collector execution
// - dictionary compression of attribute string representing the attribute names
// - numeric (integer and real) data types with different sizes, variable and fixed-sized string data
// - fixed and variable length attribute data types

#ifndef GRIDSTORE_TABLES_H
#define GRIDSTORE_TABLES_H

#include <core/error.h>
#include <stddef.h>
#include <stdio.h>
#include "stddef.h"

typedef enum gsEnumDataType {
    GS_DATA_TYP_UINT8,
    GS_DATA_TYP_UINT16,
    GS_DATA_TYP_UINT32,
    GS_DATA_TYP_UINT64,
    GS_DATA_TYP_INT8,
    GS_DATA_TYP_INT16,
    GS_DATA_TYP_INT32,
    GS_DATA_TYP_INT64,
    GS_DATA_TYP_FLOAT32,
    GS_DATA_TYP_FLOAT64,
    GS_DATA_TYP_FIXEDSTRING,
    GS_DATA_TYP_VARIABLESTRING,
    GS_DATA_TYP_BINARY
} gsDataType_t;

enum gsEnumAttributeFlags {
    GS_ATTR_FLAG_UNIQUE,
    GS_ATTR_FLAG_PRIMARY,
    GS_ATTR_FLAG_NON_NULL,
    GS_ATTR_FLAG_AUTO_INCR,
    GS_ATTR_FLAG_NULLABLE
};

typedef struct gsStructAttribute
{
    const char *name;
    gsDataType_t dataType;
    size_t len;
    uint8_t flags;
    size_t attribute_id;
} gsAttribute_t;

typedef struct gsStructAttributeHeapInfo
{
    size_t numAttributesInUse;
    size_t numAttributesInFreeList;
    size_t bytesInUse;
    size_t bytesInFreeList;
    uint32_t lastGcExectTime;

} gsAttributeHeapInfo_t;

enum pan_error gsShutdownAttributesManager();

/// Constructs a new attribute object with given parameters.
///
/// The system is requested to create a new attribute object. A new attribute object is maybe constructed and a
/// read only pointer is returned, or a pointer to an already constructed attribute object is returned. The latter
/// case happens if other callers requested the same attribute before. Freeing the return attribute object
/// from caller side leaves to undefined behavior. To remove an attribute use function
/// <code>gsDisposeAttribute(...)</code>.
///
/// The function returns a (shared) pointer to an attribute with the requested properties. If memory allocation
/// fails or the function parameters are illegal, the function returns NULL. In the latter case,
/// <code>gsLastError</code> is set to the specific reason of failure.
///
/// \param columnName a valid pointer to non-empty string representing the columns name
/// \param dataType a data type for the attribute
/// \param length a number > 0 of subsequent elements of type \p data_type. For fixed-sized numeric values and
///            variable-length strings use 1. For fixed-length strings with a maximum of n characters, use n.
/// \param flags properties of this attribute
/// \see enum gsEnumAttributeFlags for flags used for parameter \p attr_flags.
/// \return A (shared) pointer to an attribute object have the requested properties, or NULL if failed.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
const gsAttribute_t *gsCreateAttribute(const char *columnName, gsDataType_t dataType, size_t length, uint8_t flags);

/// Compares two attributes \p lhs and \p rhs.
///
/// Performs an equals-comparision between the attribute \p lhs and \p rhs. The function returns <code>gsEquals</code>
/// if both attributes are equal, otherwise it returns <code>gsUnequals</code>. Two attributes are equal if and only if
/// they have the same column name, data type, data length and property flags. The comparison of column names is
/// case-sensitive.
///
/// \param lhs the first attribute
/// \param rhs the second attribute
/// \return <code>gsEquals</code> of both are equal, and <code>gsUnequals</code> otherwise.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
enum pan_error gsCompareAttributes(const gsAttribute_t *lhs, const gsAttribute_t *rhs);

/// Prints an attribute in a human-readable form to a stream.
///
/// The content of the attribute \p attr is printed to stream \p stream with the following format:<br/>
/// <code>Attribute(type=[data type], length=[length], name=<[column name], flags=[verbose form of flags])</code>
///
/// \param stream stream to which the attribute should be printed, e.g., <code>stdout</code>
/// \param attribute the attribute that should be printed. The attribute pointer must be valid.
/// \return <code>gsSuccess</code> if success, <code>gsIllegalArgument</code> if \p stream or \p attr are illegal.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
enum pan_error gsPrintAttribute(FILE *stream, const gsAttribute_t *attribute);

/// Notifies the system that the given attribute is no longer needed.
//
/// The system is requested to free up memory for the given attribute object. Since multiple callers might share the
/// same pointer to the attribute object \p attr, the actual deleting of the given attribute object might be done at
/// a later time. However, after calling this function the caller should treat its attribute object as freed and
/// should no longer use the pointer to \p attr.
///
/// \param attribute A valid pointer to an attribute object
/// \return <code>gsSuccess</code> if success, <code>gsIllegalArgument</code> if \p is non-valid attribute.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
enum pan_error gsDisposeAttribute(gsAttribute_t *attribute);

/// Receives information to the attribute heap.
///
/// \param info a pointer to an <code>gsAttributeHeapInfo_t</code> object in which the info will be stored
/// \returns  <code>gsSuccess</code>.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
enum pan_error gsAttributeHeapInfo(gsAttributeHeapInfo_t *info);

/// Requests to free memory for attributes object that are no longer in use.
///
/// If <code>gsCreateAttribute</code> is called more than one times with same parameters, subsequent callers receive
/// a shared pointer to the first created attribute object. A call to <code>gsDisposeAttribute</code> requests to
/// remove these objects but none of these request actually starts a memory deallocation. Using this function,
/// the system is notified to free allocated memory for attribute objects that are no longer needed.
///
/// \return <code>gsSuccess</code> if garbage collection removes at least one object, otherwise <code>gsNoOperation</code>.
///
/// \author Marcus Pinnecke
/// \date 2017-01-11
/// \since 1.00.00
enum pan_error gsExecAttributesGarbageCollection();

#endif
