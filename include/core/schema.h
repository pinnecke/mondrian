#include <stdbool.h>
#include "attribute.h"

#ifndef GRIDSTORE_SCHEMA_H
#define GRIDSTORE_SCHEMA_H

typedef struct gsStructSchema
{
    bool sealed;
    const gsAttribute_t *attr_v;
    size_t num_attr;
} gsSchema_t;

gsSchema_t *gsCreateSchema(const gsAttribute_t *attr_v, size_t num_attr);

gsSchema_t *gsCompareSchemes(const gsAttribute_t *lhs, const gsAttribute_t *rhs);

gsSchema_t *gsPrintSchema(const gsAttribute_t *attr);

gsError_t gsAppendAttribute(gsSchema_t *schema, const gsAttribute_t *attr_v, size_t num_attr);

gsError_t *gsSealSchema(gsSchema_t *schema);

gsError_t gsDisposeSchema(gsSchema_t *schema);

#endif //GRIDSTORE_SCHEMA_H
