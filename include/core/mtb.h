/**
 * MTB  Macro Toolbox
 */

#ifndef GRIDSTORE_MACROS_H
#define GRIDSTORE_MACROS_H

#include <core/stdinc.h>
#include "error.h"

#define _REQUIRE_MATCH(expression, equals, compareValue, errorString)  \
{                                                                      \
    if (((expression) == compareValue) != equals) {                    \
        fprintf(stderr, errorString);                                  \
        exit (EXIT_FAILURE);                                           \
    }                                                                  \
}

#define REQUIRE_SUCCESS(expression, errorString)                   \
    _REQUIRE_MATCH(expression, true, GS_SUCCESS, errorString);

#define MTB_REQUIRE_UNEQUAL(expression, value, errorString)             \
    _REQUIRE_MATCH(expression, false, value, errorString);

#define REQUIRE_EQUAL(expression, value, errorString)               \
    _REQUIRE_MATCH(expression, true, value, errorString);

#define REQUIRE_NON_NULL(expression, errorString)                   \
    MTB_REQUIRE_UNEQUAL(expression, NULL, errorString);

#define MTB_FILL_INCREASING(Type, begin, end, initValue)                                     \
{                                                                                            \
    Type value = initValue;                                                                  \
    for(Type *first = begin, *last = end; first != last; first++) {                          \
        *first = value++;                                                                    \
    }                                                                                        \
}

#define MTB_REQUIRE_LESS(expression, value, errorString)               \
{                                                                      \
    if ((expression) >= value) {                                       \
        fprintf(stderr, errorString);                                  \
        exit (EXIT_FAILURE);                                           \
    }                                                                  \
}

#define MTB_REQUIRE_GREATER_EQ(expression, value, errorString)         \
{                                                                      \
    if ((expression) < value) {                                        \
        fprintf(stderr, errorString);                                  \
        exit (EXIT_FAILURE);                                           \
    }                                                                  \
}

#define REQUIRE_IN_RANGE(expression, lower, upper, errorString)    \
    MTB_REQUIRE_GREATER_EQ(expression, lower, errorString)             \
    MTB_REQUIRE_LESS(expression, upper, errorString)



#endif //GRIDSTORE_MACROS_H
