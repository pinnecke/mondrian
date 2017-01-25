#include <core/error.h>
#include <core/stddef.h>

#ifndef GRIDSTORE_VECTOR_H
#define GRIDSTORE_VECTOR_H

struct field {
    struct {
        u8  is_null : 1;
    } flags;
};

// A stripe is a generalization of a vector data structure and suitable for both n-nary storage model (NSM) and
// decomposed storage model (DSM)
struct stripe {
    void *data;
    u64 capacity;
    u64 size;
};

struct stripe_

enum pan_error vector_create(struct stripe *out, u64 capacity, u64 element_size);

enum pan_error vector_free(struct stripe *vec);

const void *vector_at(const struct stripe *vec, u64 index);

const void *vector_front(const struct stripe *vec);

const void *vector_back(const struct stripe *vec);

enum pan_error vector_is_empty(const struct stripe *vec);

enum pan_error vector_get_num_of_elements(u64 *out, const struct stripe *vec);

enum pan_error vector_reserve_elements(struct stripe *vec, u64 num_elements);

enum pan_error vector_capacity(u64 *capacity, const struct stripe *vec);

enum pan_error vector_auto_fit(struct stripe *vec);

enum pan_error vector_clear(struct stripe *vec);

enum pan_error vector_push_back(struct stripe *vec, void *data);

enum pan_error vector_delete(struct stripe *vec, u64 position);

enum pan_error vector_swap(struct stripe *vec, u64 pos_lhs, u64 pos_rhs);

#endif //GRIDSTORE_VECTOR_H
