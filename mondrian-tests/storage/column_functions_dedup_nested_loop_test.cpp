#include "gtest/gtest.h"

#include <storage/host_vector_column.hpp>
#include <policies.hpp>

TEST (ColumnFunctionsDedupNestedLoopTest, ContainedNonNullableNewNonNullable)
{
    using namespace pantheon;
    using namespace pantheon::storage;

    auto dedup_fn = dedup_functions::sort_based<unsigned>();

    // Contained is empty, new is non-empty - new doesn't contain duplicates
    unsigned *con_val1;
    unsigned new_val1[] = { 0,5,6,0,9,2,3,1,7,8 };

    size_t num_elements_out;
    bool free_required;
    unsigned *deduped_val1 = dedup_fn(&num_elements_out, nullptr, &free_required, ValueOrderPolicy::Stable, con_val1, 0, Trilean::True,
                                      false, nullptr, new_val1, 10, false, nullptr, pantheon::functional::comparators::less<unsigned>(), NullDuplicateHandlingPolicy::TreatAsRegularValue);

    EXPECT_EQ(free_required, true);
    EXPECT_NE(deduped_val1, nullptr);
    EXPECT_EQ(num_elements_out, 10);
    for (int i = 0; i < 10; i++)
        EXPECT_EQ(deduped_val1[i], new_val1[i]);

    // Contained is empty, new is non-empty - new contains duplicates

    // Contained is empty, new is empty

    // Contained is non-empty, new is non-empty - contained contains duplicates, new doesn't contain duplicates

    // Contained is non-empty, new is non-empty - contained contains duplicates, new contains duplicates

    // Contained is non-empty, new is non-empty - contained doesn't duplicates, new doesn't contain duplicates

    // Contained is non-empty, new is non-empty - contained doesn't duplicates, new contains duplicates

    // Contained is non-empty, new is empty - contained contains duplicates

    // Contained is non-empty, new is empty - contained doesn't duplicates


}

TEST (ColumnFunctionsDedupNestedLoopTest, ContainedNonNullableNewNullable)
{

}

TEST (ColumnFunctionsDedupNestedLoopTest, ContainedNullableNewNonNullable)
{

}

TEST (ColumnFunctionsDedupNestedLoopTest, ContainedNullableNewNullable)
{

}