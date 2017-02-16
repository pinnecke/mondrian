#include "gtest/gtest.h"
#include <storage/host_vector_column.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Check configuration and parameter validation
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEF_COLUMN_DNULL_DINC(nullable, autoinc)                            \
    pantheon::storage::host_vector_column<unsigned>("MyColumn", 1024,       \
        pantheon::ThreadSafenessPolicy::DontUseLocks,                       \
        pantheon::UpdatePolicy::InPlaceUpdates,                             \
        pantheon::AccessPolicy::ReadAppend,                                 \
        pantheon::NullPolicy::nullable,                                     \
        pantheon::CollectionBehaviorPolicy::Bag,                            \
        pantheon::KeyPolicy::NoRestriction,                                 \
        pantheon::AutoIncrementPolicy::autoinc);

#define APPEND_RESOLVE(column, conflict_policy)                                                                     \
        column->append(nullptr, nullptr, nullptr, 0, pantheon::DeduplicationPolicy::DontCare,               \
            pantheon::LockHandling::Auto, pantheon::ReuseOfTupletIdsPolicy::ForceNewTupletIdCreation,       \
            pantheon::AutoIncAndNullConflictPolicy::conflict_policy)

TEST (HostVectorColumn, AppendAutoIncConflictChecks)
{
    auto non_nullable_non_inc_column = new DEF_COLUMN_DNULL_DINC(NonNull, NoAutoIncrement);
    auto nullable_non_inc_column     = new DEF_COLUMN_DNULL_DINC(Nullable, NoAutoIncrement);
    auto non_nullable_inc_column     = new DEF_COLUMN_DNULL_DINC(NonNull, AutoIncrement);
    auto nullable_inc_column         = new DEF_COLUMN_DNULL_DINC(Nullable, AutoIncrement);

    EXPECT_EQ (APPEND_RESOLVE(non_nullable_non_inc_column, DontCare), pantheon::ErrorType::OK);
    EXPECT_EQ (APPEND_RESOLVE(non_nullable_non_inc_column, AddNullValues), pantheon::ErrorType::UnsupportedOpertation);
    EXPECT_EQ (APPEND_RESOLVE(non_nullable_non_inc_column, IncrementValues), pantheon::ErrorType::UnsupportedOpertation);

    EXPECT_EQ (APPEND_RESOLVE(nullable_non_inc_column, DontCare), pantheon::ErrorType::OK);
    EXPECT_EQ (APPEND_RESOLVE(nullable_non_inc_column, AddNullValues), pantheon::ErrorType::OK);
    EXPECT_EQ (APPEND_RESOLVE(nullable_non_inc_column, IncrementValues), pantheon::ErrorType::UnsupportedOpertation);

    EXPECT_EQ (APPEND_RESOLVE(non_nullable_inc_column, DontCare), pantheon::ErrorType::OK);
    EXPECT_EQ (APPEND_RESOLVE(non_nullable_inc_column, AddNullValues), pantheon::ErrorType::UnsupportedOpertation);
    EXPECT_EQ (APPEND_RESOLVE(non_nullable_inc_column, IncrementValues), pantheon::ErrorType::OK);

    EXPECT_EQ (APPEND_RESOLVE(nullable_inc_column, DontCare), pantheon::ErrorType::UnresolvedConflict);
    EXPECT_EQ (APPEND_RESOLVE(nullable_inc_column, AddNullValues), pantheon::ErrorType::OK);
    EXPECT_EQ (APPEND_RESOLVE(nullable_inc_column, IncrementValues), pantheon::ErrorType::OK);

    delete non_nullable_non_inc_column;
    delete nullable_non_inc_column;
    delete non_nullable_inc_column;
    delete nullable_inc_column;
}

#define DEF_COLUMN_DKEY_TYPE(constraint)                                \
    pantheon::storage::host_vector_column<unsigned>("MyColumn", 1024,   \
        pantheon::ThreadSafenessPolicy::DontUseLocks,                   \
        pantheon::UpdatePolicy::InPlaceUpdates,                         \
        pantheon::AccessPolicy::ReadAppend,                             \
        pantheon::NullPolicy::Nullable,                                 \
        pantheon::CollectionBehaviorPolicy::Bag,                        \
        pantheon::KeyPolicy::constraint,                                \
        pantheon::AutoIncrementPolicy::AutoIncrement);

TEST (HostVectorColumn, AppendKeyConstraintHandling)
{
    // <b>Non-compound key constrain handling</b>. In case \p values is non-<code>null</code>
     ///                   and this column is constrained to contain values that are from a pre-defined range
     ///                   (e.g., a foreign-key constraint) but at least one value in \p values is not contained
     ///                   in this range, the operation will be rejected. If this column is configured to
     ///                   contain non-compound (explicit) primary key values, the operation is rejected if
     ///                   \p values contains duplicates (see below) or if at least one value in \p values is
     ///                   already contained in this column. In case \p values is <code>null</code>, the
     ///                   request will be rejected if the column is configured to contain
     ///                   non-compound foreign key or compound primary/foreign key values.
     ///                   In the same case, if the column is configured to contain non-compound primary key
     ///                   values, the column must be configured to behave like a set, auto-increment must
     ///                   be turned on. <br/><br/>
    auto no_rest_col      = new DEF_COLUMN_DKEY_TYPE(NoRestriction);
    auto expl_pri_key_col = new DEF_COLUMN_DKEY_TYPE(ExplicitPrimaryKey);
    auto expl_for_key_col = new DEF_COLUMN_DKEY_TYPE(ExplicitForeignKey);
    auto comp_pri_key_col = new DEF_COLUMN_DKEY_TYPE(CompoundPrimaryKey);
    auto comp_for_key_col = new DEF_COLUMN_DKEY_TYPE(CompoundForeignKey);

    unsigned *values = (unsigned *) calloc (10, sizeof(unsigned));
    for (unsigned i = 0; i < 10; i++)
        values[i] = i;
    expl_pri_key_col->append(nullptr, nullptr, values, 10, pantheon::DeduplicationPolicy::DontCare,
                        pantheon::LockHandling::Auto, pantheon::ReuseOfTupletIdsPolicy::ForceNewTupletIdCreation,
                        pantheon::AutoIncAndNullConflictPolicy::IncrementValues);
    for (unsigned i = 0; i < 10; i++)
        values[i] = i + 10;
    expl_pri_key_col->append(nullptr, nullptr, values, 10, pantheon::DeduplicationPolicy::DontCare,
                             pantheon::LockHandling::Auto, pantheon::ReuseOfTupletIdsPolicy::ForceNewTupletIdCreation,
                             pantheon::AutoIncAndNullConflictPolicy::IncrementValues);
    delete values;


    delete no_rest_col;
    delete expl_pri_key_col;
    delete expl_for_key_col;
    delete comp_pri_key_col;
    delete comp_for_key_col;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Host column configured to act like a std::vector
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST (HostVectorColumn, DefaultConstructorConfiguresDefaultSettings)
{
    auto column = new pantheon::storage::host_vector_column<unsigned>();

    EXPECT_EQ (column->get_column_type(), pantheon::ColumnType::HostVectorColumn);
    EXPECT_EQ (column->get_access_policy(), pantheon::AccessPolicy::ReadAppend);
    EXPECT_EQ (column->get_auto_increment_policy(), pantheon::AutoIncrementPolicy::NoAutoIncrement);
    EXPECT_EQ (column->get_behavior_policy(), pantheon::CollectionBehaviorPolicy::Bag);
    EXPECT_EQ (column->get_key_policy(), pantheon::KeyPolicy::NoRestriction);
    EXPECT_EQ (column->get_lock_policy(), pantheon::ThreadSafenessPolicy::DontUseLocks);
    EXPECT_EQ (column->get_null_policy(), pantheon::NullPolicy::NonNull);
    EXPECT_EQ (column->get_update_policy(), pantheon::UpdatePolicy::InPlaceUpdates);

    delete column;
}


/*
TEST (HostVectorColumn, CapacityAndSizeTest)
{
    auto column = new pantheon::storage::host_vector_column<unsigned>("MyColumn", 2048);
    EXPECT_EQ (column->get_capacity(), 2048);
    EXPECT_EQ (column->get_num_elements(), 0);

    unsigned *values = (unsigned *) malloc (10 * sizeof(unsigned));
    column->append(nullptr, nullptr, values, 10);
    EXPECT_EQ (column->get_capacity(), 2048);
    EXPECT_EQ (column->get_num_elements(), 10);

    free (values);
    delete column;
}*/

