#ifndef PANTHEON_VECTOR_COLUMN_HPP
#define PANTHEON_VECTOR_COLUMN_HPP

#include <vector>
#include <functional>

using namespace std;

#include <storage/base_column.hpp>

namespace pantheon
{
    namespace storage
    {
        template <typename ValueType, typename TupletIdType = unsigned>
        class host_vector_column : public base_column<ValueType, TupletIdType>
        {
            using Base = base_column<ValueType, TupletIdType>;
        public:
            using typename Base::value_t;
            using typename Base::tuplet_id_t;

        protected:

            vector<bool> null_mask;
            value_t *data;

            size_t capacity;
            size_t num_elements;

        protected:

            virtual ErrorType on_append(tuplet_id_t *tuplet_ids, const value_t *values, size_t num_of_values,
                                        LockHandling lock_policy = LockHandling::Lock,
                                        ReuseOfTupletIdsPolicy tuplet_id_policy = ReuseOfTupletIdsPolicy::RecycleTupletIdsIfPossible,
                                        AutoIncAndNullConflictPolicy conflict_policy = AutoIncAndNullConflictPolicy::DontCare) noexcept override
            {

            }

            virtual ErrorType on_update(const tuplet_id_t *tuplets, size_t num_of_ids, const value_t *new_value) noexcept override
            {

            }

            virtual ErrorType on_remove(const tuplet_id_t *tuplets, size_t num_of_ids, RemoveExecutionPolicy policy = RemoveExecutionPolicy::ImmediatlyRemove) noexcept override
            {

            }

            virtual bool is_empty() noexcept override
            {

            }

            virtual size_t get_num_of_tuplets() noexcept override
            {

            }

            virtual void on_clear() noexcept override
            {

            }

            virtual void on_dispose() noexcept override
            {

            }

        public:
            host_vector_column(const char *column_name = "default",
                               size_t capacity = 1024,
                               ThreadSafenessPolicy lock_policy = ThreadSafenessPolicy::UseLocks,
                               UpdatePolicy update_policy = UpdatePolicy::MultiVersionFields,
                               AccessPolicy access_policy = AccessPolicy::ReadAppend,
                               NullPolicy null_policy = NullPolicy::Nullable,
                               CollectionBehaviorPolicy collection_behavior_policy = CollectionBehaviorPolicy::Bag,
                               KeyPolicy key_policy = KeyPolicy::NoRestriction,
                               AutoIncrementPolicy auto_increment_policy = AutoIncrementPolicy::NoAutoIncrement,
                               function<value_t()> default_value_supplier = column_function_factory<value_t>::default_constructor(),
                               function<bool(const value_t *value)> value_constraints = column_function_factory<value_t>::no_constraints(),
                               function<ValueType(ValueType *value)> increment_function = column_function_factory<value_t>::numeric_increment()) :
                      Base(column_name, lock_policy, update_policy, access_policy, null_policy, collection_behavior_policy, key_policy,
                           auto_increment_policy, default_value_supplier, value_constraints, increment_function),
                      capacity(capacity), num_elements(0)
            {
                assert (capacity > 0);
                data = (value_t *) malloc (capacity * sizeof(value_t));
                assert (data != nullptr);
            }

            size_t get_capacity()
            {
                return capacity;
            }

            size_t get_num_elements()
            {
                return num_elements;
            }

            virtual ColumnType get_column_type() const noexcept override
            {
                return ColumnType::HostVectorColumn;
            }

            virtual ErrorType refresh_duplication_state(LockHandling lock_policy = LockHandling::Auto) noexcept override
            {

            }


        };
    }
}

#endif //PANTHEON_VECTOR_COLUMN_HPP
