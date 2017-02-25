#ifndef PANTHEON_VECTOR_COLUMN_HPP
#define PANTHEON_VECTOR_COLUMN_HPP

#include <vector>
#include <functional>

using namespace std;

#include <logger.hpp>
#include "base_column.hpp"

namespace pantheon
{
    namespace storage
    {
        template <typename ValueType, typename TupletIdType = unsigned>
        class host_vector_column : public xbase_column<ValueType, TupletIdType>
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

            virtual ErrorType on_append(tuplet_id_t *tuplet_ids,
                                        size_t *num_of_tuplet_ids,
                                        const value_t *values,
                                        size_t num_of_values,
                                        DeduplicationPolicy deduplication_policy,
                                        LockHandling lock_policy,
                                        ReuseOfTupletIdsPolicy tuplet_id_policy,
                                        AutoIncAndNullConflictPolicy conflict_policy) noexcept override
            {
                if ((Base::auto_increment_policy == AutoIncrementPolicy::AutoIncrement) &&
                        (Base::null_policy == NullPolicy::Nullable) &&
                        (conflict_policy == AutoIncAndNullConflictPolicy::DontCare)) {
                    LOG_WARNING("Column %p has unresolved conflicts w.r.t. to how values should be added. Operation "
                                        "was aborted.", this);
                    return ErrorType::UnresolvedConflict;
                }

                if (((conflict_policy == AutoIncAndNullConflictPolicy::AddNullValues) &&
                        (Base::null_policy == NullPolicy::NonNull)) ||
                    ((conflict_policy == AutoIncAndNullConflictPolicy::IncrementValues) &&
                        (Base::auto_increment_policy == AutoIncrementPolicy::NoAutoIncrement))) {
                    LOG_WARNING("Column %p was scheduled to append values with a strategy that is not supported. "
                                        "Operation was aborted.", this);
                    return ErrorType::UnsupportedOpertation;
                }


                if (values == nullptr)
                {
                    if (Base::key_policy != KeyPolicy::NoRestriction) {
                        if (Base::key_policy != KeyPolicy::ExplicitPrimaryKey) {
                            LOG_WARNING("Only for non-compound primiary key columns, the values to be added can point to"
                                                "NULL. Operation on column %p is illegal and was aborted.", this);
                            return ErrorType::IllegalNullReference;
                        } else if (Base::key_policy == KeyPolicy::ExplicitPrimaryKey &&
                                   (Base::collection_behavior_policy != CollectionBehaviorPolicy::Set ||
                                    Base::auto_increment_policy != AutoIncrementPolicy::AutoIncrement ||
                                    Base::null_policy != NullPolicy::NonNull)) {
                            LOG_WARNING("Non-compound primiary key column %p is configured illegally. "
                                                "Operation was aborted.", this);
                            return ErrorType::ExplicitPrimiaryKeyMightNotHold;
                        }
                    }
                }
                else /* (values != nullptr) */
                {
                    if ((Base::key_policy == KeyPolicy::ExplicitForeignKey ||
                         Base::key_policy == KeyPolicy::CompoundForeignKey) &&
                        not Base::foreign_value_exists_check(values, num_of_values)) {
                        LOG_WARNING("Foreign key column %p contains values that violates foreign-key constraint. "
                                            "Operation was aborted.", this);
                        return ErrorType::ForeignKeyConstraintViolated;
                    } else if (Base::key_policy == KeyPolicy::ExplicitPrimaryKey) {
#if not defined(NDEBUG)
                        LOG_INFO("Append new values for non-compound primiary key column %p, duplication check [START]", this);

                        bool duplicates_in_data = (Base::contains_unique_values == Trilean::False);
                        // TODO: Improve this to reduce O(n^2)
                        for (const value_t *lhs = data; !duplicates_in_data && lhs != data + num_elements; ++lhs)
                        {
                            for (const value_t *rhs = data; !duplicates_in_data && rhs != data + num_elements; ++rhs)
                            {
                                duplicates_in_data |= (lhs != rhs) ? false : (*lhs == *rhs);
                            }
                        }
                        if (duplicates_in_data)
                        {
                            LOG_WARNING("Values contained in non-compound primiary key column %p are not duplicate-free. Operation was aborted.", this);
                            return ErrorType::PrimaryKeyConstraintViolated;
                        }

                        // TODO: Improve this to reduce O(n^2)
                        for (const value_t *lhs = values; lhs != (values + num_of_values); ++lhs)
                        {
                            for (const value_t *rhs = values; rhs != (values + num_of_values); ++rhs)
                            {
                                if (lhs != rhs && *lhs == *rhs)
                                {
                                    LOG_WARNING("Values to be added to non-compound primiary key column %p are not duplicate-free. Operation was aborted.", this);
                                    return ErrorType::PrimaryKeyConstraintViolated;
                                }
                            }
                        }

                        LOG_INFO("Append new values for non-compound primiary key column %p, duplication check [DONE]", this);
#endif
                    }
                }

                bool values_satisfy_constraints;
                char *message = Base::value_constraints(&values_satisfy_constraints, this->data, this->num_elements,
                                                        (this->null_policy == NullPolicy::Nullable),
                                                         values, num_of_values, &this->null_mask);

                if (!values_satisfy_constraints) {
                    LOG_WARNING("Value constraints violated in column %p, details: '%s'. Operation was aborted.", this,
                                (message != nullptr ? message : "No details"));
                    if (message != nullptr)
                        free (message);
                    return ErrorType::ValueConstraintNotSatisfied;
                }

                ///                   <b>Duplication handling</b>. If this column is configured to behave like a set
                ///                   (see constructor), a deduplication on \p values is executed if \p deduplication_policy
                ///                   is set to <code>RunDeduplicationIfNeeded</code>. If this column is configured to
                ///                   behave like a bag (see constructor), <i>no</i> deduplication is executed (in this
                ///                   case the method <code>contains_only_unique_values()</code> will return
                ///                   <code>Unknown</code>). A deduplication can be forced independent of whether this
                ///                   column is a set or a bag, if \p deduplication_policy is set to
                ///                   <code>ForceDeduplication</code> (in this case the method
                ///                   <code>contains_only_unique_values()</code> will return <code>True</code> if all
                ///                   calls to this method were configured with <code>ForceDeduplication</code> or
                ///                   if this column is a set and all calls to this method were configured with
                ///                   <code>RunDeduplicationIfNeeded</code> or <code>ForceDeduplication</code>. Otherwise,
                ///                   the method <code>contains_only_unique_values()</code> will return
                ///                   <code>Unknown</code>). In case this column behaves like a set and all calls to this
                ///                   method run on <code>RunDeduplicationIfNeeded</code>, the values stored in this column
                ///                   are guaranteed to be unique (in this case the method
                ///                   <code>contains_only_unique_values()</code> will return <code>True</code>).
                ///                   If at least one call to this method was not configured with
                ///                   <code>RunDeduplicationIfNeeded</code> (or <code>ForceDeduplication</code>) for a
                ///                   set-behaving column, the values are not guaranteed to be unique (in this case the
                ///                   method <code>contains_only_unique_values()</code> will return <code>Unknown</code>).
                ///                   Since deduplication might be expensive (depending on the column implementation), the
                ///                   duplication-free guarantee required for a set-behaving column can be managed outside
                ///                   this column, i.e., the column assumes that \p values is duplication-free and no
                ///                   deduplication is executed. This behavior is activated if \p deduplication_policy is
                ///                   set to <code>DontCare</code> (in this case the method
                ///                   <code>contains_only_unique_values()</code> will return <code>Unknown</code>). In case
                ///                   this column is configured to be a non-compound (explicit) primary key, all calls
                ///                   to this method which are configured with <code>DontCare</code> are rejected.

                return ErrorType::OK;
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
            ///
            ///
            /// \param column_name
            /// \param capacity the number of elements intended to be stored
            /// \param lock_policy
            /// \param update_policy
            /// \param access_policy
            /// \param null_policy
            /// \param collection_behavior_policy
            /// \param key_policy
            /// \param auto_increment_policy
            /// \param default_value_supplier
            /// \param value_t
            host_vector_column(const char *column_name = "default",
                               size_t capacity = 1024,
                               ThreadSafenessPolicy lock_policy = ThreadSafenessPolicy::DontUseLocks,
                               UpdatePolicy update_policy = UpdatePolicy::InPlaceUpdates,
                               AccessPolicy access_policy = AccessPolicy::ReadAppend,
                               NullPolicy null_policy = NullPolicy::NonNull,
                               CollectionBehaviorPolicy collection_behavior_policy = CollectionBehaviorPolicy::Bag,
                               KeyPolicy key_policy = KeyPolicy::NoRestriction,
                               AutoIncrementPolicy auto_increment_policy = AutoIncrementPolicy::NoAutoIncrement,
                               foreign_value_exists_checks::function_t<value_t> foreign_value_exists_check = foreign_value_exists_checks::any_value_exists<value_t>(),
                               default_value_suppliers::function_t<value_t> default_value_supplier = default_value_suppliers::default_constructor<value_t>(),
                               value_constraints::function_t<value_t> value_constraints = value_constraints::no_constraints<value_t>(),
                               increment_functions::function_t<value_t> increment_function = increment_functions::numeric_increment<value_t>()) :
                      Base(column_name, lock_policy, update_policy, access_policy, null_policy, collection_behavior_policy, key_policy,
                           auto_increment_policy, foreign_value_exists_check, default_value_supplier, value_constraints, increment_function),
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
