#ifndef PANTHEON_BASE_COLUMN_HPP
#define PANTHEON_BASE_COLUMN_HPP

#include <cassert>
#include <stdlib.h>

#include <policies.hpp>
#include <error.hpp>
#include <utils/strings.hpp>

using namespace std;

namespace pantheon
{
    namespace storage
    {
        template <typename ValueType>
        struct column_function_factory
        {
            static constexpr function<char *(bool *all_satisfy, const ValueType *values_contained, size_t num_of_values, bool values_might_be_null,
                                           const ValueType *values_to_be_added, size_t num_of_values_to_be_added,
                                           const vector<bool> *values_contained_null_mask)> no_constraints()
            {
                return ([] (bool *all_satisfy, const ValueType *, size_t, bool, const ValueType *, size_t num_of_values,
                            const vector<bool> *) -> char *
                {
                    if (num_of_values == 0) {
                        assert (all_satisfy != nullptr);
                        *all_satisfy = true;
                        return nullptr;
                    } else {
                        *all_satisfy = false;
                        char *m = (char *) malloc(100 * sizeof(char));
                        strcpy(m, "Too many ;D");
                        return m;
                    }
                });
            }

            static constexpr function<ValueType()> default_constructor() {
                return ([] () { return ValueType(); });
            }

            static constexpr function<ValueType(ValueType *value)> numeric_increment() {
                return ([] (ValueType *value) { return (*value)++; });
            }

            static constexpr function<bool(const ValueType *value, size_t num_of_values)> any_value_exists() {
                return ([] (const ValueType *, size_t) { return true; });
            }
        };

        template <typename ValueType, typename TupletIdType = unsigned>
        class base_column
        {
            using Self = base_column<ValueType, TupletIdType>;

        public:
            using value_t = ValueType;
            using tuplet_id_t = TupletIdType;

        protected:
            enum class InternalState
            {
                New, InUse
            };

            mutex column_mutex;
            char *column_name;

            NullPolicy null_policy;
            KeyPolicy key_policy;
            CollectionBehaviorPolicy collection_behavior_policy;
            AutoIncrementPolicy auto_increment_policy;
            AccessPolicy access_policy;
            UpdatePolicy update_policy;
            ThreadSafenessPolicy lock_policy;

            bool locked_flag;
            Trilean contains_unique_values;
            InternalState internal_state;

            function<char *(bool *all_satisfy, const value_t *values_contained, size_t num_of_values, bool values_might_be_null,
                          const value_t *values_to_be_added, size_t num_of_values_to_be_added,
                          const vector<bool> *values_contained_null_mask)> value_constraints;
            function<value_t()> default_value_supplier;
            function<ValueType(ValueType *value)> increment_function;
            function<bool(const value_t *value, size_t num_of_values)> foreign_value_exists_check;

            void lock()
            {
                column_mutex.lock();
                locked_flag = true;
            }

            void unlock()
            {
                column_mutex.unlock();
                locked_flag = false;
            }

            inline void reinitialize()
            {
                internal_state = Self::InternalState::New;
                contains_unique_values = Trilean::Unknown;
            }

        protected:

            virtual bool is_empty() noexcept = 0;
            virtual size_t get_num_of_tuplets() noexcept = 0;

            virtual ErrorType on_append(tuplet_id_t *tuplet_ids, size_t *num_of_tuplet_ids,
                                        const value_t *values,
                                        size_t num_of_values,
                                        DeduplicationPolicy deduplication_policy,
                                        LockHandling lock_policy,
                                        ReuseOfTupletIdsPolicy tuplet_id_policy,
                                        AutoIncAndNullConflictPolicy conflict_policy) noexcept = 0;

            virtual ErrorType on_update(const tuplet_id_t *tuplets, size_t num_of_ids, const value_t *new_value) noexcept = 0;

            virtual ErrorType on_remove(const tuplet_id_t *tuplets, size_t num_of_ids, RemoveExecutionPolicy policy = RemoveExecutionPolicy::ImmediatlyRemove) noexcept = 0;

            virtual void on_clear() noexcept = 0;

            virtual void on_dispose() noexcept = 0;

        public:
            base_column(const char *column_name,
                        ThreadSafenessPolicy lock_policy = ThreadSafenessPolicy::DontUseLocks,
                        UpdatePolicy update_policy = UpdatePolicy::InPlaceUpdates,
                        AccessPolicy access_policy = AccessPolicy::ReadAppend,
                        NullPolicy null_policy = NullPolicy::NonNull,
                        CollectionBehaviorPolicy collection_behavior_policy = CollectionBehaviorPolicy::Bag,
                        KeyPolicy key_policy = KeyPolicy::NoRestriction,
                        AutoIncrementPolicy auto_increment_policy = AutoIncrementPolicy::NoAutoIncrement,
                        function<bool(const value_t *value, size_t num_of_values)> foreign_value_exists_check = column_function_factory<value_t>::any_value_exists(),
                        function<value_t()> default_value_supplier = column_function_factory<value_t>::default_constructor(),
                        function<char *(bool *all_satisfy, const value_t *values_contained, size_t num_of_values, bool values_might_be_null,
                                      const value_t *values_to_be_added, size_t num_of_values_to_be_added,
                                      const vector<bool> *values_contained_null_mask)> value_constraints = column_function_factory<value_t>::no_constraints(),
                        function<ValueType(ValueType *value)> increment_function = column_function_factory<value_t>::numeric_increment()):
                    column_name(nullptr),
                    lock_policy(lock_policy),
                    update_policy(update_policy),
                    access_policy(access_policy),
                    null_policy(null_policy),
                    collection_behavior_policy(collection_behavior_policy),
                    key_policy(key_policy),
                    auto_increment_policy(auto_increment_policy),
                    default_value_supplier(default_value_supplier),
                    value_constraints(value_constraints),
                    increment_function(increment_function),
                    foreign_value_exists_check(foreign_value_exists_check),
                    locked_flag(false)
            {
                reinitialize();

                assert (column_name != nullptr);
                this->column_name = utils::strings::trim_inplace(strdup(column_name));
                assert (strlen(this->column_name) > 0);
            }

            virtual ColumnType get_column_type() const noexcept = 0;

            /// \brief Refreshs the statement on whether this column contains distinct values or not, when the method
            /// <code>contains_only_unique_values()</code> is called.
            ///
            /// A column might contain distinct values or not. If a column contains only unique values depends
            /// on the history of calls to the function <code>append()</code> and the deduplication policy that
            /// was used. However, to save compute effort, in some cases it is not known whether the column
            /// actually contains duplicates or not. Thus, the method <code>contains_only_unique_values()</code> might
            /// return <code>Unknown</code>. By invoking the method <code>refresh_duplication_state()</code>,
            /// <code>contains_only_unique_values()</code> returns either <code>True</code> or </code>False</code>
            /// depending on the real state of the values stored in this column.
            ///
            /// \param lock_policy [in] The policy that controls whether this operation should be protected for
            ///                    concurrent modifications. If this column is configured to use locking (see
            ///                    constructor), the locking can be turned off for this specific call when
            ///                    <code>DontLock</code> is used. In the same case, if <code>Lock</code>
            ///                    or <code>Auto</code> is used, the column will be locked before modification and
            ///                    unlocked afterwards. In case this column is configured to never lock at all
            ///                    and <code>Lock</code> is given, the request will be rejected. If the policy is set
            ///                    to <code>Auto</code>, the locking behavior is determined by
            ///                    <code>ThreadSafenessPolicy</code> as configured in the constructor of this column.
            /// \return This method returns
            ///             <ul>
            ///                 <li><code>True</code> if this column is duplication-free.</li>
            ///                 <li><code>False</code> if this column contains at least one duplicate value.</li>
            ///             </ul>
            ///          If this method returns either <code>True</code> or </code>False</code>, it is guaranteed
            ///          that a call to <code>contains_only_unique_values()</code> returns the same result (until
            ///          a next modification on the column content is executed).<br/><br/>
            ///          The method can also return
            ///             <ul>
            ///                   <li><code>LockNotSupported</code> if \p lock_policy forces to use locking but locking
            ///                                                     is disabled for this column.</li>
            ///             </ul>
            ///          In this case <code>LockNotSupported</code> is returned, the return value of
            ///          <code>contains_only_unique_values()</code> is unchanged.<br/><br/>
            ///         This method might return (depending on the implementation)
            ///             <ul>
            ///                 <li><code>NotImplemented</code> if the implementation does not implement this
            ///                                                 method.</li>
            ///                 <li><code>InternalError</code> if an internal error occur (e.g., missing switch
            ///                                                case).</li>
            ///             </ul>
            ///
            /// \note This method will block the calling thread.
            /// \author Marcus Pinnecke
            /// \date 2017-02-15
            /// \since 1.0.00
            virtual ErrorType refresh_duplication_state(LockHandling lock_policy = LockHandling::Auto) noexcept = 0;

            ///
            /// \param tuplet_ids [out] A nullable pointer to an array for tuplet ids that map to the values newly added
            ///                   to this column. Depending on the parameter \p tuplet_id_policy, tuplet ids to be used
            ///                   for this purpose might be recycled if there are tuplet ids reserved but currently not
            ///                   in use. If the parameter \p tuplet_id_policy determines recycling of tuplet ids,
            ///                   the request for this method might be rejected depending on whether the column
            ///                   implementation supports recycling of tuplet ids.<br/><br/>
            ///                   If this column is configured to hold only unique values (see constructor) and if at
            ///                   least one element in the parameter \p values is already stored in this column,
            ///                   \p tuplet_ids will contain tuplet ids that are already in-use (i.e., in this case the
            ///                   tuplet ids for these values will be the same as the tuplet ids for the original values
            ///                   added in a previous call).<br/><br/>
            ///                   If \p tuplet_ids is non-<code>null</code>, it must be capable to hold at least
            ///                   \p num_of_values elements. Note here, that the number of elements in \p tuplet_ids
            ///                   is less than or equal to \p num_of_values if deduplication was executed. However,
            ///                   \p tuplet_ids is assumed to contain at least the supremum of returnable tuplet ids.
            ///                   <br/><br/>
            ///                   If \p tuplet_ids is <code>null</code>, \p tuplet_ids will be ignored.
            /// \param num_of_tuplet_ids [out] A nullable pointer to an numeric that sets the number of elements
            ///                          for \p tuplet_ids to the caller. The parameter \p num_of_tuplet_ids is less or
            ///                          equal to \p num_of_values. Without deduplication, \p num_of_tuplet_ids and
            ///                          \p num_of_values are the same value. In case of \p values contains duplicated
            ///                          values that are removed during deduplication, \p num_of_tuplet_ids is less
            ///                          than \p num_of_values. <br/><br/>
            ///                          If \p num_of_tuplet_ids is <code>null</code>, \p num_of_tuplet_ids will be
            ///                          ignored.
            /// \param values     [in] A nullable pointer to an array of values to be added to this column. <br/><br/>
            ///                   <b>General usage</b>.<br/><br/>
            ///                   If \p values is non-<code>null</code>, it must point to at least
            ///                   \p num_of_values values. The values this parameter point to will be copied into the
            ///                   memory managed by this column and will be freed by the column itself. <br/><br/>
            ///                   If \p values is <code>null</code>, then the behavior depends on
            ///                   the null-value policy, the auto-increment policy, and the default-value-supplier
            ///                   function. In case the column is configured to contain null values but auto-increment
            ///                   is disabled, then the default-value-supplier function will be ignored and nulls will
            ///                   be added. Similar, in case the column is configured to use auto-increment and is not
            ///                   allowed to contain null values, the default value supplier function will be ignored
            ///                   and incremented values will be added. In case neither null values are allowed nor
            ///                   auto-increment is enabled, the default value supplier function will be considered
            ///                   to add new values. If this column is configured to have both capabilities containing
            ///                   null values and to use auto-increment values, this is a valid but conflicting
            ///                   configuration. It is valid since a tuplet might be updated to null, but
            ///                   newly added values should be auto-incremented. It is conflicting since it is not
            ///                   decidable whether to add null values or to auto-increment values. The conflict
            ///                   resolving is managed by the parameter \p conflict_policy. If \p conflict_policy
            ///                   does not resolve this conflict (i.e., it is set to <code>DontCare</code>), the request
            ///                   for adding values will be rejected. Also, if \p conflict_policy requests a
            ///                   strategy that is not supported by the column (e.g., add null but the column
            ///                   is non-nullable), the request will be rejected.<br/><br/>
            ///                   <b>Non-compound key constraint handling</b>. In case \p values is non-<code>null</code>
            ///                   and this column is constrainted to contain values that are from a pre-defined range
            ///                   (i.e., a foreign-key constraint) but at least one value in \p values is not contained
            ///                   in this range, the operation will be rejected. If this column is configured to
            ///                   contain non-compound (explicit) primary key values, the operation is rejected if
            ///                   \p values contains duplicates, or if at least one value in \p values is
            ///                   already contained in this column. Since this check is expensive, it is
            ///                   statically turned off when <code>NDEBUG</code> is defined. In case \p values is
            ///                   <code>null</code>, the request will be rejected if the column is configured to contain
            ///                   non-compound foreign key. In the same case, if the column is configured to contain
            ///                   non-compound primary key values, the column must be configured to behave like a set,
            ///                   auto-increment must be turned on, and null-values must be disabled. <br/><br/>
            ///                   <b>Compound key constraint handling</b>. In case this column is configured to be a
            ///                   compound key, the constraint check is not done for this operation and must be done at
            ///                   a higher-level. However, in case \p values is
            ///                   <code>null</code>, the request will be rejected. <br/><br/>
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
            ///
            /// \param num_of_values [in] The number of values to be added to this column. If the sum of
            ///                   \p num_of_values and the number of values currently stored in this column exceeds
            ///                   the maximum number of tuplet ids for this column, the request will be rejected. If
            ///                   \p tuplet_id_policy forces the column to recycle tuplet ids, the request will be
            ///                   rejected if the sum of \p num_of_values and the number of currently values incl.
            ///                   the free list of tuplet ids exceed the maximum number of tuplet ids. In case
            ///                   this column is not able to hold the desired number of elements, a wider
            ///                   tuplet_id_t type might be considered.
            /// \param lock_policy [in] The policy that controls whether this operation should be protected for
            ///                    concurrent modifications. If this column is configured to use locking (see
            ///                    constructor), the locking can be turned off for this specific call when
            ///                    <code>DontLock</code> is used. In the same case, if <code>Lock</code>
            ///                    or <code>Auto</code> is used, the column will be locked before modification and
            ///                    unlocked afterwards. In case this column is configured to never lock at all
            ///                    and <code>Lock</code> is given, the request will be rejected. If the policy is set
            ///                    to <code>Auto</code>, the locking behavior is determined by
            ///                    <code>ThreadSafenessPolicy</code> as configured in the constructor of this column.
            /// \param tuplet_id_policy [in] The policy that controls whether tuplet ids should be recycled if there are
            ///                         some free ids, or if always new tuplet ids will be created.
            /// \param conflict_policy [in] The policy that controls whether null values should be added, or whether
            ///                             values should be auto-incremented (if there is a conflict w.r.t.
            ///                             it is not decidable which one to chose). In case no conflict occurs
            ///                             (i.e., either null-values are allowed and no auto-incrementation (or vice
            ///                             versa), or neither null-values are allowed nor auto-incrementation is
            ///                             enabled, this parameter must be at least match the configuration, or it must
            ///                             be <code>DontCare</code>. In case this parameter does not match the
            ///                             configuration (e.g., the policy states to add null values but the column
            ///                             is not configured to be able to contain null values), the request for
            ///                             adding new values to this column will be rejected. Likewise, if the
            ///                             policy is configured for <code>DontCare</code> but there is a conflict,
            ///                             the request will be rejected, too.
            /// \return This method returns
            ///             <ul>
            ///                 <li><code>OK</code> if the request was successfully executed.</li>
            ///                 <li><code>UnresolvedConflict</code> if \p values is <code>null</code> and
            ///                     \p conflict_policy does not resolve the conflict if both nullable values and
            ///                     auto-incrementation is turned on for this column.</li>
            ///                 <li><code>UnsupportedOpertation</code> if \p conflict_policy requests for an
            ///                     proceeding that is not supported for this column configuration (e.g., add
            ///                     null values but null values are not allowed).</li>
            ///                 <li><code>LockNotSupported</code> if \p lock_policy forces to use locking but locking
            ///                     is disabled for this column.</li>
            ///                 <li><code>CapacityExceeded</code> if there are more values to be added than the
            ///                     column is able to manage. This specific limit depends on the type of
            ///                     <code>tuplet_id_t</code> that was statically defined for this column.</li>
            ///                 <li><code>ForeignKeyConstraintViolated</code> if this column is configured to
            ///                     contain values from a range outside this column and at least one value in
            ///                     \p values does not satisfy this requirement. This check only considers non-compound
            ///                     (explicit) foreign key value constraints.</li>
            ///                 <li><code>PrimaryKeyConstraintViolated</code> if this column is configured to
            ///                     contain unique values used as primary key and at least one value in \p values
            ///                     does not satisfy this requirement. This check only considers non-compound
            ///                     (explicit) primary key value constraints. <b>Note</b>: This check is only
            ///                     done if <code>NDEBUG</code> is <i>not</i> defined.</li>
            ///                 <li><code>IllegalNullReference</code> if \p values is null, this column
            ///                     has a non-unrestricted key value policy, and this column is not a non-compound
            ///                     primary key (i.e., this column is configured as a foreign key, or compound primary
            ///                     key).</li>
            ///                 <li><code>ExplicitPrimiaryKeyMightNotHold</code> if \p values is null,
            ///                     this column is configured as a non-compound primary key, but either this column
            ///                     is able to contain null values, or auto-incrementation is not enabled.</li>
            ///                 <li>code>ValueConstraintNotSatisfied</code> the values to be added (or the values
            ///                     with consideration of the already stored values) does not satify a given
            ///                     constraint.</li>
            ///                 <li>code>ModificationNotAllowed</code> if this column is read only
            ///                                                        (see constructor)</li>
            ///             </ul>
            ///         This method might return (depending on the implementation)
            ///             <ul>
            ///                 <li><code>ResizingFailed</code> if the operation requires the column to reserve further
            ///                     memory to satisfy the request, but this operation fails.</li>
            ///                 <li><code>RecyclingNoSupported</code> if the implementation does not support to recycle
            ///                                                 tuplet ids but the caller requested this.</li>
            ///                 <li><code>NotImplemented</code> if the implementation does not implement this
            ///                                                 method.</li>
            ///                 <li><code>PartlyNotImplemented</code> if the implementation does not implement this
            ///                                                 specific configuration for the method.</li>
            ///                 <li><code>IllegalOperation</code> if the implementation does not allow to call this
            ///                                                   method.</li>
            ///                 <li><code>InternalError</code> if an internal error occurs (e.g., missing switch
            ///                                                case).</li>
            ///             </ul>
            ///
            /// \note This method will block the calling thread. In case the method request is rejected, it is
            ///       guaranteed that the content is not changed.
            /// \author Marcus Pinnecke
            /// \date 2017-02-15
            /// \since 1.0.00
            ErrorType append(tuplet_id_t *tuplet_ids,
                                     size_t *num_of_tuplet_ids,
                                     const value_t *values,
                                     size_t num_of_values,
                                     DeduplicationPolicy deduplication_policy = DeduplicationPolicy::RunDeduplicationIfNeeded,
                                     LockHandling lock_policy = LockHandling::Auto,
                                     ReuseOfTupletIdsPolicy tuplet_id_policy = ReuseOfTupletIdsPolicy::RecycleTupletIdsIfPossible,
                                     AutoIncAndNullConflictPolicy conflict_policy = AutoIncAndNullConflictPolicy::DontCare
            ) noexcept
            {


                return on_append(tuplet_ids, num_of_tuplet_ids, values, num_of_values, deduplication_policy,
                                 lock_policy, tuplet_id_policy, conflict_policy);
            }

            ErrorType update(const tuplet_id_t *tuplets, size_t num_of_ids, const value_t *new_value) noexcept
            {
                return on_update(tuplets, num_of_ids, new_value);
            }

            ErrorType remove(const tuplet_id_t *tuplets, size_t num_of_ids, RemoveExecutionPolicy policy = RemoveExecutionPolicy::ImmediatlyRemove) noexcept
            {
                return on_remove(tuplets, num_of_ids, policy);
            }

            void clear() noexcept
            {
                reinitialize();
                return on_clear();
            }



            // Todo: Transformer!



            bool is_locked() const noexcept
            {
                return locked_flag;
            }

            const char *get_column_name() const noexcept
            {
                return column_name;
            }

            NullPolicy get_null_policy() const noexcept
            {
                return null_policy;
            };

            KeyPolicy get_key_policy() const noexcept
            {
                return key_policy;
            }

            Trilean contains_only_unique_values() const noexcept
            {
                return contains_unique_values;
            }

            AutoIncrementPolicy get_auto_increment_policy() const noexcept
            {
                return auto_increment_policy;
            }

            AccessPolicy get_access_policy() const noexcept
            {
                return access_policy;
            }

            ThreadSafenessPolicy get_lock_policy() const noexcept
            {
                return lock_policy;
            }

            UpdatePolicy get_update_policy() const noexcept
            {
                return update_policy;
            }

            CollectionBehaviorPolicy get_behavior_policy() const noexcept
            {
                return collection_behavior_policy;
            }

            virtual void dispose() noexcept
            {
                if (column_name != nullptr)
                    free (column_name);
                on_dispose();
            }

        };

    }

}

#endif //PANTHEON_BASE_COLUMN_HPP
