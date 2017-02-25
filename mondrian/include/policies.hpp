#ifndef PANTHEON_POLICIES_HPP
#define PANTHEON_POLICIES_HPP

namespace pantheon
{
    enum class UpdatePolicy
    {
        InPlaceUpdates, MultiVersionFields
    };

    enum class NullHandlingPolicy
    {
        AlwaysIncludeNullValues, AlwaysExcludeNullValues
    };

    enum class ScanStrategy
    {
        Sequential, Parallel
    };

    enum class NullPolicy
    {
        NonNull, Nullable
    };

    enum class AutoIncrementPolicy
    {
        NoAutoIncrement, AutoIncrement
    };

    enum class CollectionBehaviorPolicy
    {
        Set, Bag
    };

    enum class Trilean
    {
        True, False, Unknown
    };

    enum class KeyPolicy
    {
        NoRestriction, ExplicitPrimaryKey, ExplicitForeignKey, CompoundPrimaryKey, CompoundForeignKey
    };

    enum class AccessPolicy
    {
        ReadOnly, AppendOnly, ReadAppend
    };

    enum class ColumnType
    {
        HostVectorColumn
    };

    enum class RemoveExecutionPolicy
    {
        ImmediatlyRemove, LazyRemove
    };

    enum class ReuseOfTupletIdsPolicy
    {
        RecycleTupletIdsIfPossible, ForceNewTupletIdCreation
    };

    enum class AutoIncAndNullConflictPolicy
    {
        AddNullValues, IncrementValues, DontCare
    };

    enum class DeduplicationPolicy
    {
        RunDeduplicationIfNeeded, ForceDeduplication, DontCare
    };

    enum class LockHandling
    {
        Auto, Lock, DontLock
    };

    enum class ThreadSafenessPolicy
    {
        UseLocks, DontUseLocks
    };

    enum class NullDuplicateHandlingPolicy
    {
        TreatAsRegularValue, AllowDuplicateNullValues, RemoveNullValues
    };

    enum class ValueOrderPolicy
    {
        Stable, Instable
    };
}

#endif //PANTHEON_POLICIES_HPP
