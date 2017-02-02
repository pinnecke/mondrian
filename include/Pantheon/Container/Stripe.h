#include <Pantheon/error.h>
#include <Pantheon/stddef.h>

#ifndef GRIDSTORE_VECTOR_H
#define GRIDSTORE_VECTOR_H

namespace Pantheon {

    /*
     * Table size for HTAP grow over time; for efficient MMDBMS even cold data must kept in VM. In case table
     * becomes huge in size over time (i.e., data cannot be hold in physical memory), parts of the table are
     * swapped to disk (since traditional buffers are gone...).
     * Swapping is delegated to OS due to VM management. Modern MM rely heavily on parallelized
     * memory-bound scanning rather than cache-inefficent index structures such as B-Trees which also comes to
     * the cost of index maintainance during updates inside the table. Unfortunately, scanning requires to touch
     * all Tuples inside a relation which inherently requires to load previously swapped data from disk back
     * to VM. Suggestion: Divide table data into three parts: Hot, Cold, and Frozen Data.
     * Hot: fresh data of interest, heavily in use. Must be kept in continuous memory in MM for fast transactional
     *      processing. Is target of analytic (say once per day) along with Cold/Frozen data
     * Frozen: out-dated data which is "append" mostly, and does not significantly change. It's used
     *      primarily for analytic (say once per month) along with Hot/Cold data.
     * Cold: hot data that becomes frozen in the future, or vice versa
     *
     * Assumption: Frozen data is not "dead" but the majority of it won't be touched for a long time
     *
     * Goal: Avoid performance penalty of OS VM paging for MMDBMS having constant grow in table size
     * Problem: "All or nothing": let OS manage VM or have penalty of db-buffer manager
     *
     * Suggestion: Divide storage of single table data into two (three?) tiers: HotStore, and Cold Store
     *      Hot Store is continuous memory reserved for Hot data, in Main Memory
     *      Cold Store is db-buffered data that might be swapped to disk if not used regularly
     *      Transition between hot and cold store depending on access pattern to data as given by ever-day workload
     *
     */

    enum class DataType {
        Boolean,
        Byte,
        UnsignedByte,
        Short,
        UnsignedShort,
        Int,
        UnsignedInt,
        Long,
        UnsignedLong,
        Float,
        Double,
        Char,
        String
    };

    enum class StorageModel { NSM, DSM };

    struct RecordDescriptor {
        DataType *FieldTypes;
        DWORD NumberOfFields;
        WORD SingleTupleSizeInByte;
        StorageModel StorageModel;
    };

    struct StoreStatistics {
        DWORD NumberOfNullRecords;
        DWORD NumberRecordsInUse;
        DWORD NumberRecordsNotInUse;
        DWORD NumberHiddenRecords;
        DWORD NumberReadOnlyRecords;
        DWORD NumberPinnedRecords;
    };

    struct TableStatistics {
        StoreStatistics HotStoreStatistics;
        StoreStatistics ColdStoreStatistics;
    };

    struct Record {
        // Bit-packed record header
        struct {
            bool IsNull : 1;    // the record is set to be NULL
            bool IsUse : 1;    // the record is removed, and its position can be re-used
            bool IsHidden : 1;    // the record should not be listed for enumeration
            bool IsReadOnly : 1;    // the record value cannot be modified
            bool IsInHotStore : 1;
            bool IsPinned : 1;
        } RecordMetaData;

        struct {
            DWORD Version : 32;
            WORD NumberOfReads : 16;
            WORD NumberOfUpdates : 16;
            DWORD TimestampInserted : 30;
            DWORD TimestampFirstUpdated : 30;
            DWORD TimestampLastUpdated : 30;
            DWORD TimestampDeleted : 30;
            DWORD TimestampFirstRead : 30;
            DWORD TimestampFirstWrite : 30;
            DWORD TimestampLastRead : 30;
            DWORD TimestampLastWrite : 30;
        };
        BYTE **FieldOffsetsInHotStore;
    };

// A Stripe is a generalization of a vector data structure and suitable for both n-nary storage model (NSM) and
// decomposed storage model (DSM)
    struct Stripe
    {
        void *Data;
        QWORD Capacity;
        QWORD Size;

        PRESULT Create(Stripe *Stripe, QWORD Capacity, QWORD ElementSize);

        PRESULT Dispose(Stripe *Stripe);

        const BYTE *At(const Stripe *Stripe, QWORD Index);

        void *Front(const Stripe *Stripe);

        void *Back(const Stripe *Stripe);

        PRESULT IsEmpty(const Stripe *Stripe);

        PRESULT GetNumberOfElements(retval QWORD *number, const Stripe *Stripe);

        PRESULT ReserveElements(Stripe *Stripe, QWORD NumberOfElements);

        PRESULT GetCapacity(retval QWORD *capacity, const Stripe *Stripe);

        PRESULT ShrinkToFit(Stripe *Stripe);

        PRESULT Clear(Stripe *Stripe);

        PRESULT PushBack(Stripe *Stripe, const BYTE *data);

        PRESULT Remove(Stripe *Stripe, QWORD index);

        PRESULT Swap(Stripe *Stripe, QWORD PositionLeft, QWORD PositionRight);
    };



}

#endif //GRIDSTORE_VECTOR_H
