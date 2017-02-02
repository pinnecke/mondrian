#include "Shared/Common.h"
#include "Shared/Utility.h"
#include "Host/Query.h"

const size_t ITEMS_BOUGTH_BY_CUSTOMER_AVG = 150;
const size_t BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS = 40000;

void WriteOberservation(size_t NumberOfCustomers, size_t NumberOfItems,
                        size_t NumberOfRecordsToProcess, size_t CurrentRepetition,
                        const char *StorageModelName,
                        size_t DurationQ1SingleThreaded, size_t DurationQ2SingleThreaded, size_t DurationQ3SingleThreaded,
                        size_t DurationQ1MultiThreaded, size_t DurationQ2MultiThreaded, size_t DurationQ3MultiThreaded)
{
    size_t dataSetSizeCustomersInByte = NumberOfCustomers * (sizeof(uint64_t) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned) +
                                                             sizeof(uint64_t) +
                                                             sizeof(unsigned) +
                                                             sizeof(uint64_t) +
                                                             sizeof(uint32_t) +
                                                             sizeof(uint64_t) +
                                                             sizeof(short) +
                                                             sizeof(short) +
                                                             sizeof(unsigned) +
                                                             sizeof(unsigned));


    size_t dataSetSizeItemsInByte = NumberOfItems * (sizeof(uint64_t) +
                                                     sizeof(unsigned) +
                                                     sizeof(unsigned) +
                                                     sizeof(unsigned) +
                                                     sizeof(unsigned));

    size_t dataSetSizeJoinTableInByte = sizeof(size_t) * NumberOfRecordsToProcess + sizeof(size_t);

    printf("%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%s\n",
           (size_t)time(NULL),
           (size_t)NumberOfCustomers,
           (size_t)CurrentRepetition,
           (size_t)DurationQ1SingleThreaded,
           (size_t)DurationQ2SingleThreaded,
           (size_t)DurationQ3SingleThreaded,
           (size_t)DurationQ1MultiThreaded,
           (size_t)DurationQ2MultiThreaded,
           (size_t)DurationQ3MultiThreaded,
           dataSetSizeCustomersInByte,
           dataSetSizeItemsInByte,
           dataSetSizeJoinTableInByte,
           NumberOfRecordsToProcess,
           StorageModelName);
    fflush(stdout);
}

void SampleColumnStore(size_t NumberOfCustomers, size_t NumberOfItems, size_t CurrentRepetition)
{
    srand(0);

    /* Setup */
    CustomerTableDSM customerTable;
    ItemTableDSM itemTable;

    CreateOrdersTables(&customerTable, NumberOfCustomers);
    CreateItemsTables(&itemTable, NumberOfItems);
    FillOrdersTable(&customerTable, NumberOfCustomers);
    FillItemsTable(&itemTable, NumberOfItems);

    /* Query Q1 */
    ResultSetQ1 result1st, result1mt;
    size_t itemsBougth = ITEMS_BOUGTH_BY_CUSTOMER_AVG;
    auto queryParamsQ1 = CreateQueryParamsQ1(NumberOfCustomers, itemsBougth, NumberOfItems);

    auto durationQ1st = measure<>::run(QueryForDataQ1ColumnStore(&result1st, &itemTable, queryParamsQ1, ThreadingPolicy::SingleThreaded));
    auto durationQ1mt = measure<>::run(QueryForDataQ1ColumnStore(&result1mt, &itemTable, queryParamsQ1, ThreadingPolicy::MultiThreaded));

    DisposeQueryParamsQ1(&queryParamsQ1);

    /* Query Q2 */
    ResultSetQ2 result2st, result2mt;
    size_t buyingCustomers = BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS;
    auto queryParamsQ2 = CreateQueryParamsQ2(NumberOfItems, buyingCustomers, NumberOfCustomers);

    auto durationQ2st = measure<>::run(QueryForDataQ2ColumnStore(&result2st, &customerTable, queryParamsQ2, ThreadingPolicy::SingleThreaded));
    auto durationQ2mt = measure<>::run(QueryForDataQ2ColumnStore(&result2mt, &customerTable, queryParamsQ2, ThreadingPolicy::MultiThreaded));
    DisposeQueryParamsQ2(&queryParamsQ2);

    /* Query Q3 */
    ResultSetQ1 result3st, result3mt;
    auto queryParamsQ3 = CreateQueryParamsQ1(NumberOfCustomers, NumberOfItems, NumberOfItems);

    auto durationQ3st = measure<>::run(QueryForDataQ1ColumnStore(&result3st, &itemTable, queryParamsQ3, ThreadingPolicy::SingleThreaded));
    auto durationQ3mt = measure<>::run(QueryForDataQ1ColumnStore(&result3mt, &itemTable, queryParamsQ3, ThreadingPolicy::MultiThreaded));

    DisposeQueryParamsQ1(&queryParamsQ3);

    WriteOberservation(NumberOfCustomers, NumberOfItems, queryParamsQ1.NumOfItemsTupleIds, CurrentRepetition,
                       "ColumnStore",
                       durationQ1st, durationQ2st, durationQ3st, durationQ1mt, durationQ2mt, durationQ3mt);

    /* Cleanup */
    free(result2st.CustomerInfos);
    free(result2mt.CustomerInfos);
    DisposeTables(&customerTable, &itemTable);
}

void SampleRowStore(size_t NumberOfCustomers, size_t NumberOfItems, size_t CurrentRepetition)
{
    srand(0);

    /* Setup */
    CustomerTableNSM customerTable;
    ItemTableNSM itemTable;

    CreateOrdersTables(&customerTable, NumberOfCustomers);
    CreateItemsTables(&itemTable, NumberOfItems);
    FillOrdersTable(&customerTable, NumberOfCustomers);
    FillItemsTable(&itemTable, NumberOfItems);

    /* Query Q1 */
    ResultSetQ1 result1st, result1mt;
    size_t itemsBougth = ITEMS_BOUGTH_BY_CUSTOMER_AVG;
    auto queryParamsQ1 = CreateQueryParamsQ1(NumberOfCustomers, itemsBougth, NumberOfItems);

    auto durationQ1st = measure<>::run(QueryForDataQ1RowStore(&result1st, &itemTable, queryParamsQ1, ThreadingPolicy::SingleThreaded));
    auto durationQ1mt = measure<>::run(QueryForDataQ1RowStore(&result1mt, &itemTable, queryParamsQ1, ThreadingPolicy::MultiThreaded));

    DisposeQueryParamsQ1(&queryParamsQ1);

    /* Query Q2 */
    ResultSetQ2 result2st, result2mt;
    size_t buyingCustomers = BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS;
    auto queryParamsQ2 = CreateQueryParamsQ2(NumberOfItems, buyingCustomers, NumberOfCustomers);

    auto durationQ2st = measure<>::run(QueryForDataQ2RowStore(&result2st, &customerTable, queryParamsQ2, ThreadingPolicy::SingleThreaded));
    auto durationQ2mt = measure<>::run(QueryForDataQ2RowStore(&result2mt, &customerTable, queryParamsQ2, ThreadingPolicy::MultiThreaded));

    DisposeQueryParamsQ2(&queryParamsQ2);

    /* Query Q3 */
    ResultSetQ1 result3st, result3mt;
    auto queryParamsQ3 = CreateQueryParamsQ1(NumberOfCustomers, NumberOfItems, NumberOfItems);

    auto durationQ3st = measure<>::run(QueryForDataQ1RowStore(&result3st, &itemTable, queryParamsQ3, ThreadingPolicy::SingleThreaded));
    auto durationQ3mt = measure<>::run(QueryForDataQ1RowStore(&result3mt, &itemTable, queryParamsQ3, ThreadingPolicy::MultiThreaded));

    DisposeQueryParamsQ1(&queryParamsQ3);

    WriteOberservation(NumberOfCustomers, NumberOfItems, queryParamsQ1.NumOfItemsTupleIds, CurrentRepetition,
                       "RowStore",
                       durationQ1st, durationQ2st, durationQ3st, durationQ1mt, durationQ2mt, durationQ3mt);

    /* Cleanup */
    free(result2st.CustomerInfos);
    free(result2mt.CustomerInfos);
    DisposeTables(&customerTable, &itemTable);

}

void Sample(size_t N, size_t NumberOfRepetitions) {
    size_t NUMBER_OF_ITEMS = 0.7f * N;
    for (size_t currentRepetition = 0; currentRepetition < NumberOfRepetitions; currentRepetition++) {
        SampleColumnStore(N, NUMBER_OF_ITEMS, currentRepetition);
        SampleRowStore(N, NUMBER_OF_ITEMS, currentRepetition);
    }
}

int main() {

    size_t numberOfIndependentVariableSamples = 30;
    size_t numberOfRepititions = 150;
    size_t numberOfCustomersStart = 300000;
    size_t numberOfRecordsEnd = numberOfCustomersStart * 10 * 15;
    size_t stepSize = (numberOfRecordsEnd - numberOfCustomersStart)  / numberOfIndependentVariableSamples;

    printf("Timestamp;"
           "N;"
           "Repetition;"
           "Q1TimeInNanoSec_SingleThreaded;"
           "Q2TimeInNanoSec_SingleThreaded;"
           "Q3TimeInNanoSec_SingleThreaded;"
           "Q1TimeInNanoSec_MultiThreaded;"
           "Q2TimeInNanoSec_MultiThreaded;"
           "Q3TimeInNanoSec_MultiThreaded;"
           "CustomerTableInByte;"
           "ItemTableInByte;"
           "JoinTableInByte;"
           "NumRecordsToProcess;"
           "Type\n");

    //for (size_t N = numberOfCustomersStart; N <= numberOfRecordsEnd; N += stepSize) {
    for (size_t N = numberOfCustomersStart; true; N += stepSize) {
        Sample(N, numberOfRepititions);
    }

    return EXIT_SUCCESS;
}