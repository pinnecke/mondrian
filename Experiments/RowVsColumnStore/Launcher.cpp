#include "Shared/Common.h"
#include "Shared/Utility.h"
#include "Host/Query.h"
#include "Device/Query.cuh"
#include <unistd.h>

const size_t ITEMS_BOUGTH_BY_CUSTOMER_AVG = 150;
const size_t BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS = 40000;

#define NANO_TO_SEC(x)   ( x / (float)1e9 )
#define BYTE_TO_GBYTE(x) ( x / (float)1e9 ) /* GB, not GiB! */

statm_t strat;

void WriteOberservation(size_t NumberOfCustomers, size_t NumberOfItems,
                        size_t NumberOfRecordsToProcess, size_t CurrentRepetition,
                        bool columnStore,
                        size_t DurationQ1SingleThreaded, size_t DurationQ2SingleThreaded, size_t DurationQ3SingleThreaded,
                        size_t DurationQ1MultiThreaded, size_t DurationQ2MultiThreaded, size_t DurationQ3MultiThreaded,
                        bool hostCode, size_t ProcessVmSize)
{
	size_t dataSetSizeCustomersInByte = NumberOfCustomers * sizeof(CustomerTableNSM::Tuple);
	size_t dataSetSizeItemsInByte = NumberOfItems * sizeof(ItemTableNSM::Tuple);
    size_t dataSetSizeJoinTableInByte = NumberOfRecordsToProcess * sizeof(size_t*);

    size_t Q1NumRecordsToProcess = hostCode? NumberOfRecordsToProcess : 0;
    size_t Q1NumRecordToProcessedDeRefSize = hostCode? (columnStore? sizeof(size_t) : sizeof(ItemTableNSM::Tuple)) : 0;
    float Q1GB = float(Q1NumRecordsToProcess * Q1NumRecordToProcessedDeRefSize);
    float Q1GBpsecSingleThreaded = hostCode? (BYTE_TO_GBYTE (Q1GB) / NANO_TO_SEC(DurationQ1SingleThreaded)) : 0;
    float Q1GBpsecMultiThreaded = hostCode? (BYTE_TO_GBYTE (Q1GB) / NANO_TO_SEC(DurationQ1MultiThreaded)) : 0;

    size_t Q2NumRecordsToProcess = NumberOfRecordsToProcess;
    size_t Q2NumRecordToProcessedDeRefSize = hostCode? (sizeof(CustomerTableNSM::Tuple)) : COLUMN_NUMBER_TO_COPY * sizeof(size_t);
    float Q2GB = float(Q2NumRecordsToProcess * Q2NumRecordToProcessedDeRefSize);
	float Q2GBpsecSingleThreaded = BYTE_TO_GBYTE (Q2GB) / NANO_TO_SEC(DurationQ2SingleThreaded);
	float Q2GBpsecMultiThreaded = BYTE_TO_GBYTE (Q2GB) / NANO_TO_SEC(DurationQ2MultiThreaded);

    size_t Q3NumRecordsToProcess = NumberOfItems;
    size_t Q3NumRecordToProcessedDeRefSize = columnStore? sizeof(size_t) : sizeof(ItemTableNSM::Tuple);
    float Q3GB = float(Q3NumRecordsToProcess * Q3NumRecordToProcessedDeRefSize);
	float Q3GBpsecSingleThreaded = BYTE_TO_GBYTE (Q3GB) / NANO_TO_SEC(DurationQ3SingleThreaded);
	float Q3GBpsecMultiThreaded = BYTE_TO_GBYTE (Q3GB) / NANO_TO_SEC(DurationQ3MultiThreaded);

    printf("%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%f;%f;%f;%f;%f;%f;%s;%s,%f\n",
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
           Q1NumRecordsToProcess,
		   0,
		   Q1NumRecordToProcessedDeRefSize,
		   Q2NumRecordsToProcess,
		   0,
		   Q2NumRecordToProcessedDeRefSize,
		   Q3NumRecordsToProcess,
		   0,
		   Q3NumRecordToProcessedDeRefSize,
		   Q1GBpsecSingleThreaded,
		   Q2GBpsecSingleThreaded,
		   Q3GBpsecSingleThreaded,
		   Q1GBpsecMultiThreaded,
		   Q2GBpsecMultiThreaded,
		   Q3GBpsecMultiThreaded,
		   columnStore? "ColumnStore" : "RowStore",
		   hostCode? "Host" : "Device",
		   (ProcessVmSize * getpagesize())/1024/1024/1024.0);
    fflush(stdout);
}

void SampleColumnStoreHost(size_t NumberOfCustomers, size_t NumberOfItems, size_t CurrentRepetition)
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

    read_off_memory_status(strat);

    DisposeQueryParamsQ1(&queryParamsQ3);

    WriteOberservation(NumberOfCustomers, NumberOfItems, queryParamsQ1.NumOfItemsTupleIds, CurrentRepetition,
                       true,
                       durationQ1st, durationQ2st, durationQ3st, durationQ1mt, durationQ2mt, durationQ3mt,
                       true, strat.resident);

    /* Cleanup */
    free(result2st.CustomerInfos);
    free(result2mt.CustomerInfos);
    DisposeTables(&customerTable, &itemTable);
}

void SampleRowStoreHost(size_t NumberOfCustomers, size_t NumberOfItems, size_t CurrentRepetition)
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
                       false,
                       durationQ1st, durationQ2st, durationQ3st, durationQ1mt, durationQ2mt, durationQ3mt,
                       true, strat.resident);

    /* Cleanup */
    free(result2st.CustomerInfos);
    free(result2mt.CustomerInfos);
    DisposeTables(&customerTable, &itemTable);
}


void SampleColumnStoreDevice(size_t NumberOfCustomers, size_t NumberOfItems, size_t CurrentRepetition)
{
    srand(0);

    /* Setup */
    CustomerTableDSM customerTable;
    ItemTableDSM itemTable;

    CreateOrdersTables(&customerTable, NumberOfCustomers);
    CreateItemsTables(&itemTable, NumberOfItems);
    FillOrdersTable(&customerTable, NumberOfCustomers);
    FillItemsTable(&itemTable, NumberOfItems);


   	size_t itemsBougth = ITEMS_BOUGTH_BY_CUSTOMER_AVG;


    /* Device Query Q2: Sum 10 price column */
	ResultSetQ1 result2st, result2mt;
	auto queryParamsQ2 = CreateQueryParamsQ1(NumberOfCustomers, itemsBougth, NumberOfItems);

	auto Query2st = DeviceQueryForDataQ2ColumnStore(&result2st, &itemTable, queryParamsQ2, ThreadingPolicy::SingleThreaded, NumberOfItems);
	auto durationQ2st_to   = measure<>::run([&Query2st] () { Query2st.CopyToDevice(); });
	auto durationQ2st_opp  = measure<>::run(Query2st);
	auto durationQ2st_from = measure<>::run([&Query2st] () { Query2st.ReceiveFromDevice(); });
	Query2st.CleanUp();

	auto Query2mt = DeviceQueryForDataQ2ColumnStore(&result2mt, &itemTable, queryParamsQ2, ThreadingPolicy::MultiThreaded, NumberOfItems);
	auto durationQ2mt_to   = measure<>::run([&Query2mt] () { Query2mt.CopyToDevice(); });
	auto durationQ2mt_opp  = measure<>::run(Query2mt);
	auto durationQ2mt_from = measure<>::run([&Query2mt] () { Query2mt.ReceiveFromDevice(); });
	Query2mt.CleanUp();

   	DisposeQueryParamsQ1(&queryParamsQ2);



	/* Device Query Q3: Sum single price column */
	ResultSetQ1 result3st, result3mt;
	auto queryParamsQ3 = CreateQueryParamsQ1(NumberOfCustomers, itemsBougth, NumberOfItems);

	auto Query3st = DeviceQueryForDataQ1ColumnStore(&result3st, &itemTable, queryParamsQ3, ThreadingPolicy::SingleThreaded, NumberOfItems);
	auto durationQ3st_to   = measure<>::run([&Query3st] () { Query3st.CopyToDevice(); });
	auto durationQ3st_opp  = measure<>::run(Query3st);
	auto durationQ3st_from = measure<>::run([&Query3st] () { Query3st.ReceiveFromDevice(); });
	Query3st.CleanUp();

	auto Query3mt = DeviceQueryForDataQ1ColumnStore(&result3mt, &itemTable, queryParamsQ3, ThreadingPolicy::MultiThreaded, NumberOfItems);
	auto durationQ3mt_to   = measure<>::run([&Query3mt] () { Query3mt.CopyToDevice(); });
	auto durationQ3mt_opp  = measure<>::run(Query3mt);
	auto durationQ3mt_from = measure<>::run([&Query3mt] () { Query3mt.ReceiveFromDevice(); });
	Query3mt.CleanUp();

	DisposeQueryParamsQ1(&queryParamsQ3);

	WriteOberservation(NumberOfCustomers, NumberOfItems, NumberOfItems, CurrentRepetition,
	                       true,
	                       0, durationQ2st_opp, durationQ3st_opp, 0, durationQ2mt_opp, durationQ3mt_opp,
	                       false, strat.resident);

    DisposeTables(&customerTable, &itemTable);
}

void Sample(size_t N, size_t NumberOfRepetitions) {
    size_t NUMBER_OF_ITEMS = 0.7f * N;
    for (size_t currentRepetition = 0; currentRepetition < NumberOfRepetitions; currentRepetition++) {
        SampleColumnStoreHost(N, NUMBER_OF_ITEMS, currentRepetition);
    	SampleRowStoreHost(N, NUMBER_OF_ITEMS, currentRepetition);
        SampleColumnStoreDevice(N, NUMBER_OF_ITEMS, currentRepetition);
    }
}

int main() {

    size_t numberOfIndependentVariableSamples = 30;
    size_t numberOfRepititions = 1;
    size_t numberOfCustomersStart = 1000000;
    size_t numberOfRecordsEnd = numberOfCustomersStart * 50;
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
           "Q1NumRecordsToProcess;"
    	   "Q1NumRecordToProcessedRefSize;"
    	   "Q1NumRecordToProcessedDeRefSize;"
		   "Q2NumRecordsToProcess;"
		   "Q2NumRecordToProcessedRefSize;"
		   "Q2NumRecordToProcessedDeRefSize;"
    	   "Q3NumRecordsToProcess;"
		   "Q3NumRecordToProcessedRefSize;"
		   "Q3NumRecordToProcessedDeRefSize;"
      	   "Q1GBpsecSingleThreaded;"
     	   "Q2GBpsecSingleThreaded;"
    	   "Q3GBpsecSingleThreaded;"
    	   "Q1GBpsecMultiThreaded;"
		   "Q2GBpsecMultiThreaded;"
		   "Q3GBpsecMultiThreaded;"
           "Type;"
           "ProcessResistendMemoryGiB\n");

    //for (size_t N = numberOfCustomersStart; N <= numberOfRecordsEnd; N += stepSize) {
    for (size_t N = numberOfCustomersStart; true; N += stepSize) {
        Sample(N, numberOfRepititions);
    }

    return EXIT_SUCCESS;
}
