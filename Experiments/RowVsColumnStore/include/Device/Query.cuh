#pragma once

#include <Shared/Common.h>

extern "C"
{
	void *CopyDataToDevice(const void *SourceHost, unsigned long long NumberOfBytes);
    bool CopyDataFromDevice(void *DestinationHost, void *SourceDevice, unsigned long long NumberOfBytes);
    bool FreeDataInDevice(void *DestinationDevice);
    void ResetDevice();
    void *EvaluateSum(void *DevicePriceColumnHandle, size_t NumberOfItems, bool MultipleThreads);
    size_t GetSumValue(void *deviceResultHandle);
    void cleanUp(void *DevicePriceColumnHandle, void *deviceResultHandle);
}

struct DeviceQueryForDataQ1ColumnStore : public QueryForDataQ1
{
	unsigned long long *hostPriceColumnPointer;
	void *devicePriceColumnHandle;
	void *deviceResultHandle;
	size_t NumberOfItems;

	DeviceQueryForDataQ1ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *itemTableColumnStore, QueryParamsQ1 Params,
                              ThreadingPolicy Policy, size_t NumberOfItems):
            QueryForDataQ1(ResultSet, Params, Policy), hostPriceColumnPointer((unsigned long long *) itemTableColumnStore->I_PRICE),
            devicePriceColumnHandle(NULL), NumberOfItems(NumberOfItems) { }

    void CopyToDevice() {

    	// TODO: XXX
    	for (size_t i = 0; i < NumberOfItems; i++)
    		hostPriceColumnPointer[i] = 2;

    	devicePriceColumnHandle = CopyDataToDevice(hostPriceColumnPointer, NumberOfItems * sizeof(unsigned long long *));
    	// !!!!! TODO: Nicht die ganze Liste wird gelesen, sondern nur über das Index Array!
    }

    void operator()()
    {
    	deviceResultHandle = EvaluateSum(devicePriceColumnHandle, NumberOfItems, (Policy == ThreadingPolicy::MultiThreaded));
    }

    void ReceiveFromDevice()
    {
    	if (hostPriceColumnPointer != NULL && deviceResultHandle != NULL)
    	{
        	ResultSet->CustomerId = Params.CustomerTupleId;
        	ResultSet->TotalPrice = GetSumValue(deviceResultHandle);

        	printf("TOTAL: %zu, #ITEMS: %zu\n",ResultSet->TotalPrice ,NumberOfItems);

    	} else {
    		fprintf(stderr, "ERROR: cannot fetch data from device. Something went wrong.\n");
    	}





//    	if (devicePriceColumnHandle != NULL) {
//			if(!CopyDataFromDevice(&ResultSet->TotalPrice, devicePriceColumnHandle, sizeof(size_t)))
//				return;
//			if(!FreeDataInDevice(devicePriceColumnHandle))
//				return;
//			ResetDevice();
//    	}
    }

    void CleanUp()
    {

    }
};






//struct QueryForDataQ1RowStore : public QueryForDataQ1
//{
//    ItemTableNSM *itemTableColumnStore;
//
//    QueryForDataQ1RowStore(ResultSetQ1 *ResultSet, ItemTableNSM *itemTableColumnStore, QueryParamsQ1 Params,
//                           ThreadingPolicy Policy):
//            QueryForDataQ1(ResultSet, Params, Policy), itemTableColumnStore(itemTableColumnStore) { }
//
//    void operator()() const;
//};
//
//struct QueryForDataQ2ColumnStore : public QueryForDataQ2 {
//
//    CustomerTableDSM *customerTableColumnStore;
//
//    QueryForDataQ2ColumnStore(ResultSetQ2 *ResultSet, CustomerTableDSM *customerTableColumnStore,
//                              QueryParamsQ2 Params, ThreadingPolicy Policy):
//            QueryForDataQ2(ResultSet, Params, Policy), customerTableColumnStore(customerTableColumnStore) { }
//
//    void operator()() const;
//};
//
//struct QueryForDataQ2RowStore : public QueryForDataQ2 {
//
//    CustomerTableNSM *customerTableColumnStore;
//
//    QueryForDataQ2RowStore(ResultSetQ2 *ResultSet, CustomerTableNSM *customerTableColumnStore,
//                           QueryParamsQ2 Params, ThreadingPolicy Policy):
//            QueryForDataQ2(ResultSet, Params, Policy), customerTableColumnStore(customerTableColumnStore) { }
//
//    void operator()() const;
//};
