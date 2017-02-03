#pragma once

#include <Shared/Common.h>

extern "C"
{
	void *CopyDataToDevice(const void *SourceHost, unsigned long long NumberOfBytes);
    bool CopyDataFromDevice(void *DestinationHost, void *SourceDevice, unsigned long long NumberOfBytes);
    bool FreeDataInDevice(void *DestinationDevice);
    void ResetDevice();
    unsigned long long EvaluateSum(void *DevicePriceColumnHandle, unsigned long long NumberOfItems, bool MultipleThreads);
}

struct DeviceQueryForDataQ1ColumnStore : public QueryForDataQ1
{
	unsigned long long *hostPriceColumnPointer;
	void *devicePriceColumnHandle;
	size_t NumberOfItems;

	DeviceQueryForDataQ1ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *itemTableColumnStore, QueryParamsQ1 Params,
                              ThreadingPolicy Policy, size_t NumberOfItems):
            QueryForDataQ1(ResultSet, Params, Policy), hostPriceColumnPointer((unsigned long long *) itemTableColumnStore->I_PRICE),
            devicePriceColumnHandle(NULL), NumberOfItems(NumberOfItems) { }

    void CopyToDevice() {

    	// TODO: XXX
    	for (size_t i = 0; i < NumberOfItems; i++)
    		hostPriceColumnPointer[i] = 1;

    	devicePriceColumnHandle = CopyDataToDevice(hostPriceColumnPointer, NumberOfItems * sizeof(unsigned long long *));
    	// !!!!! TODO: Nicht die ganze Liste wird gelesen, sondern nur über das Index Array!
    }

    void operator()() const
    {
    	ResultSet->TotalPrice = EvaluateSum(devicePriceColumnHandle, NumberOfItems, (Policy == ThreadingPolicy::MultiThreaded));
    	ResultSet->CustomerId = Params.CustomerTupleId;

    	printf("TOTAL: %zu, #ITEMS: %zu\n",ResultSet->TotalPrice ,NumberOfItems);
    }

    void ReceiveFromDevice()
    {
//    	if (devicePriceColumnHandle != NULL) {
//			if(!CopyDataFromDevice(&ResultSet->TotalPrice, devicePriceColumnHandle, sizeof(size_t)))
//				return;
//			if(!FreeDataInDevice(devicePriceColumnHandle))
//				return;
//			ResetDevice();
//    	}
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
