#pragma once

#include <Shared/Common.h>

extern "C"
{
	/* The following functions are implemented in CUDA C, compiled with NVCC and linked to host code */

	DEVICE_MEM_HANDLE deviceCopyFromHostToDevice(const HOST_MEM_POINTER SourceHost, size_t NumberOfBytes);
    bool deviceCopyFromDeviceToHost(HOST_MEM_POINTER DestinationHost, DEVICE_MEM_HANDLE SourceDevice, size_t NumberOfBytes);
    void deviceCleanUp(DEVICE_MEM_HANDLE DevicePriceColumnHandle, DEVICE_MEM_HANDLE deviceResultHandle);

    DEVICE_MEM_HANDLE deviceQueryEvaluateSum(DEVICE_MEM_HANDLE DevicePriceColumnHandle, size_t NumberOfItems, bool MultipleThreads);
    size_t deviceQueryFetchSumValue(DEVICE_MEM_HANDLE deviceResultHandle);
}

struct DeviceQueryForDataQ3ColumnStore : public QueryForDataQ1
{
	size_t *HostPriceColumn;
	DEVICE_MEM_HANDLE PriceColumn;
	DEVICE_MEM_HANDLE TotalPriceSum;
	size_t NumberOfItems;

	DeviceQueryForDataQ3ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *ItemTable, QueryParamsQ1 Params,
                              ThreadingPolicy Policy, size_t NumberOfItems):
            QueryForDataQ1(ResultSet, Params, Policy), HostPriceColumn((size_t *) ItemTable->I_PRICE),
            PriceColumn(NULL), NumberOfItems(NumberOfItems) { }

    void CopyToDevice()
    {
    	// TODO: XXX
    	for (size_t i = 0; i < NumberOfItems; i++)
    		HostPriceColumn[i] = 2;

    	PriceColumn = deviceCopyFromHostToDevice(HostPriceColumn, NumberOfItems * sizeof(size_t *));
    }

    void operator()()
    {
    	TotalPriceSum = deviceQueryEvaluateSum(PriceColumn, NumberOfItems, (Policy == ThreadingPolicy::MultiThreaded));
    }

    void ReceiveFromDevice()
    {
    	if (HostPriceColumn != NULL && TotalPriceSum != NULL)
    	{
        	ResultSet->CustomerId = Params.CustomerTupleId;
        	ResultSet->TotalPrice = deviceQueryFetchSumValue(TotalPriceSum);

        	printf("TOTAL: %zu, #ITEMS: %zu\n",ResultSet->TotalPrice ,NumberOfItems);

    	} else {
    		fprintf(stderr, "ERROR: cannot fetch data from device. Something went wrong.\n");
    	}
    }

    void CleanUp()
    {
    	deviceCleanUp(PriceColumn, TotalPriceSum);
    }
};
