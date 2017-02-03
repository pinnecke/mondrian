#pragma once

#include <Shared/Common.h>
#include <cstring>

extern "C"
{
	/* The following functions are implemented in CUDA C, compiled with NVCC and linked to host code */

	DEVICE_MEM_HANDLE deviceCopyFromHostToDevice(const HOST_MEM_POINTER SourceHost, size_t NumberOfBytes);
    bool deviceCopyFromDeviceToHost(HOST_MEM_POINTER DestinationHost, DEVICE_MEM_HANDLE SourceDevice, size_t NumberOfBytes);
    void deviceCleanUp(DEVICE_MEM_HANDLE *DevHandle, size_t NumDevHandle);

    DEVICE_MEM_HANDLE deviceQueryEvaluateSum(DEVICE_MEM_HANDLE DevicePriceColumnHandle, size_t NumberOfItems, bool MultipleThreads);
    size_t deviceQueryFetchSumValue(DEVICE_MEM_HANDLE deviceResultHandle);

    DEVICE_MEM_HANDLE *deviceQueryEvaluateManySum(DEVICE_MEM_HANDLE *DevicePriceColumnHandle, size_t NumberOfColumns, size_t NumberOfItems, bool MultipleThreads);
    size_t *deviceQueryFetchManySumValues(DEVICE_MEM_HANDLE *deviceResultHandle, size_t NumberOfColumns);
}

struct DeviceQueryForDataQ1ColumnStore : public QueryForDataQ1
{
	size_t *HostPriceColumn;
	DEVICE_MEM_HANDLE PriceColumn;
	DEVICE_MEM_HANDLE TotalPriceSum;
	size_t NumberOfItems;

	DeviceQueryForDataQ1ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *ItemTable, QueryParamsQ1 Params,
                              ThreadingPolicy Policy, size_t NumberOfItems):
            QueryForDataQ1(ResultSet, Params, Policy), HostPriceColumn((size_t *) ItemTable->I_PRICE),
            PriceColumn(NULL), NumberOfItems(NumberOfItems) { }

    virtual void CopyToDevice()
    {
    	PriceColumn = deviceCopyFromHostToDevice(HostPriceColumn, NumberOfItems * sizeof(size_t));
    }

    virtual void operator()()
    {
    	if (PriceColumn != NULL) {
    		TotalPriceSum = deviceQueryEvaluateSum(PriceColumn, NumberOfItems, (Policy == ThreadingPolicy::MultiThreaded));
    	}
    }

    virtual void ReceiveFromDevice()
    {
    	if (PriceColumn != NULL && TotalPriceSum != NULL)
    	{
        	ResultSet->CustomerId = Params.CustomerTupleId;
        	ResultSet->TotalPrice = deviceQueryFetchSumValue(TotalPriceSum);
    	} else {
    		fprintf(stderr, "ERROR: cannot fetch data from device. Something went wrong.\n");
    	}
    }

    virtual void CleanUp()
    {
    	DEVICE_MEM_HANDLE handles[2] = { PriceColumn, TotalPriceSum };
    	deviceCleanUp(handles, 2);
    }
};



struct DeviceQueryForDataQ2ColumnStore : public DeviceQueryForDataQ1ColumnStore
{
	DEVICE_MEM_HANDLE PriceColumn[COLUMN_NUMBER_TO_COPY];
	DEVICE_MEM_HANDLE *TotalPriceSum;
	size_t *Sums;

	DeviceQueryForDataQ2ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *ItemTable, QueryParamsQ1 Params,
	                              ThreadingPolicy Policy, size_t NumberOfItems):
	                            	  DeviceQueryForDataQ1ColumnStore(ResultSet, ItemTable, Params, Policy,
									  NumberOfItems)
	{
		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
			PriceColumn[i] = NULL;
		}

		TotalPriceSum = NULL;
		Sums = NULL;
	}



	virtual void CopyToDevice() override
	{
		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
			PriceColumn[i] = deviceCopyFromHostToDevice(HostPriceColumn, NumberOfItems * sizeof(size_t));
		}
	}

	virtual void operator()() override
	{
		if (PriceColumn != NULL) {
			bool allNonNull = true;
			for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY && allNonNull; ++i) {
				allNonNull = (PriceColumn[i] != NULL);
			}
			if (allNonNull) {
				TotalPriceSum = deviceQueryEvaluateManySum(PriceColumn, COLUMN_NUMBER_TO_COPY, NumberOfItems, (Policy == ThreadingPolicy::MultiThreaded));
			} else fprintf(stderr, "ERROR: single column in price column array is NULL.\n");
		} else fprintf(stderr, "ERROR: price column array is NULL.\n");
	}

	virtual void ReceiveFromDevice() override
	{
		bool valid = (TotalPriceSum != NULL);

		for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY && valid; ++i) {
			valid = (PriceColumn[i] != NULL) && (TotalPriceSum[i] != NULL);
		}

		if (valid)
		{
			ResultSet->CustomerId = Params.CustomerTupleId;
			Sums = deviceQueryFetchManySumValues(TotalPriceSum, COLUMN_NUMBER_TO_COPY);
			// ResultSet->TotalPrice[i = 1...COLUMN_NUMBER_TO_COPY] = ...
			for (size_t i = 0; i < COLUMN_NUMBER_TO_COPY; ++i) {
				ResultSet->TotalPrice = Sums[i];			// some effort w/o new structure ResultSet' //
			}
		} else {
			fprintf(stderr, "ERROR: cannot fetch data from device. Something went wrong.\n");
		}
	}


	virtual void CleanUp() override
	{
		DEVICE_MEM_HANDLE handles[2 * COLUMN_NUMBER_TO_COPY];
		std::memcpy(handles, PriceColumn, COLUMN_NUMBER_TO_COPY * sizeof(DEVICE_MEM_HANDLE));
		if (TotalPriceSum != NULL) {
			std::memcpy(handles + COLUMN_NUMBER_TO_COPY, TotalPriceSum, COLUMN_NUMBER_TO_COPY * sizeof(DEVICE_MEM_HANDLE));
		}
		deviceCleanUp(handles, (TotalPriceSum != NULL)? 2 * COLUMN_NUMBER_TO_COPY : COLUMN_NUMBER_TO_COPY);
		if (Sums != NULL)
			free (Sums);
		if (TotalPriceSum != NULL)
			free (TotalPriceSum);
	}
};
