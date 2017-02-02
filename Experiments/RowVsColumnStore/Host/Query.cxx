#include "Query.h"
#include "Parallel.h"
#include "Operations.h"


void QueryForDataQ1ColumnStore::operator()() const
{
    size_t sum = 0;

    if (this->Policy == ThreadingPolicy::SingleThreaded) {
        // Evaluate aggregate function
        for (size_t cursor = 0; cursor < Params.NumOfItemsTupleIds; cursor++) {
            sum += itemTableColumnStore->I_PRICE[Params.ItemsTupleIds[cursor]];
        }
    } else {
        SUM_IN_PARALLEL(Params.NumOfItemsTupleIds,
                        Params.ItemsTupleIds,
                        itemTableColumnStore->I_PRICE,
                        local_sum += dataArray[indexArray[cursor]]; )
    }

    // Materialize
    ResultSet->CustomerId = Params.CustomerTupleId;
    ResultSet->TotalPrice = sum;
}


void QueryForDataQ1RowStore::operator()() const {

    size_t sum = 0;

    if (this->Policy == ThreadingPolicy::SingleThreaded) {
        // Evaluate aggregate function
        for (size_t cursor = 0; cursor < Params.NumOfItemsTupleIds; cursor++) {
            sum += itemTableColumnStore->Tuples[Params.ItemsTupleIds[cursor]].I_PRICE;
        }
    } else {
        SUM_IN_PARALLEL(Params.NumOfItemsTupleIds,
                        Params.ItemsTupleIds,
                        itemTableColumnStore->Tuples,
                        local_sum += dataArray[indexArray[cursor]].I_PRICE; )
    }

    // Materialize
    ResultSet->CustomerId = Params.CustomerTupleId;
    ResultSet->TotalPrice = sum;
}



void QueryForDataQ2ColumnStore::operator()() const
{
    ResultSet->ITEM_ID = Params.ItemTupleId;
    ResultSet->NumOfCustomerInfos = Params.NumOfCustomerTupleIds;
    ResultSet->CustomerInfos = (ResultSetQ2CustomerInfo *) malloc(sizeof(ResultSetQ2CustomerInfo) *
                                                                          Params.NumOfCustomerTupleIds);

    if (this->Policy == ThreadingPolicy::SingleThreaded) {
        // Materialization
        for (size_t cursor = 0; cursor < Params.NumOfCustomerTupleIds; cursor++) {
            INVOKE_DMS_SINGLE_TUPLE_MATERIALIZATION(cursor)
        }
   } else {
        MATERIALIZE_IN_PARALLEL(INVOKE_DMS_SINGLE_TUPLE_MATERIALIZATION(cursor))
    }
}

void QueryForDataQ2RowStore::operator()() const
{
    ResultSet->ITEM_ID = Params.ItemTupleId;
    ResultSet->NumOfCustomerInfos = Params.NumOfCustomerTupleIds;
    ResultSet->CustomerInfos = (ResultSetQ2CustomerInfo *) malloc(
            sizeof(ResultSetQ2CustomerInfo) * Params.NumOfCustomerTupleIds);

    if (this->Policy == ThreadingPolicy::SingleThreaded) {
        for (size_t cursor = 200; cursor < Params.NumOfCustomerTupleIds; cursor++) {
            INVOKE_NMS_SINGLE_TUPLE_MATERIALIZATION(cursor)
        }
    } else {
        MATERIALIZE_IN_PARALLEL(INVOKE_NMS_SINGLE_TUPLE_MATERIALIZATION(cursor))
    }
}
