#include "Common.h"

void CreateOrdersTables(CustomerTableDSM *Table, size_t NumberOfRecords)
{
    Table->C_ID = (uint64_t *) malloc(sizeof(uint64_t) * NumberOfRecords);
    Table->C_D_ID = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_W_ID = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_FIRST = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_MIDDLE = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_LAST = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_STREET_1 = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_STREET_2 = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_CITY = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_STATE = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_ZIP = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_PHONE = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_SINCE = (uint64_t *) malloc(sizeof(uint64_t) * NumberOfRecords);
    Table->C_CREDIT = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_CREDIT_LIM = (uint64_t *) malloc(sizeof(uint64_t) * NumberOfRecords);
    Table->C_DISCOUNT = (uint32_t *) malloc(sizeof(uint32_t) * NumberOfRecords);
    Table->C_BALANCE = (uint64_t *) malloc(sizeof(uint64_t) * NumberOfRecords);
    Table->C_YTD_PAYMENT = (unsigned short*) malloc(sizeof(unsigned short) * NumberOfRecords);
    Table->C_PAYMENT_CNT = (unsigned short *) malloc(sizeof(unsigned short) * NumberOfRecords);
    Table->C_DELIVERY_CNT = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->C_DATA = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
}

void CreateOrdersTables(CustomerTableNSM *Table, size_t NumberOfRecords)
{
    Table->Tuples = (CustomerTableNSM::Tuple *) malloc(sizeof(CustomerTableNSM::Tuple) * NumberOfRecords);
    Table->NumOfTuples = NumberOfRecords;
}

void CreateItemsTables(ItemTableDSM *Table, size_t NumberOfRecords)
{
    Table->I_ID = (uint64_t *) malloc(sizeof(uint64_t) * NumberOfRecords);
    Table->I_IM_ID = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->I_NAME = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
    Table->I_PRICE = (size_t *) malloc(sizeof(size_t) * NumberOfRecords);
    Table->I_DATA = (unsigned int *) malloc(sizeof(unsigned) * NumberOfRecords);
}

void CreateItemsTables(ItemTableNSM *Table, size_t NumberOfRecords)
{
    Table->Tuples = (ItemTableNSM::Tuple *) malloc(sizeof(ItemTableNSM::Tuple) * NumberOfRecords);
    Table->NumOfTuples = NumberOfRecords;
}

void FillOrdersTable(CustomerTableDSM *Table, size_t NumberOfRecords)
{
    for (size_t i = 0; i < NumberOfRecords; i++) {
        Table->C_ID[i] = i;
        Table->C_D_ID[i] = (unsigned) rand() % 10000000;
        Table->C_W_ID[i] = 1;
        Table->C_FIRST[i] = (unsigned) rand() % 997;
        Table->C_MIDDLE[i] = (unsigned) rand() % 997;
        Table->C_LAST[i] = (unsigned) rand() % 997;
        Table->C_STREET_1[i] = (unsigned) rand() % 997;
        Table->C_STREET_2[i] = (unsigned) rand() % 997;
        Table->C_CITY[i] = (unsigned) rand() % 997;
        Table->C_STATE[i] = (unsigned) rand() % 997;
        Table->C_ZIP[i] = (unsigned) rand() % 997;
        Table->C_PHONE[i] = (unsigned) rand() % 997;
        Table->C_SINCE[i] = (u_int64_t) rand() % 997;
        Table->C_CREDIT[i] = (unsigned) rand() % 997;
        Table->C_CREDIT_LIM[i] = (u_int64_t) rand() % 997;
        Table->C_DISCOUNT[i] = (unsigned) rand() % 997;
        Table->C_BALANCE[i] = (u_int64_t) rand() % 997;
        Table->C_YTD_PAYMENT[i] = (unsigned short) (rand() % 997);
        Table->C_PAYMENT_CNT[i] = (unsigned short) (rand() % 997);
        Table->C_DELIVERY_CNT[i] = (unsigned) rand() % 997;
        Table->C_DATA[i] = (unsigned) rand() % 997;
    }
}

void FillOrdersTable(CustomerTableNSM *Table, size_t NumberOfRecords)
{
    for (size_t i = 0; i < NumberOfRecords; i++) {
        auto tuple = &Table->Tuples[i];
        tuple->C_ID = i;
        tuple->C_D_ID = (unsigned) rand() % 10000000;
        tuple->C_W_ID = 1;
        tuple->C_FIRST = (unsigned) rand() % 997;
        tuple->C_MIDDLE = (unsigned) rand() % 997;
        tuple->C_LAST = (unsigned) rand() % 997;
        tuple->C_STREET_1 = (unsigned) rand() % 997;
        tuple->C_STREET_2 = (unsigned) rand() % 997;
        tuple->C_CITY = (unsigned) rand() % 997;
        tuple->C_STATE = (unsigned) rand() % 997;
        tuple->C_ZIP = (unsigned) rand() % 997;
        tuple->C_PHONE = (unsigned) rand() % 997;
        tuple->C_SINCE = (u_int64_t) rand() % 997;
        tuple->C_CREDIT = (unsigned) rand() % 997;
        tuple->C_CREDIT_LIM = (u_int64_t) rand() % 997;
        tuple->C_DISCOUNT = (unsigned) rand() % 997;
        tuple->C_BALANCE = (u_int64_t) rand() % 997;
        tuple->C_YTD_PAYMENT = (unsigned short) rand() % 997;
        tuple->C_PAYMENT_CNT = (unsigned short) rand() % 997;
        tuple->C_DELIVERY_CNT = (unsigned) rand() % 997;
        tuple->C_DATA = (unsigned) rand() % 997;
    }
}

void FillItemsTable(ItemTableDSM *Table, size_t NumberOfRecords)
{
    for (size_t i = 0; i < NumberOfRecords; i++) {
        Table->I_ID[i] = NumberOfRecords;
        Table->I_IM_ID[i] = rand() % 9973;
        Table->I_NAME[i] = rand() % 9973;
        Table->I_PRICE[i] = std::max(99, rand() % 9973);
        Table->I_DATA[i] = rand() % 9973;
    }
}

void FillItemsTable(ItemTableNSM *Table, size_t NumberOfRecords) {
    for (size_t i = 0; i < NumberOfRecords; i++) {
        auto tuple = &Table->Tuples[i];
        tuple->I_ID = NumberOfRecords;
        tuple->I_IM_ID = rand() % 9973;
        tuple->I_NAME = rand() % 9973;
        tuple->I_PRICE = std::max(99, rand() % 9973);
        tuple->I_DATA = rand() % 9973;
    }
}

void DisposeTables(CustomerTableDSM *CustomerTable, ItemTableDSM *ItemTable) {
    free(CustomerTable->C_ID);
    free(CustomerTable->C_D_ID);
    free(CustomerTable->C_W_ID);
    free(CustomerTable->C_FIRST);
    free(CustomerTable->C_MIDDLE);
    free(CustomerTable->C_LAST);
    free(CustomerTable->C_STREET_1);
    free(CustomerTable->C_STREET_2);
    free(CustomerTable->C_CITY);
    free(CustomerTable->C_STATE);
    free(CustomerTable->C_ZIP);
    free(CustomerTable->C_PHONE);
    free(CustomerTable->C_SINCE);
    free(CustomerTable->C_CREDIT);
    free(CustomerTable->C_CREDIT_LIM);
    free(CustomerTable->C_DISCOUNT);
    free(CustomerTable->C_BALANCE);
    free(CustomerTable->C_YTD_PAYMENT);
    free(CustomerTable->C_PAYMENT_CNT);
    free(CustomerTable->C_DELIVERY_CNT);
    free(CustomerTable->C_DATA);

    free(ItemTable->I_ID);
    free(ItemTable->I_IM_ID);
    free(ItemTable->I_NAME);
    free(ItemTable->I_PRICE);
    free(ItemTable->I_DATA);
}

void DisposeTables(CustomerTableNSM *CustomerTable, ItemTableNSM *ItemTable)
{
    free(CustomerTable->Tuples);
    free(ItemTable->Tuples);
}


QueryParamsQ1 CreateQueryParamsQ1(size_t NumberOfCustomers, size_t NumberOfItemsBoughtBySingeCustomer,
                                  size_t NumberOfItems)
{
    size_t recordId = rand() % NumberOfCustomers;
    QueryParamsQ1 returnValue;
    returnValue.CustomerTupleId = recordId;
    returnValue.ItemsTupleIds = (size_t *) malloc(sizeof(size_t) * NumberOfItemsBoughtBySingeCustomer);

    for (size_t i = 0; i < NumberOfItemsBoughtBySingeCustomer; i++) {
        returnValue.ItemsTupleIds[i] = rand() % NumberOfItems;
    }

    if (NumberOfItemsBoughtBySingeCustomer > NumberOfItems) {
        std::cerr << "ERROR: NumberOfItemsBoughtBySingeCustomer > NumberOfItems\n";
        exit(EXIT_FAILURE);
    }

    std::sort(returnValue.ItemsTupleIds, returnValue.ItemsTupleIds + NumberOfItemsBoughtBySingeCustomer);
    returnValue.NumOfItemsTupleIds = NumberOfItemsBoughtBySingeCustomer;
    return returnValue;
}

QueryParamsQ2 CreateQueryParamsQ2(size_t NumberOfItems, size_t NumberOfCustomersBoughtThatItem,
                                  size_t NumberOfCustomers)
{
    size_t recordId = rand() % NumberOfItems;
    QueryParamsQ2 returnValue;
    returnValue.ItemTupleId = recordId;
    returnValue.CustomerTupleIds = (size_t *) malloc(sizeof(size_t) * NumberOfCustomersBoughtThatItem);

    if (NumberOfCustomersBoughtThatItem > NumberOfCustomers) {
        std::cerr << "ERROR: NumberOfCustomersBoughtThatItem > NumberOfCustomers\n";
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NumberOfCustomersBoughtThatItem; i++) {
        returnValue.CustomerTupleIds[i] = rand() % NumberOfCustomers;
    }
    std::sort(returnValue.CustomerTupleIds, returnValue.CustomerTupleIds + NumberOfCustomersBoughtThatItem);
    returnValue.NumOfCustomerTupleIds = NumberOfCustomersBoughtThatItem;
    return returnValue;
}

void DisposeQueryParamsQ1(QueryParamsQ1 *Params) {
    free (Params->ItemsTupleIds);
}

void DisposeQueryParamsQ2(QueryParamsQ2 *Params) {
    free (Params->CustomerTupleIds);
}