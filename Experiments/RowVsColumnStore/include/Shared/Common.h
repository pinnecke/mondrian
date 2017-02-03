#pragma once

#include <cstdlib>
#include <utility>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>
#include <random>
#include <iostream>

struct CustomerTableDSM
{
    uint64_t *C_ID;
    unsigned *C_D_ID;
    unsigned *C_W_ID;
    unsigned *C_FIRST;
    unsigned *C_MIDDLE;
    unsigned *C_LAST;
    unsigned *C_STREET_1;
    unsigned *C_STREET_2;
    unsigned *C_CITY;
    unsigned *C_STATE;
    unsigned *C_ZIP;
    unsigned *C_PHONE;
    uint64_t *C_SINCE;
    unsigned *C_CREDIT;
    uint64_t *C_CREDIT_LIM;
    uint32_t *C_DISCOUNT;
    uint64_t *C_BALANCE;
    unsigned short *C_YTD_PAYMENT;
    unsigned short *C_PAYMENT_CNT;
    unsigned *C_DELIVERY_CNT;
    unsigned *C_DATA;
};

struct CustomerTableNSM
{
    struct Tuple
    {
        uint64_t C_ID;
        unsigned C_D_ID;
        unsigned C_W_ID;
        unsigned C_FIRST;
        unsigned C_MIDDLE;
        unsigned C_LAST;
        unsigned C_STREET_1;
        unsigned C_STREET_2;
        unsigned C_CITY;
        unsigned C_STATE;
        unsigned C_ZIP;
        unsigned C_PHONE;
        uint64_t C_SINCE;
        unsigned C_CREDIT;
        uint64_t C_CREDIT_LIM;
        uint32_t C_DISCOUNT;
        uint64_t C_BALANCE;
        unsigned short C_YTD_PAYMENT;
        unsigned short C_PAYMENT_CNT;
        unsigned C_DELIVERY_CNT;
        unsigned C_DATA;
    };

    Tuple *Tuples;
    size_t NumOfTuples;
};

struct ItemTableDSM
{
    uint64_t *I_ID;
    unsigned *I_IM_ID;
    unsigned *I_NAME;
    size_t *I_PRICE;
    unsigned *I_DATA;
};

struct ItemTableNSM
{
    struct Tuple
    {
        uint64_t I_ID;
        unsigned I_IM_ID;
        unsigned I_NAME;
        size_t I_PRICE;
        unsigned I_DATA;
    };

    Tuple *Tuples;
    size_t NumOfTuples;
};


struct QueryParamsQ1
{
    size_t CustomerTupleId;
    size_t *ItemsTupleIds;
    size_t NumOfItemsTupleIds;
};

struct QueryParamsQ2
{
    size_t ItemTupleId;
    size_t *CustomerTupleIds;
    size_t NumOfCustomerTupleIds;
};

struct ResultSetQ1
{
    unsigned CustomerId;
    u_int64_t TotalPrice;
};

struct ResultSetQ2CustomerInfo
{
    uint64_t C_ID;
    unsigned C_D_ID;
    unsigned C_W_ID;
    unsigned C_FIRST;
    unsigned C_MIDDLE;
    unsigned C_LAST;
    unsigned C_STREET_1;
    unsigned C_STREET_2;
    unsigned C_CITY;
    unsigned C_STATE;
    unsigned C_ZIP;
    unsigned C_PHONE;
    uint64_t C_SINCE;
    unsigned C_CREDIT;
    uint64_t C_CREDIT_LIM;
    uint32_t C_DISCOUNT;
    uint64_t C_BALANCE;
    short C_YTD_PAYMENT;
    short C_PAYMENT_CNT;
    unsigned C_DELIVERY_CNT;
    unsigned C_DATA;
};

struct ResultSetQ2
{
    unsigned ITEM_ID;
    ResultSetQ2CustomerInfo *CustomerInfos;
    u_int64_t NumOfCustomerInfos;
};

enum class ThreadingPolicy
{
    SingleThreaded,
    MultiThreaded
};

struct QueryForDataQ1
{
    ResultSetQ1 *ResultSet;
    QueryParamsQ1 Params;
    ThreadingPolicy Policy;

    QueryForDataQ1(ResultSetQ1 *ResultSet, QueryParamsQ1 Params, ThreadingPolicy ThreadingPolicy):
            ResultSet(ResultSet), Params(Params), Policy(ThreadingPolicy) { }
};


struct QueryForDataQ2
{
    ResultSetQ2 *ResultSet;
    QueryParamsQ2 Params;
    ThreadingPolicy Policy;

    QueryForDataQ2(ResultSetQ2 *ResultSet, QueryParamsQ2 Params, ThreadingPolicy ThreadingPolicy):
            ResultSet(ResultSet), Params(Params), Policy(ThreadingPolicy) { }
};


void CreateOrdersTables(CustomerTableDSM *Table, size_t NumberOfRecords);

void CreateOrdersTables(CustomerTableNSM *Table, size_t NumberOfRecords);

void CreateItemsTables(ItemTableDSM *Table, size_t NumberOfRecords);

void CreateItemsTables(ItemTableNSM *Table, size_t NumberOfRecords);

void FillOrdersTable(CustomerTableDSM *Table, size_t NumberOfRecords);

void FillOrdersTable(CustomerTableNSM *Table, size_t NumberOfRecords);

void FillItemsTable(ItemTableDSM *Table, size_t NumberOfRecords);

void FillItemsTable(ItemTableNSM *Table, size_t NumberOfRecords);

void DisposeTables(CustomerTableDSM *CustomerTable, ItemTableDSM *ItemTable);

void DisposeTables(CustomerTableNSM *CustomerTable, ItemTableNSM *ItemTable);

QueryParamsQ1 CreateQueryParamsQ1(size_t NumberOfCustomers, size_t NumberOfItemsBoughtBySingeCustomer, size_t NumberOfItems);

QueryParamsQ2 CreateQueryParamsQ2(size_t NumberOfItems, size_t NumberOfCustomersBoughtThatItem, size_t NumberOfCustomers);

void DisposeQueryParamsQ1(QueryParamsQ1 *Params);

void DisposeQueryParamsQ2(QueryParamsQ2 *Params);