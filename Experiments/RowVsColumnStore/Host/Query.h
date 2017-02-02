#pragma once

#include "../Shared/Common.h"
#include "../Configure.h"

struct QueryForDataQ1ColumnStore : public QueryForDataQ1
{
    ItemTableDSM *itemTableColumnStore;

    QueryForDataQ1ColumnStore(ResultSetQ1 *ResultSet, ItemTableDSM *itemTableColumnStore, QueryParamsQ1 Params,
                              ThreadingPolicy Policy):
            QueryForDataQ1(ResultSet, Params, Policy), itemTableColumnStore(itemTableColumnStore) { }

    void operator()() const;
};

struct QueryForDataQ1RowStore : public QueryForDataQ1
{
    ItemTableNSM *itemTableColumnStore;

    QueryForDataQ1RowStore(ResultSetQ1 *ResultSet, ItemTableNSM *itemTableColumnStore, QueryParamsQ1 Params,
                           ThreadingPolicy Policy):
            QueryForDataQ1(ResultSet, Params, Policy), itemTableColumnStore(itemTableColumnStore) { }

    void operator()() const;
};

struct QueryForDataQ2ColumnStore : public QueryForDataQ2 {

    CustomerTableDSM *customerTableColumnStore;

    QueryForDataQ2ColumnStore(ResultSetQ2 *ResultSet, CustomerTableDSM *customerTableColumnStore,
                              QueryParamsQ2 Params, ThreadingPolicy Policy):
            QueryForDataQ2(ResultSet, Params, Policy), customerTableColumnStore(customerTableColumnStore) { }

    void operator()() const;
};

struct QueryForDataQ2RowStore : public QueryForDataQ2 {

    CustomerTableNSM *customerTableColumnStore;

    QueryForDataQ2RowStore(ResultSetQ2 *ResultSet, CustomerTableNSM *customerTableColumnStore,
                           QueryParamsQ2 Params, ThreadingPolicy Policy):
            QueryForDataQ2(ResultSet, Params, Policy), customerTableColumnStore(customerTableColumnStore) { }

    void operator()() const;
};