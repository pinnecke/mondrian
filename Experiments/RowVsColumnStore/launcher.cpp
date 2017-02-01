

#include <utility>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>
#include <random>
#include <iostream>

#ifndef NUMBER_OF_THREADS
#define NUMBER_OF_THREADS 8
#endif

/* Taken from http://stackoverflow.com/questions/2808398/easily-measure-elapsed-time */
template<typename TimeT = std::chrono::nanoseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT>
                (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

struct TcpCCustomerTableColumnStore {
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
    short *C_YTD_PAYMENT;
    short *C_PAYMENT_CNT;
    unsigned *C_DELIVERY_CNT;
    unsigned *C_DATA;
};

struct TcpCCustomerTableRowStore {

    struct Tuple {
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

    Tuple *tuples;
    size_t numOfTuples;
};

struct TcpCItemTableColumnStore {
    uint64_t *I_ID;
    unsigned *I_IM_ID;
    unsigned *I_NAME;
    size_t *I_PRICE;
    unsigned *I_DATA;
};

struct TcpCItemTableRowStore {

    struct Tuple {
        uint64_t I_ID;
        unsigned I_IM_ID;
        unsigned I_NAME;
        size_t I_PRICE;
        unsigned I_DATA;
    };

    Tuple *tuples;
    size_t numOfTuples;
};


void CreateOrdersTables(TcpCCustomerTableColumnStore *pTable, size_t i)
{
    pTable->C_ID = (uint64_t *) malloc(sizeof(uint64_t) * i);
    pTable->C_D_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_W_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_FIRST = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_MIDDLE = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_LAST = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_STREET_1 = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_STREET_2 = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_CITY = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_STATE = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_ZIP = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_PHONE = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_SINCE = (uint64_t *) malloc(sizeof(uint64_t) * i);
    pTable->C_CREDIT = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_CREDIT_LIM = (uint64_t *) malloc(sizeof(uint64_t) * i);
    pTable->C_DISCOUNT = (uint32_t *) malloc(sizeof(uint32_t) * i);
    pTable->C_BALANCE = (uint64_t *) malloc(sizeof(uint64_t) * i);
    pTable->C_YTD_PAYMENT = (short*) malloc(sizeof(short) * i);
    pTable->C_PAYMENT_CNT = (short *) malloc(sizeof(short) * i);
    pTable->C_DELIVERY_CNT = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->C_DATA = (unsigned int *) malloc(sizeof(unsigned) * i);
}

void CreateOrdersTables(TcpCCustomerTableRowStore *pTable, size_t i)
{
    pTable->tuples = (TcpCCustomerTableRowStore::Tuple *) malloc(sizeof(TcpCCustomerTableRowStore::Tuple) * i);
    pTable->numOfTuples = i;
}

void CreateItemsTables(TcpCItemTableColumnStore *pTable, size_t i)
{
    pTable->I_ID = (uint64_t *) malloc(sizeof(uint64_t) * i);
    pTable->I_IM_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->I_NAME = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->I_PRICE = (size_t *) malloc(sizeof(size_t) * i);
    pTable->I_DATA = (unsigned int *) malloc(sizeof(unsigned) * i);
}

void CreateItemsTables(TcpCItemTableRowStore *pTable, size_t i)
{
    pTable->tuples = (TcpCItemTableRowStore::Tuple *) malloc(sizeof(TcpCItemTableRowStore::Tuple) * i);
    pTable->numOfTuples = i;
}

void FillOrdersTable(TcpCCustomerTableColumnStore *pTable, size_t NumRecords) {
    for (size_t i = 0; i < NumRecords; i++) {
        pTable->C_ID[i] = i;
        pTable->C_D_ID[i] = rand() % 10000000;
        pTable->C_W_ID[i] = 1;
        pTable->C_FIRST[i] = rand() % 997;
        pTable->C_MIDDLE[i] = rand() % 997;
        pTable->C_LAST[i] = rand() % 997;
        pTable->C_STREET_1[i] = rand() % 997;
        pTable->C_STREET_2[i] = rand() % 997;
        pTable->C_CITY[i] = rand() % 997;
        pTable->C_STATE[i] = rand() % 997;
        pTable->C_ZIP[i] = rand() % 997;
        pTable->C_PHONE[i] = rand() % 997;
        pTable->C_SINCE[i] = rand() % 997;
        pTable->C_CREDIT[i] = rand() % 997;
        pTable->C_CREDIT_LIM[i] = rand() % 997;
        pTable->C_DISCOUNT[i] = rand() % 997;
        pTable->C_BALANCE[i] = rand() % 997;
        pTable->C_YTD_PAYMENT[i] = rand() % 997;
        pTable->C_PAYMENT_CNT[i] = rand() % 997;
        pTable->C_DELIVERY_CNT[i] = rand() % 997;
        pTable->C_DATA[i] = rand() % 997;
    }
}

void FillOrdersTable(TcpCCustomerTableRowStore *pTable, size_t NumRecords) {
    for (size_t i = 0; i < NumRecords; i++) {
        auto tuple = &pTable->tuples[i];
        tuple->C_ID = i;
        tuple->C_D_ID = rand() % 10000000;
        tuple->C_W_ID = 1;
        tuple->C_FIRST = rand() % 997;
        tuple->C_MIDDLE = rand() % 997;
        tuple->C_LAST = rand() % 997;
        tuple->C_STREET_1 = rand() % 997;
        tuple->C_STREET_2 = rand() % 997;
        tuple->C_CITY = rand() % 997;
        tuple->C_STATE = rand() % 997;
        tuple->C_ZIP = rand() % 997;
        tuple->C_PHONE = rand() % 997;
        tuple->C_SINCE = rand() % 997;
        tuple->C_CREDIT = rand() % 997;
        tuple->C_CREDIT_LIM = rand() % 997;
        tuple->C_DISCOUNT = rand() % 997;
        tuple->C_BALANCE = rand() % 997;
        tuple->C_YTD_PAYMENT = rand() % 997;
        tuple->C_PAYMENT_CNT = rand() % 997;
        tuple->C_DELIVERY_CNT = rand() % 997;
        tuple->C_DATA = rand() % 997;
    }
}

void FillItemsTable(TcpCItemTableColumnStore *pTable, size_t NumRecordsInItems) {
    for (size_t i = 0; i < NumRecordsInItems; i++) {
        pTable->I_ID[i] = NumRecordsInItems;
        pTable->I_IM_ID[i] = rand() % 9973;
        pTable->I_NAME[i] = rand() % 9973;
        pTable->I_PRICE[i] = std::max(99, rand() % 9973); /*100.00 Coins maximum price, 0.99 Coins minimum price*/
        pTable->I_DATA[i] = rand() % 9973;
    }
}

void FillItemsTable(TcpCItemTableRowStore *pTable, size_t NumRecordsInItems) {
    for (size_t i = 0; i < NumRecordsInItems; i++) {
        auto tuple = &pTable->tuples[i];
        tuple->I_ID = NumRecordsInItems;
        tuple->I_IM_ID = rand() % 9973;
        tuple->I_NAME = rand() % 9973;
        tuple->I_PRICE = std::max(99, rand() % 9973); /*100.00 Coins maximum price, 0.99 Coins minimum price*/
        tuple->I_DATA = rand() % 9973;
    }
}

void DisposeTables(TcpCCustomerTableColumnStore *pTable1, TcpCItemTableColumnStore *pTable2) {
    free(pTable1->C_ID);
    free(pTable1->C_D_ID);
    free(pTable1->C_W_ID);
    free(pTable1->C_FIRST);
    free(pTable1->C_MIDDLE);
    free(pTable1->C_LAST);
    free(pTable1->C_STREET_1);
    free(pTable1->C_STREET_2);
    free(pTable1->C_CITY);
    free(pTable1->C_STATE);
    free(pTable1->C_ZIP);
    free(pTable1->C_PHONE);
    free(pTable1->C_SINCE);
    free(pTable1->C_CREDIT);
    free(pTable1->C_CREDIT_LIM);
    free(pTable1->C_DISCOUNT);
    free(pTable1->C_BALANCE);
    free(pTable1->C_YTD_PAYMENT);
    free(pTable1->C_PAYMENT_CNT);
    free(pTable1->C_DELIVERY_CNT);
    free(pTable1->C_DATA);

    free(pTable2->I_ID);
    free(pTable2->I_IM_ID);
    free(pTable2->I_NAME);
    free(pTable2->I_PRICE);
    free(pTable2->I_DATA);
}

void DisposeTables(TcpCCustomerTableRowStore *pTable1, TcpCItemTableRowStore *pTable2) {
    free(pTable1->tuples);
    free(pTable2->tuples);
}


struct QueryParamsQ1 {
    size_t TUPLE_ID_CUSTOMER;
    size_t *TUPLE_IDS_ITEMS;
    size_t NUM_TUPLE_IDS_ITEMS;
};

struct QueryParamsQ2 {
    size_t TUPLE_ID_ITEM;
    size_t *TUPLE_IDS_CUSTOMERS;
    size_t NUM_TUPLE_IDS_CUSTOMERS;
};

struct ResultSetQ1 {
    unsigned RETURN_CUSTOMER_ID;
    u_int64_t RETURN_TOTAL_PRICE;
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

struct ResultSetQ2 {
    unsigned RETURN_ITEM_ID;
    ResultSetQ2CustomerInfo *RETURN_CUSTOMER_INFO;
    u_int64_t NUM_RETURN_CUSTOMER_INFO;
};

QueryParamsQ1 CreateQueryParamsQ1(size_t NumberOfCustomers, size_t NumberOfItemsBoughtBySingeCustomer, size_t NumberOfItems) {
    size_t recordId = rand() % NumberOfCustomers;
    QueryParamsQ1 returnValue;
    returnValue.TUPLE_ID_CUSTOMER = recordId;
    returnValue.TUPLE_IDS_ITEMS = (size_t *) malloc(sizeof(size_t) * NumberOfItemsBoughtBySingeCustomer);
    for (size_t i = 0; i < NumberOfItemsBoughtBySingeCustomer; i++) {
        returnValue.TUPLE_IDS_ITEMS[i] = rand() % NumberOfItems;
    }

    if (NumberOfItemsBoughtBySingeCustomer > NumberOfItems) {
        std::cerr << "IMPOSSIBLE, too!\n";
        exit(EXIT_FAILURE);
    }

    std::sort(returnValue.TUPLE_IDS_ITEMS, returnValue.TUPLE_IDS_ITEMS + NumberOfItemsBoughtBySingeCustomer);
    returnValue.NUM_TUPLE_IDS_ITEMS = NumberOfItemsBoughtBySingeCustomer;
    return returnValue;
}

QueryParamsQ2 CreateQueryParamsQ2(size_t NumberOfItems, size_t NumberOfCustomersBoughtThatItem, size_t NumberOfCustomers) {
    size_t recordId = rand() % NumberOfItems;
    QueryParamsQ2 returnValue;
    returnValue.TUPLE_ID_ITEM = recordId;
    returnValue.TUPLE_IDS_CUSTOMERS = (size_t *) malloc(sizeof(size_t) * NumberOfCustomersBoughtThatItem);

    /*std::vector<size_t> allCustomerNumbers;
    allCustomerNumbers.reserve(NumberOfCustomers);
    for (size_t i = 0; i < NumberOfCustomers; ++i) {
        allCustomerNumbers[i];
    }
    for (size_t i = 0; i < 2*NumberOfCustomers; ++i) {
        auto lhs = rand() % NumberOfCustomers;
        auto rhs = rand() % NumberOfCustomers;
        size_t temp = allCustomerNumbers[lhs];
        allCustomerNumbers[lhs] = rhs;
        allCustomerNumbers[rhs] = temp;
    }

    if (NumberOfCustomersBoughtThatItem > NumberOfCustomers) {
        std::cerr << "IMPOSSIBLE!\n";
        exit(EXIT_FAILURE);
    }*/

    if (NumberOfCustomersBoughtThatItem > NumberOfCustomers) {
        std::cerr << "IMPOSSIBLE!\n";
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NumberOfCustomersBoughtThatItem; i++) {
        returnValue.TUPLE_IDS_CUSTOMERS[i] = rand() % NumberOfCustomers;
    }
    std::sort(returnValue.TUPLE_IDS_CUSTOMERS, returnValue.TUPLE_IDS_CUSTOMERS + NumberOfCustomersBoughtThatItem);
    returnValue.NUM_TUPLE_IDS_CUSTOMERS = NumberOfCustomersBoughtThatItem;
    return returnValue;
}

void DisposeQueryParamsQ1(QueryParamsQ1 *paramsQ1) {
    free (paramsQ1->TUPLE_IDS_ITEMS);
}

void DisposeQueryParamsQ2(QueryParamsQ2 *paramsQ2) {
    free (paramsQ2->TUPLE_IDS_CUSTOMERS);
}

struct QueryForDataQ1ColumnStore {

    ResultSetQ1 *out;
    TcpCItemTableColumnStore *itemTableColumnStore;
    QueryParamsQ1 params;

    QueryForDataQ1ColumnStore(ResultSetQ1 *out, TcpCItemTableColumnStore *itemTableColumnStore, QueryParamsQ1 params) {
        this->out = out;
        this->itemTableColumnStore = itemTableColumnStore;
        this->params = params;
    }

    void operator()() const {

    /*    size_t sum = 0;


        // Evaluate aggregate function
        for (size_t cursor = 0; cursor < params.NUM_TUPLE_IDS_ITEMS; cursor++) {
            sum += itemTableColumnStore->I_PRICE[params.TUPLE_IDS_ITEMS[cursor]];
        }

*/


        size_t *localResults = (size_t *) malloc(sizeof(size_t) * NUMBER_OF_THREADS);
        for(unsigned threadId = 0; threadId < NUMBER_OF_THREADS; ++threadId) {
            localResults[threadId] = 0;
        }

        size_t arrayNumberOfElements = params.NUM_TUPLE_IDS_ITEMS;
        size_t *indexArray = params.TUPLE_IDS_ITEMS;
        size_t *dataArray = itemTableColumnStore->I_PRICE;
        unsigned numberOfThreadsInUse = NUMBER_OF_THREADS;

        std::vector<std::thread> threads(numberOfThreadsInUse);


     //   printf("\narrayNumberOfElements = %zu, numberOfThreadsInUse = %zu\n", arrayNumberOfElements, numberOfThreadsInUse);

        for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {
            threads[threadId] = std::thread(
                    [threadId, &localResults, &arrayNumberOfElements, &indexArray, &dataArray, &numberOfThreadsInUse]() -> void {
                        size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));

                        size_t local_sum = 0;
                        auto arrayMaxIdx = arrayNumberOfElements;
                        auto threadMaxIdx = chunkSize * threadId + chunkSize;
                        auto end = std::min(threadMaxIdx, arrayNumberOfElements);

                        for (size_t cursor = chunkSize * threadId; cursor < end; cursor++) {
                            local_sum += dataArray[indexArray[cursor]];
                        }

                        localResults[threadId] = local_sum;
                    });
        }
        for (auto &thread : threads)
            thread.join();


        // printf("\nlocal results: ");
        size_t msum = 0;
        for (size_t i = 0; i < numberOfThreadsInUse; i++) {
     //       printf("%zu=%zu, ", i, localResults[i]);
            msum += localResults[i];
        }

        free(localResults);
     //   printf("\ncompare single theraded= %zu VS multi threaded= %zu\n", sum, msum);














        // Materialize
        out->RETURN_CUSTOMER_ID = params.TUPLE_ID_CUSTOMER;
        out->RETURN_TOTAL_PRICE = msum;

    }
};

struct QueryForDataQ1RowStore {

    ResultSetQ1 *out;
    TcpCItemTableRowStore *itemTableColumnStore;
    QueryParamsQ1 params;

    QueryForDataQ1RowStore(ResultSetQ1 *out, TcpCItemTableRowStore *itemTableColumnStore, QueryParamsQ1 params) {
        this->out = out;
        this->itemTableColumnStore = itemTableColumnStore;
        this->params = params;
    }

    void operator()() const {

        size_t sum = 0;


        // Evaluate aggregate function
        for (size_t cursor = 0; cursor < params.NUM_TUPLE_IDS_ITEMS; cursor++) {
            sum += itemTableColumnStore->tuples[params.TUPLE_IDS_ITEMS[cursor]].I_PRICE;
        }








        size_t *localResults = (size_t *) malloc(sizeof(size_t) * NUMBER_OF_THREADS);
        for(unsigned threadId = 0; threadId < NUMBER_OF_THREADS; ++threadId) {
            localResults[threadId] = 0;
        }

        size_t arrayNumberOfElements = params.NUM_TUPLE_IDS_ITEMS;
        size_t *indexArray = params.TUPLE_IDS_ITEMS;
        TcpCItemTableRowStore::Tuple *dataArray = itemTableColumnStore->tuples;
        unsigned numberOfThreadsInUse = NUMBER_OF_THREADS;

        std::vector<std::thread> threads(numberOfThreadsInUse);


        //   printf("\narrayNumberOfElements = %zu, numberOfThreadsInUse = %zu\n", arrayNumberOfElements, numberOfThreadsInUse);

        for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {
            threads[threadId] = std::thread(
                    [threadId, &localResults, &arrayNumberOfElements, &indexArray, &dataArray, &numberOfThreadsInUse]() -> void {
                        size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));

                        size_t local_sum = 0;
                        auto arrayMaxIdx = arrayNumberOfElements;
                        auto threadMaxIdx = chunkSize * threadId + chunkSize;
                        auto end = std::min(threadMaxIdx, arrayNumberOfElements);

                        for (size_t cursor = chunkSize * threadId; cursor < end; cursor++) {
                            local_sum += dataArray[indexArray[cursor]].I_PRICE;
                        }

                        localResults[threadId] = local_sum;
                    });
        }
        for (auto &thread : threads)
            thread.join();


        // printf("\nlocal results: ");
        size_t msum = 0;
        for (size_t i = 0; i < numberOfThreadsInUse; i++) {
            //       printf("%zu=%zu, ", i, localResults[i]);
            msum += localResults[i];
        }

        free(localResults);

        // Materialize
        out->RETURN_CUSTOMER_ID = params.TUPLE_ID_CUSTOMER;
        out->RETURN_TOTAL_PRICE = msum;

    }
};

struct QueryForDataQ2ColumnStore {

    ResultSetQ2 *out;
    TcpCCustomerTableColumnStore *customerTableColumnStore;
    QueryParamsQ2 params;

    QueryForDataQ2ColumnStore(ResultSetQ2 *out, TcpCCustomerTableColumnStore *customerTableColumnStore,
                   QueryParamsQ2 params) {
        this->out = out;
        this->customerTableColumnStore = customerTableColumnStore;
        this->params = params;
    }

    void operator()() const {

        size_t sum = 0;

        out->RETURN_ITEM_ID = params.TUPLE_ID_ITEM;
        out->NUM_RETURN_CUSTOMER_INFO = params.NUM_TUPLE_IDS_CUSTOMERS;
        out->RETURN_CUSTOMER_INFO = (ResultSetQ2CustomerInfo *) malloc(sizeof(ResultSetQ2CustomerInfo) * params.NUM_TUPLE_IDS_CUSTOMERS);


        // Materialization
      /*  for (size_t cursor = 0; cursor < params.NUM_TUPLE_IDS_CUSTOMERS; cursor++) {
            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
            info->C_ID= customerTableColumnStore->C_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_D_ID= customerTableColumnStore->C_D_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_W_ID= customerTableColumnStore->C_W_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_FIRST= customerTableColumnStore->C_FIRST[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_MIDDLE= customerTableColumnStore->C_MIDDLE[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_LAST= customerTableColumnStore->C_LAST[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_STREET_1= customerTableColumnStore->C_STREET_1[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_STREET_2= customerTableColumnStore->C_STREET_2[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_CITY= customerTableColumnStore->C_CITY[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_STATE= customerTableColumnStore->C_STATE[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_ZIP= customerTableColumnStore->C_ZIP[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_PHONE= customerTableColumnStore->C_PHONE[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_SINCE= customerTableColumnStore->C_SINCE[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_CREDIT= customerTableColumnStore->C_CREDIT[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_CREDIT_LIM= customerTableColumnStore->C_CREDIT_LIM[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_DISCOUNT= customerTableColumnStore->C_DISCOUNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_BALANCE= customerTableColumnStore->C_BALANCE[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_YTD_PAYMENT= customerTableColumnStore->C_YTD_PAYMENT[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_PAYMENT_CNT= customerTableColumnStore->C_PAYMENT_CNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_DELIVERY_CNT= customerTableColumnStore->C_DELIVERY_CNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
            info->C_DATA= customerTableColumnStore->C_DATA[params.TUPLE_IDS_CUSTOMERS[cursor]];
        }

        for (size_t cursor = 0; cursor < 5; cursor++) {
            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
            printf("\nSINGLE RECORD %zu, C_FIRST %zu, C_STREET_1 %zu, C_PAYMENT_CNT %zu\n", cursor, info->C_FIRST, info->C_STREET_1, info->C_PAYMENT_CNT);
        }*/


        size_t arrayNumberOfElements = params.NUM_TUPLE_IDS_CUSTOMERS;
        size_t *dataArray = params.TUPLE_IDS_CUSTOMERS;
        unsigned numberOfThreadsInUse = NUMBER_OF_THREADS;

        std::vector<std::thread> threads(numberOfThreadsInUse);


        //   printf("\narrayNumberOfElements = %zu, numberOfThreadsInUse = %zu\n", arrayNumberOfElements, numberOfThreadsInUse);

        for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {
            threads[threadId] = std::thread(
                    [threadId, &arrayNumberOfElements, &dataArray, &numberOfThreadsInUse, this]() -> void {
                        size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));
                        auto arrayMaxIdx = arrayNumberOfElements;
                        auto threadMaxIdx = chunkSize * threadId + chunkSize;
                        auto end = std::min(threadMaxIdx, arrayNumberOfElements);

                        for (size_t cursor = chunkSize * threadId; cursor < end; cursor++) {
                            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
                            info->C_ID= customerTableColumnStore->C_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_D_ID= customerTableColumnStore->C_D_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_W_ID= customerTableColumnStore->C_W_ID[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_FIRST= customerTableColumnStore->C_FIRST[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_MIDDLE= customerTableColumnStore->C_MIDDLE[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_LAST= customerTableColumnStore->C_LAST[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_STREET_1= customerTableColumnStore->C_STREET_1[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_STREET_2= customerTableColumnStore->C_STREET_2[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_CITY= customerTableColumnStore->C_CITY[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_STATE= customerTableColumnStore->C_STATE[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_ZIP= customerTableColumnStore->C_ZIP[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_PHONE= customerTableColumnStore->C_PHONE[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_SINCE= customerTableColumnStore->C_SINCE[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_CREDIT= customerTableColumnStore->C_CREDIT[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_CREDIT_LIM= customerTableColumnStore->C_CREDIT_LIM[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_DISCOUNT= customerTableColumnStore->C_DISCOUNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_BALANCE= customerTableColumnStore->C_BALANCE[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_YTD_PAYMENT= customerTableColumnStore->C_YTD_PAYMENT[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_PAYMENT_CNT= customerTableColumnStore->C_PAYMENT_CNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_DELIVERY_CNT= customerTableColumnStore->C_DELIVERY_CNT[params.TUPLE_IDS_CUSTOMERS[cursor]];
                            info->C_DATA= customerTableColumnStore->C_DATA[params.TUPLE_IDS_CUSTOMERS[cursor]];
                        }

                    });
        }
        for (auto &thread : threads)
            thread.join();



     /*   for (size_t cursor = 0; cursor < 5; cursor++) {
            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
            printf("\nMULTI RECORD %zu, C_FIRST %zu, C_STREET_1 %zu, C_PAYMENT_CNT %zu\n", cursor, info->C_FIRST, info->C_STREET_1, info->C_PAYMENT_CNT);
        }
*/


    }
};

struct QueryForDataQ2RowStore {

    ResultSetQ2 *out;
    TcpCCustomerTableRowStore *customerTableColumnStore;
    QueryParamsQ2 params;

    QueryForDataQ2RowStore(ResultSetQ2 *out, TcpCCustomerTableRowStore *customerTableColumnStore,
                              QueryParamsQ2 params) {
        this->out = out;
        this->customerTableColumnStore = customerTableColumnStore;
        this->params = params;
    }

    void operator()() const {

       size_t sum = 0;

        out->RETURN_ITEM_ID = params.TUPLE_ID_ITEM;
        out->NUM_RETURN_CUSTOMER_INFO = params.NUM_TUPLE_IDS_CUSTOMERS;
        out->RETURN_CUSTOMER_INFO = (ResultSetQ2CustomerInfo *) malloc(sizeof(ResultSetQ2CustomerInfo) * params.NUM_TUPLE_IDS_CUSTOMERS);

/*
        // Materialization
        for (size_t cursor = 200; cursor < params.NUM_TUPLE_IDS_CUSTOMERS; cursor++) {
            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];

            info->C_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_ID;
            info->C_D_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_D_ID;
            info->C_W_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_W_ID;
            info->C_FIRST = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_FIRST;
            info->C_MIDDLE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_MIDDLE;
            info->C_LAST = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_LAST;
            info->C_STREET_1 = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STREET_1;
            info->C_STREET_2 = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STREET_2;
            info->C_CITY = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CITY;
            info->C_STATE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STATE;
            info->C_ZIP = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_ZIP;
            info->C_PHONE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_PHONE;
            info->C_SINCE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_SINCE;
            info->C_CREDIT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CREDIT;
            info->C_CREDIT_LIM = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CREDIT_LIM;
            info->C_DISCOUNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DISCOUNT;
            info->C_BALANCE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_BALANCE;
            info->C_YTD_PAYMENT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_YTD_PAYMENT;
            info->C_PAYMENT_CNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_PAYMENT_CNT;
            info->C_DELIVERY_CNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DELIVERY_CNT;
            info->C_DATA = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DATA;
        }
        */


        size_t arrayNumberOfElements = params.NUM_TUPLE_IDS_CUSTOMERS;
        size_t *dataArray = params.TUPLE_IDS_CUSTOMERS;
        unsigned numberOfThreadsInUse = NUMBER_OF_THREADS;

        std::vector<std::thread> threads(numberOfThreadsInUse);


        //   printf("\narrayNumberOfElements = %zu, numberOfThreadsInUse = %zu\n", arrayNumberOfElements, numberOfThreadsInUse);

        for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {
            threads[threadId] = std::thread(
                    [threadId, &arrayNumberOfElements, &dataArray, &numberOfThreadsInUse, this]() -> void {
                        size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));
                        auto arrayMaxIdx = arrayNumberOfElements;
                        auto threadMaxIdx = chunkSize * threadId + chunkSize;
                        auto end = std::min(threadMaxIdx, arrayNumberOfElements);

                        for (size_t cursor = chunkSize * threadId; cursor < end; cursor++) {
                            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
                            info->C_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_ID;
                            info->C_D_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_D_ID;
                            info->C_W_ID = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_W_ID;
                            info->C_FIRST = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_FIRST;
                            info->C_MIDDLE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_MIDDLE;
                            info->C_LAST = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_LAST;
                            info->C_STREET_1 = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STREET_1;
                            info->C_STREET_2 = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STREET_2;
                            info->C_CITY = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CITY;
                            info->C_STATE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_STATE;
                            info->C_ZIP = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_ZIP;
                            info->C_PHONE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_PHONE;
                            info->C_SINCE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_SINCE;
                            info->C_CREDIT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CREDIT;
                            info->C_CREDIT_LIM = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_CREDIT_LIM;
                            info->C_DISCOUNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DISCOUNT;
                            info->C_BALANCE = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_BALANCE;
                            info->C_YTD_PAYMENT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_YTD_PAYMENT;
                            info->C_PAYMENT_CNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_PAYMENT_CNT;
                            info->C_DELIVERY_CNT = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DELIVERY_CNT;
                            info->C_DATA = customerTableColumnStore->tuples[params.TUPLE_IDS_CUSTOMERS[cursor]].C_DATA;
                        }

                    });
        }
        for (auto &thread : threads)
            thread.join();

       /* for (size_t cursor = 0; cursor < 5; cursor++) {
            ResultSetQ2CustomerInfo *info = &out->RETURN_CUSTOMER_INFO[cursor];
            printf("\nMULTI RECORD %zu, C_FIRST %zu, C_STREET_1 %zu, C_PAYMENT_CNT %zu\n", cursor, info->C_FIRST, info->C_STREET_1, info->C_PAYMENT_CNT);
        }*/

    }
};

const size_t ITEMS_BOUGTH_BY_CUSTOMER_AVG = 150;
const size_t BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS = 40000;

void SampleColumnStore(size_t N, size_t NUMBER_OF_ITEMS, size_t currentRepetition) {
    /* Setup */
    TcpCCustomerTableColumnStore customerTableColumnStore;
    TcpCItemTableColumnStore itemTableColumnStore;
    srand(0);
    CreateOrdersTables(&customerTableColumnStore, N);
    CreateItemsTables(&itemTableColumnStore, NUMBER_OF_ITEMS);
    FillOrdersTable(&customerTableColumnStore, N);
    FillItemsTable(&itemTableColumnStore, NUMBER_OF_ITEMS);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(100000 ,100);




    /* Query Q1 */
    ResultSetQ1 result1;
    size_t numberOfItemsBougthByCustomer = ITEMS_BOUGTH_BY_CUSTOMER_AVG; // <-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    auto queryParamsQ1 = CreateQueryParamsQ1(N, numberOfItemsBougthByCustomer,
                                             NUMBER_OF_ITEMS);
    auto durationQ1 = measure<>::execution(QueryForDataQ1ColumnStore(&result1, &itemTableColumnStore, queryParamsQ1));
    DisposeQueryParamsQ1(&queryParamsQ1);

    /* Query Q2 */
    ResultSetQ2 result2;
    size_t numberOfCustomersBougthThatItem = BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS; // <-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    auto queryParamsQ2 = CreateQueryParamsQ2(NUMBER_OF_ITEMS, numberOfCustomersBougthThatItem, N);
    auto durationQ2 = measure<>::execution(QueryForDataQ2ColumnStore(&result2, &customerTableColumnStore, queryParamsQ2));
    DisposeQueryParamsQ2(&queryParamsQ2);

    /* Query Q3 */
    ResultSetQ1 result3;
    auto queryParamsQ3 = CreateQueryParamsQ1(N, NUMBER_OF_ITEMS,
                                             NUMBER_OF_ITEMS);
    auto durationQ3 = measure<>::execution(QueryForDataQ1ColumnStore(&result3, &itemTableColumnStore, queryParamsQ3));
    DisposeQueryParamsQ1(&queryParamsQ3);

    size_t dataSetSizeCustomersInByte = N * (sizeof(uint64_t) +
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


    size_t dataSetSizeItemsInByte = NUMBER_OF_ITEMS * (sizeof(uint64_t) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned));

    size_t dataSetSizeJoinTableInByte = sizeof(size_t) * queryParamsQ1.NUM_TUPLE_IDS_ITEMS + sizeof(size_t);
    size_t numberOfRecordsToProcess = queryParamsQ1.NUM_TUPLE_IDS_ITEMS;

    printf("%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;ColumnStore\n",
           (size_t)time(NULL),
           (size_t)N,
           (size_t)currentRepetition,
           (size_t)durationQ1,
           (size_t)durationQ2,
           (size_t)durationQ3,
           dataSetSizeCustomersInByte,
           dataSetSizeItemsInByte,
           dataSetSizeJoinTableInByte,
           numberOfRecordsToProcess);
    fflush(stdout);

    /* Cleanup */
    free(result2.RETURN_CUSTOMER_INFO);
    DisposeTables(&customerTableColumnStore, &itemTableColumnStore);

}

void SampleRowStore(size_t N, size_t NUMBER_OF_ITEMS, size_t currentRepetition) {
    /* Setup */
    TcpCCustomerTableRowStore customerTableColumnStore;
    TcpCItemTableRowStore itemTableColumnStore;
    srand(0);
    CreateOrdersTables(&customerTableColumnStore, N);
    CreateItemsTables(&itemTableColumnStore, NUMBER_OF_ITEMS);
    FillOrdersTable(&customerTableColumnStore, N);
    FillItemsTable(&itemTableColumnStore, NUMBER_OF_ITEMS);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(100000 ,100);




    /* Query Q1 */
    ResultSetQ1 result1;
    size_t numberOfItemsBougthByCustomer = ITEMS_BOUGTH_BY_CUSTOMER_AVG; // <-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    auto queryParamsQ1 = CreateQueryParamsQ1(N, numberOfItemsBougthByCustomer,
                                             NUMBER_OF_ITEMS);
    auto durationQ1 = measure<>::execution(QueryForDataQ1RowStore(&result1, &itemTableColumnStore, queryParamsQ1));
    DisposeQueryParamsQ1(&queryParamsQ1);

    /* Query Q2 */
    ResultSetQ2 result2;
    size_t numberOfCustomersBougthThatItem = BESTSELLING_ITEM_NUMBER_OF_CUSTOMERS; // <-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    auto queryParamsQ2 = CreateQueryParamsQ2(NUMBER_OF_ITEMS, numberOfCustomersBougthThatItem, N);
    auto durationQ2 = measure<>::execution(QueryForDataQ2RowStore(&result2, &customerTableColumnStore, queryParamsQ2));
    DisposeQueryParamsQ2(&queryParamsQ2);

    /* Query Q3 */
    ResultSetQ1 result3;
    auto queryParamsQ3 = CreateQueryParamsQ1(N, NUMBER_OF_ITEMS,
                                             NUMBER_OF_ITEMS);
    auto durationQ3 = measure<>::execution(QueryForDataQ1RowStore(&result3, &itemTableColumnStore, queryParamsQ3));
    DisposeQueryParamsQ1(&queryParamsQ3);

    size_t dataSetSizeCustomersInByte = N * (sizeof(uint64_t) +
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


    size_t dataSetSizeItemsInByte = NUMBER_OF_ITEMS * (sizeof(uint64_t) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned) +
                                                       sizeof(unsigned));

    size_t dataSetSizeJoinTableInByte = sizeof(size_t) * queryParamsQ1.NUM_TUPLE_IDS_ITEMS + sizeof(size_t);
    size_t numberOfRecordsToProcess = queryParamsQ1.NUM_TUPLE_IDS_ITEMS;

    printf("%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%zu;RowStore\n",
           (size_t)time(NULL),
           (size_t)N,
           (size_t)currentRepetition,
           (size_t)durationQ1,
           (size_t)durationQ2,
           (size_t)durationQ3,
           dataSetSizeCustomersInByte,
           dataSetSizeItemsInByte,
           dataSetSizeJoinTableInByte,
           numberOfRecordsToProcess);
    fflush(stdout);

    /* Cleanup */
    free(result2.RETURN_CUSTOMER_INFO);
    DisposeTables(&customerTableColumnStore, &itemTableColumnStore);

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
    size_t numberOfRepititions = 50;
    size_t numberOfCustomersStart = 300000;
    size_t numberOfRecordsEnd = numberOfCustomersStart * 10 * 15;
    size_t stepSize = (numberOfRecordsEnd - numberOfCustomersStart)  / numberOfIndependentVariableSamples;

    printf("Timestamp;N;Repetition;Q1TimeInMsec;Q2TimeInMsec;Q3TimeInMsec;CustomerTableInByte;ItemTableInByte;JoinTableInByte;NumRecordsToProcess;Type\n");
    //for (size_t N = numberOfCustomersStart; N <= numberOfRecordsEnd; N += stepSize) {
    for (size_t N = numberOfCustomersStart; true; N += stepSize) {
        Sample(N, numberOfRepititions);
        //usleep(1 * 1000000);
    }




    return EXIT_SUCCESS;
}