
#include <cstdlib>
#include <limits>
#include <printf.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <thread>
#include <vector>

/* Taken from http://stackoverflow.com/questions/2808398/easily-measure-elapsed-time */
template<typename TimeT = std::chrono::nanoseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F &&func, Args &&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT>
                (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

struct TcpCOrderTable {
    unsigned *O_ID, *O_D_ID, *O_W_ID, *O_C_ID;  /* unique */
    u_int64_t *O_ENTRY_D; /* date */
    unsigned *O_CARRIER_ID; /* nullable unique */
    u_int64_t *O_OL_CNT, *O_ALL_LOCAL; /* numeric(0), numeric(1) */
};

void CreateTable(TcpCOrderTable *pTable, size_t i) {
    pTable->O_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->O_D_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->O_W_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->O_C_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->O_ENTRY_D = (u_int64_t *) malloc(sizeof(u_int64_t) * i);
    pTable->O_CARRIER_ID = (unsigned int *) malloc(sizeof(unsigned) * i);
    pTable->O_OL_CNT = (u_int64_t *) malloc(sizeof(u_int64_t) * i);
    pTable->O_ALL_LOCAL = (u_int64_t *) malloc(sizeof(u_int64_t) * i);
}

void FillTable(TcpCOrderTable *pTable, size_t NumRecords) {
    for (size_t i = 0; i < NumRecords; i++) {
        pTable->O_ID[i] = rand() % 10000000;
        pTable->O_D_ID[i] = rand() % 20;
        pTable->O_W_ID[i] = rand() % 20;
        pTable->O_C_ID[i] = rand() % 96000;
        pTable->O_ENTRY_D[i] = rand() % std::numeric_limits<u_int64_t >::max() - 1;
        pTable->O_CARRIER_ID[i] = rand() % 11;
        pTable->O_OL_CNT[i] = rand() % std::numeric_limits<u_int64_t >::max() - 1;
        pTable->O_ALL_LOCAL[i] = rand() % std::numeric_limits<u_int64_t >::max() - 1;
    }
}

void DisposeTable(TcpCOrderTable *pTable) {
    free(pTable->O_ID);
    free(pTable->O_D_ID);
    free(pTable->O_W_ID);
    free(pTable->O_C_ID);
    free(pTable->O_ENTRY_D);
    free(pTable->O_CARRIER_ID);
    free(pTable->O_OL_CNT);
    free(pTable->O_ALL_LOCAL);
}

struct QueryParams {
    unsigned O_W_ID, O_D_ID, O_C_ID;
    /* maximum O_ID */
    /* return O_ID, O_ENTRY_D, O_CARRIER_ID */
};

struct ResultSet {
    unsigned RETURN_O_ID;
    u_int64_t RETURN_O_ENTRY_D;
    unsigned  RETURN_O_CARRIER_ID;
};

QueryParams *CreateQueryParams(TcpCOrderTable *pTable, size_t TableSize) {
    size_t recordId = rand() % TableSize;
    QueryParams *returnValue = (QueryParams *) malloc(sizeof(QueryParams));
    returnValue->O_C_ID = pTable->O_C_ID[recordId];
    returnValue->O_W_ID = pTable->O_W_ID[recordId];
    returnValue->O_D_ID = pTable->O_D_ID[recordId];
    return returnValue;
}

struct QueryForData {

    ResultSet *out;
    TcpCOrderTable *table;
    size_t tableSize;
    QueryParams* params;

    QueryForData(ResultSet *Out, TcpCOrderTable *Table, size_t TableSize, QueryParams* Params) {
        this->out = Out;
        this->table = Table;
        this->tableSize = TableSize;
        this->params = Params;
    }

    void operator()() const {
        /*The row in the ORDER table with matching O_W_ID (equals C_W_ID), O_D_ID (equals C_D_ID), O_C_ID (equals C_ID),
         * and with the largest existing O_ID, is selected. This is the most recent order placed by that customer. O_ID,
         * O_ENTRY_D, and O_CARRIER_ID are retrieved.*/

        unsigned maxOId = 0;
        unsigned *RETURN_O_ID;
        u_int64_t *RETURN_O_ENTRY_D;
        unsigned  *RETURN_O_CARRIER_ID;


        // Scan (equivalent to compiled queried)
        /*for (size_t cursor = 0; cursor < tableSize; cursor++) {
            if (table->O_W_ID[cursor] == Params->O_W_ID &&
                    table->O_D_ID[cursor] == Params->O_D_ID &&
                    table->O_C_ID[cursor] == Params->O_C_ID) {

                if (table->O_ID[cursor] > maxOId) {
                    maxOId = table->O_ID[cursor];
                    RETURN_O_ID = table->O_ID + cursor;
                    RETURN_O_ENTRY_D = table->O_ENTRY_D + cursor;
                    RETURN_O_CARRIER_ID = table->O_CARRIER_ID + cursor;;
                }
            }
        }*/

        int NUMBER_OF_THREADS = 4;

        std::thread threads[NUMBER_OF_THREADS];
        std::vector<std::vector<size_t>> localResults(NUMBER_OF_THREADS);
        for (auto& localResult : localResults)
            localResult.reserve(1);

        for(unsigned threadId = 0; threadId < NUMBER_OF_THREADS; ++threadId) {
            threads[threadId] = std::thread([threadId, &localResults, &tableSize] () -> void {
                auto chunkSize = long(tableSize / NUMBER_OF_THREADS);
                unsigned maxOId = 0;
                int64_t recordId = -1;

                for (size_t cursor = chunkSize * threadId; cursor < chunkSize * threadId + chunkSize; cursor++) {
                    if (table->O_W_ID[cursor] == params->O_W_ID &&
                        table->O_D_ID[cursor] == params->O_D_ID &&
                        table->O_C_ID[cursor] == params->O_C_ID) {

                        if (table->O_ID[cursor] > maxOId) {
                            recordId = cursor;
                            maxOId = table->O_ID[cursor];
                        }
                    }
                }

                if (recordId != -1)
                    localResults[threadId][0] = recordId;

            });
        }
        for (auto& thread : threads)
            thread.join();

        size_t resultSetSize = 0;
        size_t offsets[NUMBER_OF_THREADS];
        for (size_t i = 0; i < NUMBER_OF_THREADS; i++) {
            auto& localResult = localResults[i];
            if(localResult.size() > 0)

        }







        // Materialize
        out->RETURN_O_ID = *RETURN_O_ID;
        out->RETURN_O_ENTRY_D = *RETURN_O_ENTRY_D;
        out->RETURN_O_CARRIER_ID = *RETURN_O_CARRIER_ID;
    }
};


void Sample(size_t N, size_t NumberOfRepetitions) {
    for (size_t currentRepetition = 0; currentRepetition < NumberOfRepetitions; currentRepetition++) {
        /* Setup */
        TcpCOrderTable table;
        srand(0);
        CreateTable(&table, N);
        FillTable(&table, N);
        auto queryParams = CreateQueryParams(&table, N);

        /* Query */
        ResultSet result;
        auto duration = measure<>::run(QueryForData(&result, &table, N, queryParams));

        size_t dataSetSizeInByte = (sizeof(unsigned) * 5 + sizeof(u_int64_t) * 3) * N;



        /* Cleanup */
        printf("%zu,%zu,%zu,%zu,%zu,%zu,%zu;%zu;%f\n",
               (size_t)result.RETURN_O_ID,
               (size_t)result.RETURN_O_ENTRY_D,
               (size_t)result.RETURN_O_CARRIER_ID,
               (size_t)time(NULL),
               (size_t)N,
               (size_t)currentRepetition,
               (size_t)duration,
               dataSetSizeInByte,
               dataSetSizeInByte/1024.0f/1024.0f/1024.0f);

        DisposeTable(&table);
    }
}

int main() {

    size_t numberOfIndependentVariableSamples = 30;
    size_t numberOfRepititions = 10;
    size_t numberOfRecordsStart = 1000000;
    size_t numberOfRecordsEnd = 1000000 * 2000;
    size_t stepSize = (numberOfRecordsEnd - numberOfRecordsStart)  / numberOfIndependentVariableSamples;

    printf("RETURN_O_ID;RETURN_O_ENTRY_D;RETURN_O_CARRIER_ID;Timestamp,N;Repetition;TimeInMsec;DataSetSizeInByte;DataSetSizeInGigaByte\n");
    for (size_t N = numberOfRecordsStart; N <= numberOfRecordsEnd; N += stepSize) {
        Sample(N, numberOfRepititions);
        //usleep(1 * 1000000);
    }




    return EXIT_SUCCESS;
}

