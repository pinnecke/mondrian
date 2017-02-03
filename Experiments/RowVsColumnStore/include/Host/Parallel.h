#pragma once

#define SUM_IN_PARALLEL(numOfElements, dataIndexesArrayPtr, dataArrayPtr, func)                                        \
{                                                                                                                      \
    size_t *localResults = (size_t *) malloc(sizeof(size_t) * NUM_HOST_THREADS);                                       \
                                                                                                                       \
    for(unsigned threadId = 0; threadId < NUM_HOST_THREADS; ++threadId) {                                              \
        localResults[threadId] = 0;                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    size_t arrayNumberOfElements = numOfElements;                                                                      \
    size_t *indexArray = dataIndexesArrayPtr;                                                                          \
    auto *dataArray = dataArrayPtr;                                                                                    \
    unsigned numberOfThreadsInUse = NUM_HOST_THREADS;                                                                  \
                                                                                                                       \
    std::vector<std::thread> threads(numberOfThreadsInUse);                                                            \
                                                                                                                       \
    for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {                                         \
        threads[threadId] = std::thread(                                                                               \
        [threadId, &localResults, &arrayNumberOfElements, &indexArray, &dataArray, &numberOfThreadsInUse]() -> void {  \
            size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));                \
                                                                                                                       \
            size_t local_sum = 0;                                                                                      \
            auto threadMaxIdx = chunkSize * threadId + chunkSize;                                                      \
            auto end = std::min(threadMaxIdx, arrayNumberOfElements);                                                  \
                                                                                                                       \
            for (size_t cursor = chunkSize * threadId; cursor < end; cursor++)                                         \
                { func }                                                                                               \
                                                                                                                       \
            localResults[threadId] = local_sum;                                                                        \
        });                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    for (auto &thread : threads)                                                                                       \
        thread.join();                                                                                                 \
                                                                                                                       \
    for (size_t i = 0; i < numberOfThreadsInUse; i++)                                                                  \
        sum += localResults[i];                                                                                        \
                                                                                                                       \
    free(localResults);                                                                                                \
}

#define MATERIALIZE_IN_PARALLEL(func)                                                                                  \
{                                                                                                                      \
    size_t arrayNumberOfElements = Params.NumOfCustomerTupleIds;                                                       \
    size_t *dataArray = Params.CustomerTupleIds;                                                                       \
    unsigned numberOfThreadsInUse = NUM_HOST_THREADS;                                                                  \
                                                                                                                       \
    std::vector<std::thread> threads(numberOfThreadsInUse);                                                            \
                                                                                                                       \
    for (unsigned threadId = 0; threadId < numberOfThreadsInUse; ++threadId) {                                         \
            threads[threadId] = std::thread(                                                                           \
            [threadId, &arrayNumberOfElements, &dataArray, &numberOfThreadsInUse, this]() -> void {                    \
            size_t chunkSize = size_t(std::ceil(arrayNumberOfElements / (float) numberOfThreadsInUse));                \
            auto threadMaxIdx = chunkSize * threadId + chunkSize;                                                      \
            auto end = std::min(threadMaxIdx, arrayNumberOfElements);                                                  \
                                                                                                                       \
            for (size_t cursor = chunkSize * threadId; cursor < end; cursor++)                                         \
                { func }                                                                                               \
        });                                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    for (auto &thread : threads)                                                                                       \
    thread.join();                                                                                                     \
}
