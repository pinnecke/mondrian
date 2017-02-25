#ifndef PANTHEON_LOGGER_HPP
#define PANTHEON_LOGGER_HPP

#include <stdio.h>

#define LOG(file, type, message, ...)                                                       \
{                                                                                           \
    time_t t = time(NULL);                                                                  \
    struct tm tm = *localtime(&t);                                                          \
    fprintf(file, "%d-%d-%d %d:%d:%d ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,       \
              tm.tm_hour, tm.tm_min, tm.tm_sec);                                            \
    fprintf(file, type);                                                                    \
    fprintf(file, " ");                                                                     \
    fprintf(file, message, __VA_ARGS__);                                                    \
    fprintf(file, "\t(%s(%d))\n", __FILE__, __LINE__);                                      \
    fflush(file);                                                                           \
}

#if not defined(NINFO)
#define LOG_INFO(message, ...)                                                              \
    LOG(stderr, "[INFO]", message, __VA_ARGS__);
#else
#define LOG_INFO(message, ...)                                                              \
    ;
#endif

#if not defined(NWARNING)
#define LOG_WARNING(message, ...)                                                           \
    LOG(stderr, "[WARNING]", message, __VA_ARGS__);
#else
#define LOG_WARNING(message, ...)                                                           \
    ;
#endif

#endif //PANTHEON_LOGGER_HPP
