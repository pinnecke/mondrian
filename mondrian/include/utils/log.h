//
// Created by gabriel on 06.03.17.
//

#ifndef MND_LOG_H
#define MND_LOG_H

#include <utils/sconfig.h>
#include <ntsid.h>
#include <time.h>
#include <printf.h>
#include <libgen.h>
#include <stdio.h>

static time_t rawtime;
static struct tm * timeinfo;

#define log(where, type, msg)                                           \
    time ( &rawtime );                                                  \
    timeinfo = localtime ( &rawtime );                                  \
    char *base = basename(__FILE__);                                    \
    fprintf(where, "[%d/%d/%d %d:%d:%d] %-10.10s:%d [%-7.7s] - %s\n",   \
            timeinfo->tm_year + 1900,                                   \
            timeinfo->tm_mon + 1,                                       \
            timeinfo->tm_mday,                                          \
            timeinfo->tm_hour,                                          \
            timeinfo->tm_min,                                           \
            timeinfo->tm_sec,                                           \
            base,                                                       \
            __LINE__,                                                   \
            type, msg);

#define logf(if_call, log_macro, ...)                 \
{                                                     \
    if_call({                                         \
        char *formatted_msg;                          \
        format_string(&formatted_msg, __VA_ARGS__);   \
        log_macro(formatted_msg)                      \
        free(formatted_msg);                          \
    })                                                \
}

#ifdef LOG_DEBUG
#define debug(msg)      { log(stdout, "DEBUG", msg); }
#define if_debug(block) { block }
#define debugf(...)     { logf(if_debug, debug, __VA_ARGS__) }

#else
#define debug(msg)       ;
#define if_debug(x)      ;
#define debugf(...)      ;
#endif

#ifdef LOG_INFO
#define info(msg)       { log(stdout, "INFO", msg); }
#define if_info(block)  { block }
#define infof(...)      { logf(if_info, info, __VA_ARGS__) }
#else
#define info(msg)       ;
#define if_info(block)  ;
#define infof(...)      ;
#endif

#ifdef LOG_WARNING
#define warning(msg)            { log(stdout, "WARNING", msg); }
#define if_warning(block)       { block }
#define warningf(...)           { logf(if_warning, warning, __VA_ARGS__) }
#else
#define warning(msg)            ;
#define if_warning(block)       ;
#define warningf(...)           ;
#endif

#ifdef LOG_ERROR
#define error(msg)              { log(stderr, "ERROR", msg); }
#define if_error(block)         { block }
#define errorf(...)             { logf(if_error, error, __VA_ARGS__) }
#else
#define error(msg)              ;
#define if_error(block)         ;
#define errorf(...)             ;
#endif

#ifdef LOG_FATAL
#define fatal(msg)              { log(stderr, "FATAL", msg); exit(EXIT_FAILURE); }
#define if_fatal(block)         { block }
#define fatalf(...)             { logf(if_fatal, fatal, __VA_ARGS__) }
#else
#define fatal(msg)              ;
#define if_fatal(block)         ;
#define fatalf(...)             ;
#endif

#endif 