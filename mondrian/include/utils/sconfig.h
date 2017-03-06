//
// Created by gabriel on 06.03.17.
//

#ifndef MONDRIAN_SCONFIG_H
#define MONDRIAN_SCONFIG_H

#include <stdbool.h>
#include <utils/debug_build.h>
#include "result.h"

#ifdef NDEBUG
#define OPT_DEBUG_BUILD false
#else
#define OPT_DEBUG_BUILD true
#endif

#ifdef LOG_DEBUG
#define OPT_LOG_DEBUG true
#else
#define OPT_LOG_DEBUG false
#endif

#ifdef LOG_INFO
#define OPT_LOG_INFO true
#else
#define OPT_LOG_INFO false
#endif

#ifdef LOG_WARNING
#define OPT_LOG_WARNING true
#else
#define OPT_LOG_WARNING false
#endif

#ifdef LOG_ERROR
#define OPT_LOG_ERROR true
#else
#define OPT_LOG_ERROR false
#endif

#ifdef LOG_FATAL
#define OPT_LOG_FATAL true
#else
#define OPT_LOG_FATAL false
#endif

#ifdef ZERO_MEMORY
#define OPT_ZERO_MEMORY true
#else
#define OPT_ZERO_MEMORY false
#endif

struct static_boolean_option
{
    const char *name;
    const char *desc;
    bool enabled;
};

enum static_config_option
{
    SCO_DEBUG_BUILD,
    SCO_LOG_DEBUG,
    SCO_LOG_INFO,
    SCO_LOG_WARNING,
    SCO_LOG_ERROR,
    SCO_LOG_FATAL,
    SCO_ZERO_MEMORY,

    /* Add a new config option above this line and after the last element so far */
    /* For each new config option, add a descriptor in the 'static_options' array */


    /* Do not change this element, and do not set any value assignments to an element in this enum */
            XXX_LAST_ELEMENT_OF_OPTION_ARRAY
};

static struct static_boolean_option static_options[XXX_LAST_ELEMENT_OF_OPTION_ARRAY] =
        {
                /* descriptions is allowed to have up to 41 characters, name is allowed to have up to 17 characters */
                /* *** new entries must be added to the end of this list *** */

                /* SCO_DEBUG_BUILD */
                {
                        .desc = "Enables debug symbols and assertions.",
                        .name = "DEBUG_BUILD",
                        .enabled = OPT_DEBUG_BUILD
                },

                /* SCO_LOG_DEBUG */
                {
                        .desc = "Displays debug messages.",
                        .name = "LOG_DEBUG",
                        .enabled = OPT_LOG_DEBUG
                },

                /* SCO_LOG_INFO */
                {
                        .desc = "Displays information messages.",
                        .name = "LOG_INFO",
                        .enabled = OPT_LOG_INFO
                },

                /* SCO_LOG_WARNING */
                {
                        .desc = "Displays warning messages.",
                        .name = "LOG_WARNING",
                        .enabled = OPT_LOG_WARNING
                },

                /* SCO_LOG_ERROR */
                {
                        .desc = "Displays error messages.",
                        .name = "LOG_ERROR",
                        .enabled = OPT_LOG_ERROR
                },

                /* SCO_LOG_FATAL */
                {
                        .desc = "Stops with message + EXIT_FAILURE.",
                        .name = "LOG_FATAL",
                        .enabled = OPT_LOG_FATAL
                },

                /* SCO_ZERO_MEMORY */
                {
                        .desc = "Set memory to zero after allocation.",
                        .name = "ZERO_MEMORY",
                        .enabled = OPT_ZERO_MEMORY
                },

                /* Add a new config option above this line and after the last element so far */
        };

static enum MND_RESULT mnd_sconfig_get(struct static_boolean_option *state, enum static_config_option option_name)
{
    if (option_name >= XXX_LAST_ELEMENT_OF_OPTION_ARRAY)
        return MND_ILLEGAL_ARGUMENT;

    *state = static_options[option_name];

    return MND_OK;
}

#endif
