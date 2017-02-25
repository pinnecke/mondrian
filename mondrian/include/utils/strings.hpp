#ifndef PANTHEON_STRING_HPP
#define PANTHEON_STRING_HPP

#include <functional>
#include <cstring>
#include <cctype>

using namespace std;

#define BOOL_TO_STRING(x)   \
     x ? "1" : "0"

namespace pantheon
{
    namespace utils
    {
        namespace strings
        {
            namespace to_string
            {
                template <typename T>
                using function_t = function<char *(const T *t)>;

                char *uint16_to_string(const uint16_t *value)
                {
                    assert (value != nullptr);
                    char *buffer = (char *) malloc (6);
                    sprintf (buffer, "%u", *value);
                    return buffer;
                }
            }

            char *trim_inplace(char * str)
            {
                char * source = str;
                int len = strlen(source);

                while (isspace(source[len - 1])) {
                    source[--len] = 0;
                }

                while (*source && isspace(*source)) {
                    ++source;
                    --len;
                }

                memmove(str, source, len + 1);
                return str;
            }
        }
    }
}

#endif //PANTHEON_STRING_HPP
