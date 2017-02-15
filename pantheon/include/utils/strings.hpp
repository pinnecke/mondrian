#ifndef PANTHEON_STRING_HPP
#define PANTHEON_STRING_HPP

namespace pantheon
{
    namespace utils
    {
        namespace strings
        {
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
