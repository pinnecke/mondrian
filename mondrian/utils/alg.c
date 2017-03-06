//
// Created by gabriel on 06.03.17.
//

#include <utils/alg.h>
#include <utils/kernel.h>

unsigned levenshtein_distance(const char *s1, const char *s2)
{
    /* This implementation based of https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C */
    unsigned s1len, s2len, x, y, lastdiag, olddiag;
    s1len = strlen(s1);
    s2len = strlen(s2);
    unsigned column[s1len+1];
    for (y = 1; y <= s1len; y++)
    {
        column[y] = y;
    }
    for (x = 1; x <= s2len; x++)
    {
        column[0] = x;
        for (y = 1, lastdiag = x-1; y <= s1len; y++)
        {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y-1] + 1, lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return column[s1len];
}

const char *str_tokenize(const char *string, const char *delimiter)
{
    static const char *working_string, *base;
    size_t end = 0, len = 0;

    if (delimiter == NULL || (string == NULL && working_string == NULL))
        return NULL;

    if (base == string) {
        string = working_string;
    } else {
        base = string;
    }

    if (string == NULL)
        return NULL;

    for (end = 0; end < strlen(string); end++) {
        for (size_t j = 0; j < strlen(delimiter); j++) {
            if (string[end] == delimiter[j]) {
                goto found;
            }
        }
    }

    working_string = NULL;
    return string;

    found:
    for (size_t j = 0; j < strlen(delimiter); j++) {
        if (string[end] == delimiter[j]) {
            end++;
            goto found;
        }
    }
    working_string = string + end;
    char *result = malloc(end);
    memcpy(result, string, end);
    result[end - 1] = '\0';

    if (strlen(working_string) == 0) {
        base = NULL;
        return base;
    }

    return result;
}