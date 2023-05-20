#include <stdio.h>
#include <strings.h>

size_t my_strlen(char* str)
{
    size_t count;
    for (count = 0; str[count] != '\0'; count++)
        ;
    return count;
}

char* my_strcat(char* dest, char* src)
{
    char* tmp = dest + my_strlen(dest);
    int i;
    for (i = 0; src[i] != '\0'; i++)
        tmp[i] = src[i];
    tmp[i] = '\0';
    return dest;
}

char* my_strchr(char* s, int c)
{
    for (int i = 0; s[i] != '\0'; i++)
        if (s[i] == c)
            return &s[i];
    return NULL;
}

int my_strcmp(char* str1, char* str2)
{
    int i;
    for (i = 0; str1[i] == str2[i]; i++)
        if (str1[i] == '\0')
            return 0;

    if (str1[i] > str2[i])
        return 1;
    else
        return -1;
}

char* my_strcpy(char* toHere, char* fromHere)
{
    int i;
    for (i = 0; fromHere[i] != '\0'; i++)
        toHere[i] = fromHere[i];
    toHere[i] = '\0';
    return toHere;
}

char* my_strstr(char* haystack, char* needle)
{
    char* tmp_needle = needle;
    char* tmp_haystack;
    while (1) {
        while (*haystack != *tmp_needle) {
            if (*haystack == '\0')
                return NULL;
            haystack++;
        }
        tmp_haystack = haystack;
        while (1) {
            haystack++;
            tmp_needle++;
            if (*tmp_needle == '\0') {
                return tmp_haystack;
            }
            if (*haystack != *tmp_needle)
                break;
        }
        haystack = tmp_haystack + 1;
        tmp_needle = needle;
    }
}

char* my_strtok(char* string, char delim)
{
    static char* last;
    if (string != NULL)
        last = string;
    if (last == NULL)
        return NULL;
    char* tmp = last;
    while (*tmp == delim)
        tmp++;
    if (*tmp == '\0')
        return NULL;
    int i;
    last = tmp;
    for (i = 0; tmp[i] != delim; i++)
        if (tmp[i] == '\0') {
            last = NULL;
            return tmp;
        }
    last += i + 1;
    tmp[i] = '\0';
    return tmp;
}

char* my_strpbrk(char* s, char* accept)
{
    for (int i = 0; s[i] != '\0'; i++)
        for (int j = 0; accept[j] != '\0'; j++)
            if (s[i] == accept[j])
                return &s[i];
    return NULL;
}

int my_isdigit(int c)
{
    if (c >= '0' && c <= '9')
        return 1;
    return 0;
}

int my_isalpha(int c)
{
    if (my_tolower(c) >= 'a' && my_tolower(c) <= 'z')
        return 1;
    return 0;
}

int my_tolower(int c)
{
    if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';
    return c;
}

int my_atoi(char* str)
{
    int minus = 0;
    if (*str == '-') {
        minus = 1;
        str++;
    }
    if (!my_isdigit(*str))
        return 0;
    int number = *str - '0';
    str++;
    while (my_isdigit(*str)) {
        if (number > 999999999) {
            number = 2147483647;
            break;
        }
        number *= 10;
        number += *str - '0';
        str++;
    }
    if (minus)
        number = 0 - number;
    return number;
}
