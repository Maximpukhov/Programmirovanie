#include <stdlib.h>
#include <lexer.h>
#include <strings.h>

int skip_spaces(char* str)
{
    int i;
    for (i = 0; str[i] == ' '; i++)
        ;
    return i;
}

int is_number(char* str)
{
    char* tmp_str = str;
    while (*tmp_str != '\0') {
        if (!my_isdigit(*tmp_str))
            return 0;
        tmp_str++;
    }
    return 1;
}

int check_end_symbol(char* str, char symbol)
{
    if (*(str + my_strlen(str) - 1) == symbol)
        return 1;
    return 0;
}

int get_tokens(char** tokens, char* str, char delim)
{
    int count = 0;
    tokens[0] = my_strtok(str, delim);
    while (tokens[count] != NULL) {
        count++;
        tokens[count] = my_strtok(NULL, delim);
    }
    return count;
}

int check_ip(char* str)
{
    if (!check_end_symbol(str, ':'))
        return 0;
    int len_str = my_strlen(str);
    char tmp_str[len_str + 1];
    my_strcpy(tmp_str, str);
    tmp_str[len_str - 1] = '\0';
    int after_space = skip_spaces(tmp_str);
    char* token;
    char* tokens[16];
    int count = get_tokens(tokens, &tmp_str[after_space], '.');
    if (count != 4)
        return 0;

    for (count = 0; count < 4; count++) {
        token = tokens[count];
        if (!is_number(token))
            return 0;
        if (my_atoi(token) > 255 || my_atoi(token) < 0)
            return 0;
    }
    return 1;
}

int check_upper_domens(char* str)
{
    if (!(my_strcmp(str, "ru") && my_strcmp(str, "com")
          && my_strcmp(str, "org")))
        return 1;
    return 0;
}

int is_word(char* str)
{
    char* tmp_str = str;
    while (*tmp_str != '\0') {
        if (!my_isalpha(*tmp_str))
            return 0;
        tmp_str++;
    }
    return 1;
}

int check_domen(char* str)
{
    if (!check_end_symbol(str, ':'))
        return 0;
    int str_len = my_strlen(str);
    char tmp_str[str_len + 1];
    my_strcpy(tmp_str, str);
    int after_space = skip_spaces(tmp_str);
    tmp_str[str_len - 1] = '\0';
    char* tokens[16];
    int count = get_tokens(tokens, &tmp_str[after_space], '.');
    if (count < 2 || count > 4)
        return 0;
    if (!check_upper_domens(tokens[count - 1]))
        return 0;
    for (int i = 0; i < count; i++)
        if (!is_word(tokens[i]))
            return 0;
    return 1;
}

int check_path_symbols(char* str, char* wrong_symbols)
{
    if (my_strpbrk(str, wrong_symbols) == NULL)
        return 1;
    return 0;
}

int check(char* str)
{
    if (my_strlen(str) > MAX_PATH)
        return OVER_MAX_LEN;
    if (check_path_symbols(str, "\\*?«<>|") == 0)
        return ILLEGAL_CHARACTER;
    char tmp_str[my_strlen(str) + 1];
    my_strcpy(tmp_str, str);
    my_strtok(tmp_str, '/');
    if (my_isdigit(tmp_str[0]) && check_ip(tmp_str))
        return SUCCESS;
    else if (my_isalpha(tmp_str[0]) && check_domen(tmp_str))
        return SUCCESS;

    return WRONG_IP_DOMEN;
}

char* convert_path(char* path)
{
    int path_len = my_strlen(path);
    char* new_path = malloc(path_len + 2);
    char* buffer = new_path;
    char path_copy[path_len + 1];
    my_strcpy(path_copy, path);
    char* tokens[16];
    int count = get_tokens(tokens, path_copy, '/');
    int ip_len = my_strlen(tokens[0]);
    tokens[0][ip_len - 1] = '\0';
    buffer[0] = '\\';
    buffer[1] = '\\';
    buffer[2] = '\0';
    buffer += 2;
    int i;
    for (i = 0; i < count - 1; i++) {
        buffer = my_strcat(buffer, tokens[i]);
        buffer += my_strlen(tokens[i]);
        *buffer = '\\';
        buffer++;
    }
    buffer = my_strcat(buffer, tokens[i]);
    if (check_end_symbol(path, '/')) {
        buffer += my_strlen(tokens[i]);
        *buffer = '\\';
        buffer++;
        *buffer = '\0';
    }
    return new_path;
}

char* process(char** tokens, char delim)
{
    char* new_paths = malloc(2048);
    char* buffer = new_paths;
    char* new_path;
    while (*tokens != NULL) {
        if (**tokens == '\0') {
            tokens++;
            continue;
        }
        new_path = convert_path(*tokens);
        buffer = my_strcat(buffer, new_path);
        buffer += my_strlen(buffer);
        *buffer = delim;
        buffer++;
        tokens++;
        free(new_path);
    }
    buffer--;
    *buffer = '\0';
    return new_paths;
}
