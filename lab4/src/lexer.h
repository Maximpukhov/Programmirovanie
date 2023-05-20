#pragma once
#define MAX_PATH 260

enum exit_codes {
    SUCCESS,
    OVER_MAX_LEN,
    ILLEGAL_CHARACTER,
    WRONG_IP_DOMEN,
};

int check(char* str);
char* process(char** tokens, char delim);
int get_tokens(char** tokens, char* str, char delim);
