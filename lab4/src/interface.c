#include <stdio.h>
#include <stdlib.h>
#include <interface.h>
#include <lexer.h>
#include <strings.h>

char* input(char* delim)
{
    char ch;
    char* paths = malloc(1024);
    printf("delim: ");
    scanf("%c%c", delim, &ch);
    while (ch != '\n') {
        scanf("%c", &ch);
    }
    printf("paths: ");
    fgets(paths, 1023, stdin);
    paths[my_strlen(paths) - 1] = '\0';
    return paths;
}

void output(char* new_paths)
{
    printf("new paths: %s\n", new_paths);
}

void error_output(char* str, int exit_code)
{
    printf("%s\n", str);
    printf(RED "Error: " NONE);
    switch (exit_code) {
    case OVER_MAX_LEN:
        printf("length of path is higher than 260 symbols\n");
        break;
    case ILLEGAL_CHARACTER:
        printf("wrong symbol detected\nYou can't use \\*?«<>| in the path\n");
        break;
    case WRONG_IP_DOMEN:
        printf("IP or domen aren't correct\n");
        break;
    }
}
