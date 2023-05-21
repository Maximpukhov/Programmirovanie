#include <mystring.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#define MAX_DELIM_SIZE 1
#define MAX_PATHSTR_SIZE 8192
#define MAX_PATH_LEN 260
#define FORBIDDEN_SYMBOLS ":\\*?«<\">|"
int input(char* str, int maxlen);
int check(char* delim, char* pathstr);
int process(char* delim, char* pathstr);
int output(char* pathstr);
