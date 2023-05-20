#pragma once
#include <stdio.h>

size_t my_strlen(char* str);
char* my_strcat(char* dest, char* src);
char* my_strchr(char* s, int c);
int my_strcmp(char* str1, char* str2);
char* my_strcpy(char* toHere, char* fromHere);
char* my_strstr(char* haystack, char* needle);
char* my_strtok(char* string, char delim);
char* my_strpbrk(char* s, char* accept);
int my_isdigit(int c);
int my_isalpha(int c);
int my_tolower(int c);
int my_atoi(char* str);
