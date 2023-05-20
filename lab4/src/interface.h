#define RED "\033[0;32;31m"
#define NONE "\033[m"

char* input(char* delim);
void output(char* new_paths);
void error_output(char* str, int exit_code);
