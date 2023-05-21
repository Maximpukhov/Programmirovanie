#include <cyg2win.h>

int main(void){
	printf("Cyg2win - convert Cygwin-style Windows paths to original format\n");
    char delim[MAX_DELIM_SIZE+3]; // 64 symbols, 1 oversize canary, 1 for endline, 1 for dirty \n hack 
    printf("delim: ");
    if (input(delim, MAX_DELIM_SIZE) == -1) {
		printf("Invalid input -- delimeter must be no longer than %d symbols.\n", MAX_DELIM_SIZE);
		return -1;
	}
    
    char pathstr[MAX_PATHSTR_SIZE+3]; // same but 2048 instead of 64...
    printf("paths: ");
    if (input(pathstr, MAX_PATHSTR_SIZE) == -1) {
		printf("Invalid input -- paths string must be no longer than %d symbols.\n", MAX_PATHSTR_SIZE);
		return -1;
	}  
    
    int invalid_index = 0;
    if (check(delim, pathstr)) {
		for(int i = -7; i < invalid_index; i++)
			printf(" ");
		printf("^-- forbidden symbol\n");
		return -1;
	}
	process(delim, pathstr);
	output(pathstr);
    return 0;
}
