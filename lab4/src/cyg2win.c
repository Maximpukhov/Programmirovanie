#include <cyg2win.h>

int input(char* str, int maxlen){
	fgets(str, maxlen+3, stdin); //
	if (slen(str) >= maxlen+2)
		return -1;
	str[slen(str)-1] = '\0'; 
	return 0;
}

int check(char* delim, char* paths){
	if (delim == NULL || paths == NULL) return -1;
	char* ptr[MAX_PATH_LEN];
	int errsym_index;
	int cnt = stok(paths, *delim, ptr);
	for (int i = 0; i < cnt; i++) {
		if (slen(ptr[i]) > MAX_PATH_LEN) return i+1;
		int forbidden_pos = sspn(ptr[i], FORBIDDEN_SYMBOLS);
		if (forbidden_pos) return -(errsym_index+forbidden_pos);
		errsym_index += slen(ptr[i]);
	}
	suntok(paths, *delim, ptr, cnt); 
	return 0;
}

int process(char* delim, char* pathstr){
	char* newpathst = calloc(MAX_PATHSTR_SIZE, sizeof(char)); 
	char* ptr[MAX_PATH_LEN];
	int cnt = stok(pathstr, *delim, ptr);
	int skiplen;
	for(int i = 0; i < cnt; i++) {
		char driveletter = '\0';
		if (!sncmp(ptr[i], "cygdrive/", 9)){
			driveletter = ptr[i][9];
			skiplen = 11;
		}
		else if (!sncmp(ptr[i], "/cygdrive/", 10)){
			driveletter = ptr[i][10];
			skiplen = 12;
		}
		if (driveletter) {
			char newstr[MAX_PATH_LEN] = {'\0'};
			newstr[0] = to_upper(driveletter);
			newstr[1] = ':';
			newstr[2] = '\\';
			for(int k = skiplen, j = 0; k < slen(ptr[i]); k++) {
				if (ptr[i][k] == '/') newstr[j+3] = '\\';
				else newstr[j+3] = ptr[i][k];
				j++;
			}
			sconcat(newpathst, newstr);
			sconcat(newpathst, delim);
		}
		else{ 
			sconcat(newpathst, ptr[i]);
			sconcat(newpathst, delim);
		}
	}
	suntok(pathstr, delim[0], ptr, cnt);
	newpathst[slen(newpathst)-1] = '\0';
	sclean(pathstr);
	scpy(pathstr, newpathst);
	return 0;
}

int output(char* pathstr){
	printf("new paths: %s\n", pathstr);
	return 0;
}
