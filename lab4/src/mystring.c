#include <mystring.h>
#include <stdio.h>

size_t slen(const char *str){
    int i = 0;
    while(str[i] != '\0') i++;
    return i;
};

int schr(char str[], const char subchar) {
	for(int i = 0; i < slen(str); i++)
		if (str[i] == subchar) 
			return i;
	return -1; 
}

// I'm not sure of it
int stok(char str[], const char delim, char *ptr[]) {
    char* suf = str;
    ptr[0] = str;
    int i, j = 1;
    while((i=schr(suf,delim))>=0){
		suf[i] = '\0';
		suf = suf + i + 1;
		ptr[j] = suf;
		j++;
	}	
	return j;
}

int suntok(char str[], char delim, char *ptr[], int cnt) {
    int i;
    for(i = 1; i < cnt; i++) {
        *(ptr[i] - 1) = delim;
    }
    return i;
}

size_t sspn(const char *str, const char *sym) {
    size_t len = 0;
    for(int i = 0; i < slen(str); i++){
        size_t tlen = len;
        for (int k = 0; k < slen(sym); k++) {
            if (str[i] == sym[k]){
                len++;
                break;
            }
        }
        if (tlen == len) return len;
    }
    return len;
}

int scmp( const char * string1, const char * string2 ){
    int i = 0;
    while (string1[i] == string2[i] && string1[i] != '\0' && string2[i] != '\0') 
        i++;
    return string1[i]-string2[i];
}

void sclean(char* in) {
    int k = slen(in);
    for (int i = 0; i < k; i++)
        in[i] = '\0';
}

char *scpy (char *dst, const char *src) {
    for (int i = 0; i < slen(src); i++)
        dst[i] = src[i];
    return dst;
}

int sncmp( const char * string1, const char * string2, int n) {
    int i = 0;
    while (string1[i] == string2[i] && string1[i] != '\0' && string2[i] != '\0' && i < n-1) {
        i++;
    }
    return string1[i]-string2[i];
}

char to_upper(char c){
    if (c >= 'a' && c <= 'z')
        return c - 0x20;
    return c;
}

int sconcat(char* str1, char* str2){
    for (int i = slen(str1), j = 0; i < slen(str1)+slen(str2); i++){
        str1[i] = str2[j];
        j++;
    } 
    return slen(str1);
}
