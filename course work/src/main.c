#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <stdlib.h>
#include "palindrome.h"

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    if (argc != 2) {
        wprintf(L"Использование:\n%s <имя файла>\n", argv[0]);
        return -1;
    }
    checkPalindromesFromFile(argv[1]);

    return 0;
}
