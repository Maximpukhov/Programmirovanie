#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <stdlib.h>
#include «palindrome.h»

Int main(int argc, char **argv) {
    Setlocale(LC_ALL, «»);
    If (argc != 2) {
        Wprintf(L»Использование:\n%s <имя файла>\n», argv[0]);
        Return -1;
    }
    checkPalindromesFromFile(argv[1]);

    return 0;
}
