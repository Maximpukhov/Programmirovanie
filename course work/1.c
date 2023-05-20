#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <wctype.h>
#include <stdlib.h>

int isPalindrome(const wchar_t* str) {
    int length = wcslen(str);
    int i = 0;
    int j = length - 1;

    while (i < j) {
        // Пропускаем пробелы и знаки пунктуации в начале строки
        while (i < length && !iswalpha(str[i])) {
            i++;
        }
        // Пропускаем пробелы и знаки пунктуации в конце строки
        while (j >= 0 && !iswalpha(str[j])) {
            j--;
        }

        if (i < j && towlower(str[i]) != towlower(str[j])) {
            return 0; // Не палиндром
        }

        i++;
        j--;
    }

    return 1; // Палиндром
}

void checkPalindromesFromFile(const char* filename) {

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        wprintf(L"Не удалось открыть файл.\n");
        return;
    }

    wchar_t* line = malloc(1000 * sizeof(wchar_t)); // Динамическое выделение памяти
    if (line == NULL) {
        wprintf(L"Ошибка выделения памяти.\n");
        fclose(file);
        return;
    }

    int totalPalindromes = 0; // Общее количество палиндромических предложений
    wprintf(L"Палиндромические предложения:\n");

    while (fgetws(line, 1000, file)) {
        // Удаление символа новой строки, если присутствует
        if (line[wcslen(line) - 1] == L'\n') {
        line[wcslen(line) - 1] = L'\0';
    }

    wchar_t* context;
    wchar_t* sentence = wcstok(line, L".!?\n", &context); // Разделение строки на предложения

    while (sentence != NULL) {
        // Удаление начальных и конечных пробелов
        while (iswspace(*sentence)) {
        sentence++;
    }
    wchar_t* end = sentence + wcslen(sentence) - 1;
    while (end > sentence && iswspace(*end)) {
        *end = L'\0';
        end--;
    }

    // Игнорирование пустых строк
    if (wcslen(sentence) > 1 && isPalindrome(sentence)) {
        wprintf(L"%ls\n", sentence);
        totalPalindromes++;
    }

    sentence = wcstok(NULL, L".!?\n", &context);
    }
    }

    wprintf(L"\nОбщее количество палиндромических предложений: %d\n", totalPalindromes);

        fclose(file);
        free(line); // Освобождение динамически выделенной памяти
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    if (argc != 2) {
        wprintf(L"Использование:\npalindrom <имя файла>\n");
        return -1;
    }
    checkPalindromesFromFile(argv[1]);

    return 0;
}
