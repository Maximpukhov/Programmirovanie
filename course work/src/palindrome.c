#include <stdio.h>
#include <wchar.h>
#include <wctype.h>
#include <stdlib.h>
#include «palindrome.h»

Int isPalindrome(const wchar_t* str) {
    Int length = wcslen(str);
    Int i = 0;
    Int j = length – 1;

    While (i < j) {
        // Пропускаем пробелы и знаки пунктуации в начале строки
        While (i < length && !iswalpha(str[i])) {
            I++;
        }
        // Пропускаем пробелы и знаки пунктуации в конце строки
        While (j >= 0 && !iswalpha(str[j])) {
            j--;
        }

        If (i < j && towlower(str[i]) != towlower(str[j])) {
            Return 0; // Не палиндром
        }

        I++;
        j--;
    }

    Return 1; // Палиндром
}

Void checkPalindromesFromFile(const char* filename) {
    FILE* file = fopen(filename, «r»);
    If (file == NULL) {
        Wprintf(L»Не удалось открыть файл.\n»);
        Return;
    }
    Wchar_t* line = malloc(1000 * sizeof(wchar_t)); // Динамическое выделение памяти
    If (line == NULL) {
        Wprintf(L»Ошибка выделения памяти.\n»);
        Fclose(file);
        Return;
    }

    Int totalPalindromes = 0; // Общее количество палиндромических предложений
    Wprintf(L»Палиндромические предложения:\n»);

    While (fgetws(line, 1000, file)) {
        // Удаление символа новой строки, если присутствует
        If (line[wcslen(line) – 1] == L'\n') {
            Line[wcslen(line) – 1] = L'\0';
        }

        Wchar_t* context;
        Wchar_t* sentence = wcstok(line, L».!?\n», &context); // Разделение строки на предложения

        While (sentence != NULL) {
            // Удаление начальных и конечных пробелов
            While (iswspace(*sentence)) {
                Sentence++;
            }
            Wchar_t* end = sentence + wcslen(sentence) – 1;
            While (end > sentence && iswspace(*end)) {
                *end = L'\0';
                End--;
            }

            // Игнорирование пустых строк
            If (wcslen(sentence) > 1 && isPalindrome(sentence)) {
                Wprintf(L»%ls\n», sentence);
                totalPalindromes++;
            }

            Sentence = wcstok(NULL, L».!?\n», &context);
        }
    }

    Wprintf(L»\nОбщее количество палиндромических предложений: %d\n», totalPalindromes);

    Fclose(file);
    Free(line); // Освобождение динамически выделенной памяти
}
