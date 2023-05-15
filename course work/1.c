#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#define BUFFER_SIZE 1024 // максимальный размер буфера для чтения из файла

// Для перевода текста в нижний регистр
char rus1[] = "ёйцукенгшщзхъфывапролджэячсмитьбю";
char rus2[] = "ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ";

char eng1[] = "qwertyuiopasdfghjklzxcvbnm";
char eng2[] = "QWERTYUIOPASDFGHJKLZXCVBNM";

char _tolower(char c)
{
  char *s;

  if ((s = strchr(rus2, c)) != NULL)
    return rus1[s - rus2];

  if ((s = strchr(eng2, c)) != NULL)
    return eng1[s - eng2];

  return c;
}

int _isalpha_digit(char c)
{
  if (c >= '0' && c <= '9')
    return 1;
  if (c >= 'a' && c <= 'z')
    return 1;
  if (c >= 'A' && c <= 'Z')
    return 1;
  if (strchr(rus1, c) || strchr(rus2, c))
    return 1;
  return 0;
}

/* Функция, удаляющая знаки препинания, пробелы и переводящая текст в нижний регистр */
char *strip_punctuation(char *str)
{
  int  len = strlen(str);
  char *new_str = (char *)malloc(len + 1); // выделяем память для новой строки
  int  i, j;
  char c;

  for (i = j = 0; i < len; ++i)
  {
    if (_isalpha_digit(c = str[i]))
    {
      // если символ не знак препинания и не пробел
      new_str[j++] = _tolower(c); // добавляем его в новую строку в нижнем регистре
    }
  }
  new_str[j] = '\0'; // ставим символ конца строки
  return new_str;
}

/* Функция, проверяющая, является ли строка str палиндромом */
int is_palindrome(char *str)
{
  int len = strlen(str), i;

  for (i = 0; i < len / 2; i++)
  {
    if (str[i] != str[len - 1 - i])
    {
      // сравниваем символы с начала и конца строки
      return 0; // если хотя бы одна пара не равна, то это не палиндром
    }
  }
  return 1; // строка является палиндромом
}

int main(int argc, char **argv)
{
  // Установить кодировку печатаемых символов
  // для корректной печати русского текста
  setlocale (LC_ALL, "Russian");

  if (argc != 2)
  { // если аргументов командной строки не два (программа и имя файла)
    printf("Usage: %s filename\n", argv[0]); // выводим сообщение об использовании программы
    return 1;
  }

  char *filename = argv[1]; // имя файла из аргумента командной строки
  FILE *file = fopen(filename, "r"); // открываем файл на чтение
  if (file == NULL)
  { // если не удалось открыть файл
    printf("Error: Cannot open file '%s'\n", filename); // выводим сообщение об ошибке
    return 1;
  }

  char buffer[BUFFER_SIZE]; // буфер для чтения из файла
  char *sentence; // указатель на текущее предложение
  char *stripped_sentence; // указатель на текущее предложение без знаков препинания
  int num_palindromes = 0; // количество найденных палиндромов

  while (fgets(buffer, BUFFER_SIZE, file))
  {
    sentence = strtok(buffer, ".?!"); // разбиваем текст на предложения
    while (sentence)
    {
      stripped_sentence = strip_punctuation(sentence); // удаляем знаки препинания
      if (is_palindrome(stripped_sentence))
      { // проверяем, является ли предложение палиндромом
        ++num_palindromes;
        printf("%s\n", stripped_sentence); // выводим палиндром на экран
      }
      free(stripped_sentence); // освобождаем выделенную под предложение память
      sentence = strtok(NULL, ".?!");
    }
  }
  printf("Total number of palindromes found: %d\n", num_palindromes);

  fclose(file);
  return 0;
}
