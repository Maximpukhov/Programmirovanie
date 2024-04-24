#include "myBigChars.h"

int
bc_strlen (char *str)
{
  if (!str)
    {
      return 0;
    }

  int counter = 0;
  for (int i = 0; str[i] != '\n';)
    {
      if (str[i] == 0x80)
        {
          counter++;
          i++;
        }
      else if ((str[i] & 0xC0) == 0xC0) // Проверяем, что первые два бита равны 10
        {
          int num_bytes = 0;
          if ((str[i] & 0xE0) == 0xC0)
          {
            num_bytes = 2;
          }
          else if ((str[i] & 0xF0) == 0xE0)
          {
            num_bytes = 3;
          }
          else if ((str[i] & 0xF8) == 0xF0)
          {
            num_bytes = 4;
          }
          else
          {
            return 0; // Некорректное количество бит
          }
          // Проверяем, что остальные байты начинаются с бита "10"
          for (int j = 1; j < num_bytes; j++)
          {
            if ((str[i+j] & 0xC0) != 0x80)
            {
              return 0; // Некорректное начало последующего байта
            }
          }
          counter++;
          i += num_bytes;
        }
      else
        {
          return 0; // Некорректный первый байт
        }
    }
  return counter;
}
