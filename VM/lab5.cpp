#include <cmath>
#include <iostream>

using namespace std;

// Функция, которую нужно интегрировать
double f(double x) { return exp(-x * x); }

// Метод центральных прямоугольников
double integrate(double a, double b, int n) {
  double h = (b - a) / n;
  double sum = 0;

  for (int i = 0; i < n; i++) { // Исправил начальное значение счетчика с 1 на 0
    sum += f((a + i * h));
  }

  return h * sum;
}

int main() {
  double a, b;
  int n; // Количество разбиений отрезка 
  double eps; // Погрешность 

  cout << "Введите начальное значение отрезка интегрирования a: ";
  cin >> a;
  cout << "Введите конечное значение отрезка интегрирования b: ";
  cin >> b;

  cout << "Выберете оценку:" << endl;
  cout << "1. Оценка по количеству разбиений N" << endl;
  cout << "2. Оценка по погрешности вычисления интеграла epsylon" << endl;
  cout << "Введите номер оценки: ";
  int choice;
  cin >> choice;

  switch (choice) {
  case 1:
    cout << "Введите количество разбиений отрезка N: ";
    cin >> n;
    cout << "Значение интеграла: " << integrate(a, b, n) << endl;
    break;
  case 2: {
    cout << "Введите погрешность вычисления интеграла epsylon: ";
    cin >> eps;

    // Итерационно находим N, удовлетворяющее заданной погрешности
    n = 10;
    double prev_result = 0;
    while (true) {
      double result = integrate(a, b, n);
      if (abs(result - prev_result) < eps) {
        cout << "Значение интеграла: " << result << endl;
        break;
      }

      prev_result = result;
      n++;
    }
    printf("колво разбиений %d", n);
    break;
  }
  default:
    cout << "Неверный выбор оценки" << endl;
  }

  return 0;
}