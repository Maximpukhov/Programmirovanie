#include <cmath>
#include <iostream>

using namespace std;

int iteration = 1;
double error;

double f(double x) {
  return 2 - sqrt(pow(x, 3)) - 2 * log(x);
}

double df(double x) {
  return -3 * pow(x, -0.5) / 2 - 2 / x;
}

double linKernighan(double a, double b, double epsilon) {
  double x = (a + b) / 2;
  double bestX = x;

  while (abs(f(x)) > epsilon) {
    double derivative = df(x);

    double step = -f(x) / derivative;

    x += step;

    if (abs(f(x)) < abs(f(bestX))) {
      bestX = x;
    }

    error = abs(x - bestX);

    cout << "Итерация: " << iteration << endl;
    cout << "Корень: " << x << endl;
    cout << "Точность: " << error << endl;
    cout << endl;
    iteration++;
  }

  return bestX;
}

int main() {
  double a, b, epsilon;

  cout << "Введите левый конец отрезка a: ";
  cin >> a;
  cout << "Введите правый конец отрезка b: ";
  cin >> b;
  cout << "Введите точность решения epsilon: ";
  cin >> epsilon;

  double root = linKernighan(a, b, epsilon);
  cout << "Корень нелинейного уравнения: " << root << endl;

  return 0;
}
