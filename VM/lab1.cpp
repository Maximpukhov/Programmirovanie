#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

struct Fraction {
  long long numerator;
  long long denominator;

  Fraction() : numerator(0), denominator(1) {}

  Fraction(long long num, long long denom): numerator(num), denominator(denom) {
    simplify();
  }

  void simplify() {
    long long gcd = __gcd(numerator, denominator);
    numerator /= gcd;
    denominator /= gcd;
    if (denominator < 0) {
      numerator *= -1;
      denominator *= -1;
    }
  }
};

Fraction operator/(const Fraction &a, const Fraction &b) {
  return Fraction(a.numerator * b.denominator, a.denominator * b.numerator);
}

Fraction operator*(const Fraction &a, const Fraction &b) {
  return Fraction(a.numerator * b.numerator, a.denominator * b.denominator);
}

Fraction operator-(const Fraction &a, const Fraction &b) {
  return Fraction(a.numerator * b.denominator - b.numerator * a.denominator,a.denominator * b.denominator);
}

void printMatrix(const vector<vector<Fraction>> &matrix) {
  for (const auto &row : matrix) {
    for (const Fraction &element : row) 
    {
      cout << element.numerator << "/" << element.denominator << " ";
    }
    cout << endl;
  }
  cout << endl;
}

vector<vector<Fraction>> readMatrixFromFile(const string &filename) {
  ifstream file(filename);
  int rows, cols;
  file >> rows >> cols;

  vector<vector<Fraction>> matrix(rows, vector<Fraction>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      file >> matrix[i][j].numerator;
      file.ignore(1);
      file >> matrix[i][j].denominator;
    }
  }

  file.close();
  return matrix;
}

void gaussianElimination(vector<vector<Fraction>> &matrix) {
  int rows = matrix.size();
  int cols = matrix[0].size();

  for (int i = 0; i < rows; ++i) {
    cout << "Шаг " << i + 1 << ":" << endl;
    printMatrix(matrix);

    int pivot_row = i;
    for (int j = i + 1; j < rows; ++j) {
      if (abs(matrix[j][i].numerator * matrix[pivot_row][i].denominator) >
          abs(matrix[pivot_row][i].numerator * matrix[j][i].denominator)) {
        pivot_row = j;
      }
    }

    if (pivot_row != i) {
      swap(matrix[pivot_row], matrix[i]);
    }

    for (int j = i + 1; j < rows; ++j) {
      Fraction factor = matrix[j][i] / matrix[i][i];
      for (int k = i; k < cols; ++k) {
        matrix[j][k] = matrix[j][k] - matrix[i][k] * factor;
      }
    }
  }

  cout << "Шаг " << rows + 1 << ":" << endl;
  printMatrix(matrix);

  for (int i = rows - 1; i >= 0; --i) {
    for (int j = i + 1; j < cols - 1; ++j) {
      matrix[i][cols - 1] =
          matrix[i][cols - 1] - matrix[i][j] * matrix[j][cols - 1];
      matrix[i][j] = Fraction(); // Set the upper triangle elements to zero
    }
    matrix[i][cols - 1] = matrix[i][cols - 1] / matrix[i][i];
    matrix[i][i] = Fraction(1, 1);
  }

  cout << "Шаг " << rows + 2 << ":" << endl;
  printMatrix(matrix);
}

int main() {
  vector<vector<Fraction>> matrix = readMatrixFromFile("input.txt");

  cout << "Исходная матрица:\n";
  printMatrix(matrix);

  gaussianElimination(matrix);

  cout << "Преобразованная матрица:\n";
  printMatrix(matrix);

  cout << "Ответ:\n";
  for (int i = 0; i < matrix.size(); ++i) {
    Fraction solution = matrix[i].back();
    cout << "x" << i + 1 << " = " << solution.numerator << "/"<< solution.denominator << endl;
  }

  return 0;
}
