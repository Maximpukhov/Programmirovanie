#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

void divide_slau(double **a, double *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        double main = a[i][i];
        for (int j = 0; j < n; j++)
        {
            if (j != i)
                a[i][j] /= main;
        }
        x[i] /= main;
        a[i][i] = 0;
    }
}

int check_diagonalsum(double **a, int n)
{
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += fabs(a[i][j]);
        }
        sum -= fabs(a[i][i]);
        if (sum > fabs(a[i][i]))
        {
            return -1;
        }
    }
    return 0;
}

int check_norm(double **a, int n)
{
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += fabs(a[i][j]);
        }
        if (sum >= 1.0)
            return 1;
    }
    return 0;
}

void swap(double *x, double *y)
{
    double tmp = *x;
    *x = *y;
    *y = tmp;
}

void convert_slau(double **a, double *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i][i] = x[i];
        for (int j = i; j > 0; j--)
            swap(&a[i][j], &a[i][j - 1]);
    }
    for (int i = 0; i < n; i++)
        for (int j = 1; j < n; j++)
            a[i][j] = -a[i][j];

    for (int i = 0; i < n; i++)
        x[i] = 0;
}

void print_slau(double **a, double *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << (a[i][j]) << "         ";
        }
        cout << " = " << x[i] << endl;
    }
    cout << endl;
}

int MPI(double **a, double *x, int n, double eps, int *iter)
{
    if (check_diagonalsum(a, n))
    {
        cout << "Эта система уравнений не имеет решения\n";
        exit(EXIT_FAILURE);
    }

    divide_slau(a, x, n);

    if (check_norm(a, n))
    {
        cout << "Эта система уравнений не имеет решения\n";
        exit(EXIT_FAILURE);
    }

    convert_slau(a, x, n);
    double *prev_x = new double[n];
    double max_diff = 10000000;

    for (int i = 0; i < n; ++i)
        prev_x[i] = 0;

    while (max_diff > eps)
    {
        int z = 0;
        for (int i = 0; i < n; ++i)
            prev_x[i] = x[i];

        for (int i = 0; i < n; ++i)
        {
            z = 0;
            x[i] = 0;
            for (int j = 1; j < n; j++, z++)
            {
                if (z == i)
                    z++;
                x[i] += a[i][j] * prev_x[z];
            }
            x[i] += a[i][0];
        }

        max_diff = fabs(x[0] - prev_x[0]);
        for (int i = 1; i < n; ++i)
        {
            if (fabs(x[i] - prev_x[i]) > max_diff)
                max_diff = fabs(x[i] - prev_x[i]);
        }

        (*iter)++;
    }
    delete[] prev_x;
    return 0;
}

int main()
{
    ifstream file((char *)"inArray3.txt");
    if (!file.is_open())
    {
        cerr << "don`t open file" << endl;
        return -1;
    }
    int n;
    file >> n;
    double num;
    double *x = new double[n];
    double **a = new double *[n];
    for (int i = 0; i < n; ++i)
    {
        a[i] = new double[n];
        for (int j = 0; j <= n; ++j)
        {
            file >> num;
            if (j == n)
            {
                x[i] = num;
                continue;
            }
            a[i][j] = num;
        }
    }
    int itera;
    double eps = 0.01;
    MPI(a, x, n, eps, &itera);
    cout << "Решение системы уравнений методом простых итераций:\n";
    for (int i = 0; i < n; i++)
    {
        cout << "x" << i + 1 << " = " << x[i] << endl;
    }

    cout << "Количество итераций: " << itera << endl;
    for (int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
    delete[] x;
    file.close();
    return 0;
}
