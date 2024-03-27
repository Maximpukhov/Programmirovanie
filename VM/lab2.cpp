#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

void division_slau(vector<vector<double>> &a, vector<double> &x, int n)
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

int сheck_diagonal_sum(vector<vector<double>> &a, int n)
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

int check_norm(vector<vector<double>> &a, int n)
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

void change_slau(vector<vector<double>> &a, vector<double> &x, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i][i] = x[i];
        for (int j = i; j > 0; j--)
            swap(a[i][j], a[i][j - 1]);
    }
    for (int i = 0; i < n; i++)
        for (int j = 1; j < n; j++)
            a[i][j] = -a[i][j];

    for (int i = 0; i < n; i++)
        x[i] = 0;
}

void print_slau(vector<vector<double>> &a, vector<double> &x, int n)
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
int MPI(vector<vector<double>> &a, vector<double> &x, int n, double eps, int *iter)
{
    bool flag = сheck_diagonal_sum(a, n);
    if (flag)
    {
        for(int i = 0 ; i < n; i++)
        {
            a[i].push_back(x[i]);
        }
        int counter = 0;
        int max_permutation = 10000;

        while (1)
        {
            next_permutation(a.begin(), a.end());
            counter++;
            if (counter > max_permutation)
                break;
            for (int i = 0; i < a.size(); i++)
            {
                for (int j = 0; j < a[i].size(); j++)
                {
                    cout << a[i][j] << " ";
                }
                cout << endl;
            }

            cout << endl;
            if (!сheck_diagonal_sum(a, n))
            {
                flag = 0;
                break;
            }
        }

        for(int i = 0 ; i < n; i++)
        {
            x[i] = a[i][n];
            a[i].pop_back();
        }

    }

    if (flag)
    {
        cout << "Эта система уравнений не имеет решения\n";
        exit(EXIT_FAILURE);
    }

    division_slau(a, x, n);

    if (check_norm(a, n))
    {
        cout << "Эта система уравнений не имеет решения\n";
        exit(EXIT_FAILURE);
    }

    change_slau(a, x, n);
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
            cout << "x" << i + 1 << "= " << x[i] << endl;
        }

        max_diff = fabs(x[0] - prev_x[0]);
        for (int i = 1; i < n; ++i)
        {
            if (fabs(x[i] - prev_x[i]) > max_diff)
                max_diff = fabs(x[i] - prev_x[i]);
        }

        (*iter)++;
        cout << *iter << endl;
        cout << endl;
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
    vector<double> x(n);
    vector<vector<double>> a(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= n; ++j)
        {
            file >> num;
            if (j == n)
            {
                x[i] = num;
                continue;
            }
            a[i].push_back(num);
        }
    }
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[i].size(); j++)
        {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }

    cout << endl;

    for (int i = 0; i < x.size(); i++)
    {
        cout << x[i] << endl;
    }

    cout << endl;

    int itera;
    double eps = 0.01;
    MPI(a, x, n, eps, &itera);
    cout << "Решение системы уравнений методом простых итераций:\n";
    for (int i = 0; i < n; i++)
    {
        cout << "x" << i + 1 << " = " << x[i] << endl;
    }

    cout << "Количество итераций: " << itera << endl;
    file.close();
    return 0;
}
