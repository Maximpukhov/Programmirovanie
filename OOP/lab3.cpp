#include <iostream>
#include <stdlib.h>
#include <cstdlib>

using namespace std;

class Matrix
{
private:
    int n;
    int sum;
    int m;
    int **arr;
    void create_matrix(int n, int m)
    {
        arr = new int *[n];
        for (int i = 0; i < n; i++)
        {
            arr[i] = new int[m];
        }
    }

public:
    Matrix() // конструктор класса матрица 0
    {
        create_matrix(0, 0);
    }
    Matrix(int n) // конструктор класса матрица квадрат единичной
    {
        this->n = n;
        this->m = n;
        this->sum = 0;
        create_matrix(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    arr[i][j] = 1;
                else
                {
                    arr[i][j] = 0;
                }
            }
        }
    }
    Matrix(int n, int m) // конструктор класса матрица указ размерность
    {
        this->n = n;
        this->m = m;
        this->sum = 0;
        create_matrix(n, m);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i][j] = 0;
            }
        }
    }
    void arr_num_user()
    {
        cout << "Enter array numbers" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int a = 0;
                cin >> a;
                arr[i][j] = a;
            }
        }
    }
    void arr_num_rand()
    {
        srand(time(0));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i][j] = rand() % 100;
            }
        }
    }
    void print_matrix()
    {
        cout << "Matrix";
        for (int i = 0; i < n; i++)
        {
            cout << endl;
            for (int j = 0; j < m; j++)
                cout << arr[i][j] << " ";
        }
        cout << endl
             << endl;
    }
    void sum_matrix()
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                sum += arr[i][j];
            }
        }
        cout << "Sum:" << sum << endl;
        cout << endl;
    }
    void multiply_i_j()
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i][j] = i * j;
            }
        }
    }
    ~Matrix()
    {
        for (int i = 0; i < n; i++)
        {
            delete[] arr[i];
        }
        delete[] arr;
    }
};
int main()
{
    int n = 5;
    Matrix obj1;
    cout << "obj2" << endl;
    Matrix obj2(n);
    obj2.print_matrix();
    obj2.multiply_i_j();
    obj2.print_matrix();
    Matrix obj3(3, 4);
    cout << "obj3" << endl;
    obj3.print_matrix();
    obj3.arr_num_rand();
    obj3.print_matrix();
    obj3.sum_matrix();
    Matrix obj4(2, 3);
    cout << "obj4" << endl;
    obj4.print_matrix();
    obj4.arr_num_user();
    obj4.print_matrix();
}
