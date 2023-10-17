#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <iomanip>

using namespace std;

template <typename T>
class Matrix
{
private:
    int n;
    T sum;
    int m;
    T **arr;
    T i_get;
    T j_get;
    T num;
    void create_matrix(int n, int m)
    {
        arr = (T **)new T *[n];
        for (int i = 0; i < n; i++)
        {
            arr[i] = (T *)new T[m];
        }
    }

public:
    Matrix() // конструктор класса матрица 0
    {
        create_matrix(0, 0);
    }
    Matrix(T n) // конструктор класса матрица квадрат единичной
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
    Matrix(T n, T m) // конструктор класса матрица указ размерность
    {
        this->n = n;
        this->m = m;
        this->sum = 0;
        this->i_get = 0;
        this->j_get = 0;
        this->num = 0;
        create_matrix(n, m);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i][j] = 0;
            }
        }
    }
    Matrix(const Matrix &p) // конструктор копирования
    {
        n = p.n;
        sum = p.sum;
        m = p.m;
        arr = (T **)new T *[n];
        for (int i = 0; i < n; i++)
        {
            arr[i] = (T *)new T[m];
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                arr[i][j] = p.arr[i][j];
            }
        }
        i_get = p.i_get;
        j_get = p.j_get;
        num = p.num;
    }
    Matrix &operator=(const Matrix &matrix)
    {
        if (&matrix != this)
        {
            n = matrix.n;
            m = matrix.m;
            for (int i = 0; i < n; i++)
            {
                delete[] arr[i];
            }
            delete[] arr;
            arr = (T **)new T *[n];
            for (int i = 0; i < n; i++)
            {
                arr[i] = (T *)new T[m];
            }
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    arr[i][j] = matrix.arr[i][j];
                }
            }
            sum = matrix.sum;
            i_get = matrix.i_get;
            j_get = matrix.j_get;
            num = matrix.num;
        }
        return *this;
    }
    void arr_num_user()
    {
        cout << "Enter array numbers" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                T a = 0;
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
    void GetM(T i_get, T j_get)
    {
        T count = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if ((i == i_get) && (j = j_get))
                {
                    cout << arr[i][j] << endl;
                    count++;
                }
            }
        }
        if (count == 0)
        {
            cout << "Unknown numbers" << endl;
        }
    }
    void SetM(int i, int j, T num)
    {
        if ((i < 0) || (i >= n))
            return;
        if ((j < 0) || (j >= m))
            return;
        arr[i][j] = num;
    }
    void prT_matrix()
    {
        cout << "Matrix";
        for (int i = 0; i < n; i++)
        {
            cout << endl;
            for (int j = 0; j < m; j++)
                cout << setw(4) << arr[i][j];
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
    // тест для класса MATRIX
    Matrix<int> obj1(3, 4);
    cout << "obj1" << endl;
    obj1.prT_matrix();
    // Заполнить матрицу значеннями по формуле
    int i, j;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            obj1.SetM(i, j, i + j);
        }
    }
    cout << "obj1" << endl;
    obj1.prT_matrix();
    Matrix<int> obj2 = obj1; // вызов конструктора копирования
    cout << "obj2" << endl;
    obj2.prT_matrix();
    Matrix<int> obj3; // вызов оператора копирования - проверка
    obj3 = obj1;
    cout << "obj3" << endl;
    obj3.prT_matrix();
    Matrix<int> obj4;
    obj4 = obj3 = obj2 = obj1; // вызов оператора копирования в виде "цепочки"
    cout << "obj4" << endl;
    obj4.prT_matrix();
}
