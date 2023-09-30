#include <iostream>
#include <cstdlib>
#include <iomanip>

using namespace std;

int **create_matrix(int n)
{
    int **arr = new int *[n];
    for (int i = 0; i < n; i++)
    {
        arr[i] = new int[n];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = rand() % 10;
        }
    }
    return arr;
}

void print_matrix(int **arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << endl;
        for (int j = 0; j < n; j++)
            cout << arr[i][j] << " ";
    }
    cout << endl
         << endl;
}

int *A(int **matrix, int *arr, int size)
{
    int n = size - 1;
    int i, j, count = 0;
    while (n >= 0)
    {
        i = 0;
        j = n--;
        while (j < size)
        {
            arr[count] = matrix[i][j];
            count++;
            i++;
            j++;
        }
    }
    n = 1;
    while (n < size)
    {
        i = n++;
        j = 0;
        while (i < size)
        {
            arr[count] = matrix[i][j];
            count++;
            i++;
            j++;
        }
    }
    return arr;
}

int *B(int **matrix, int *arr, int size)
{
    int n = 0;
    int i, j, count = 0;
    while (n < size)
    {
        j = 0;
        i = n++;
        while (i >= 0)
        {
            arr[count] = matrix[i][j];
            count++;
            j++;
            i--;
        }
    }
    n = 1;
    while (n < size)
    {
        i = size - 1;
        j = n++;
        while (j < size)
        {
            arr[count] = matrix[i][j];
            count++;
            i--;
            j++;
        }
    }
    return arr;
}

int *C(int **matrix, int *arr, int n)
{
    int shift, count, i, j;
    shift = count = i = j = 0;

    while (count < n * n)
    {
        arr[n * n - count - 1] = matrix[i][j];
        if (i == shift && j < n - shift - 1)
            j++;
        else if (j == n - shift - 1 && i < n - shift - 1)
            i++;
        else if (i == n - shift - 1 && j > shift)
            j--;
        else
            i--;

        if ((i == shift + 1) && (j == shift) && (shift != n - shift - 1))
        {
            shift++;
        }

        count++;
    }
    return arr;
}

int *D(int **matrix, int *arr, int n)
{
    int shift, count, i, j;
    shift = count = i = j = 0;

    while (count < n * n)
    {
        arr[count] = matrix[i][j];
        if (i == shift && j < n - shift - 1)
            j++;
        else if (j == n - shift - 1 && i < n - shift - 1)
            i++;
        else if (i == n - shift - 1 && j > shift)
            j--;
        else
            i--;

        if ((i == shift + 1) && (j == shift) && (shift != n - shift - 1))
        {
            shift++;
        }

        count++;
    }

    return arr;
}

void print_arr_dynamic(int *arr, int n)
{
    for (int i = 0; i < n * n; i++)
        cout << arr[i] << " ";
}

void delete_matrix(int **arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

void delete_array(int *arr)
{
    delete[] arr;
}

int **create_matrix_rand_str(int n)
{
    srand(time(0));
    int **arr = new int *[n];
    for (int i = 0; i < n; i++)
    {
        int size = rand() % 20;
        arr[i] = new int[size + 1];
        arr[i][0] = size;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 1; j <= arr[i][0]; j++)
        {
            arr[i][j] = rand() % 20;
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
    return arr;
}

void print_matrix_rand_str(int **arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << arr[i][0] << ": ";
        for (int j = 1; j <= arr[i][0]; j++)
            cout << arr[i][j] << " ";
        cout << endl;
    }
}


int main()
{
    srand(time(0));
    int n = 5;
    int *arr_el = new int[n * n];
    int **arr = create_matrix(n);
    print_matrix(arr, n);
    A(arr, arr_el, n);
    print_arr_dynamic(arr_el, n);
    cout << endl;
    B(arr, arr_el, n);
    print_arr_dynamic(arr_el, n);
    cout << endl;
    C(arr, arr_el, n);
    print_arr_dynamic(arr_el, n);
    cout << endl;
    D(arr, arr_el, n);
    print_arr_dynamic(arr_el, n);
    cout << endl;
    delete_matrix(arr, n);
    delete_array(arr_el);
    cout << endl;
    int **mat_rand_str = create_matrix_rand_str(n);
    cout << endl;
    print_matrix_rand_str(arr, n);
    delete_matrix(mat_rand_str,n);
}
