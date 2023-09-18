#include <iostream>
#include <cstdlib>

using namespace std;

int *getRandArray(int siz, int maxvalue)
{
    siz++;
    int *arr = new int[siz];
    arr[0] = siz - 1;
    for (int i = 1; i < siz; i++)
    {
        arr[i] = rand() % maxvalue;
    }
    return arr;
}
int **genRandMatrix(int size, int maxvalue)
{
    int **arr = new int *[size];
    for (int i = 0; i < size; i++)
    {
        arr[i] = getRandArray(rand() % 10, maxvalue);
    }
    return arr;
}
void print(int *arr)
{
    cout << arr[0] << ": ";
    for (int i = 1; i <= arr[0]; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
}
void printMatrix(int **arr, int size)
{
    cout << size << endl;
    for (int i = 0; i < size; i++)
    {
        print(arr[i]);
    }
}
int main()
{
    srand(time(0));
    int siz = rand() % 10;
    int maxvalue = 100;
    int *arr = getRandArray(siz, maxvalue);
    print(arr);
    cout << endl;
    delete[] arr;
    siz = rand() % 10;
    int **arr_matrix = genRandMatrix(siz, maxvalue);
    printMatrix(arr_matrix, siz);
    for (int i = 0; i < siz; i++)
    {
        delete[] arr_matrix[i];
    }
    delete[] arr_matrix;
    return 0;
}
