#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция для создания дерева турниров
void buildTournamentTree(int *array, int *tree, int index, int left, int right) {
    if (left == right) {
        tree[index] = array[left];
    } else {
        int mid = (left + right) / 2;
        buildTournamentTree(array, tree, 2 * index + 1, left, mid);
        buildTournamentTree(array, tree, 2 * index + 2, mid + 1, right);
        tree[index] = (tree[2 * index + 1] > tree[2 * index + 2]) ? tree[2 * index + 1] : tree[2 * index + 2];
    }
}

// Функция для вставки нового элемента в массив и перестройки дерева
void insertElement(int *array, int **tree, int left, int right, int position, int value, int *n) {
    // Смещаем элементы вправо, чтобы освободить место для нового элемента
    for (int i = *n; i > position; --i) {
        array[i] = array[i - 1];
    }

    array[position] = value; // Вставляем новый элемент
    (*n)++; // Увеличиваем размер массива

    // Перевыделяем память под дерево турниров
    int height = (int)ceil(log2(*n));
    int maxSize = 2 * (int)pow(2, height) - 1;
    *tree = realloc(*tree, maxSize * sizeof(int));

    // Перестраиваем дерево
    buildTournamentTree(array, *tree, 0, 0, *n - 1);
}

// Функция для удаления элемента из дерева турниров
void deleteElement(int *array, int **tree, int left, int right, int position, int *n) {
    // Элемент исчезает из массива, смещаем остальные элементы
    for (int i = position; i < *n - 1; ++i) {
        array[i] = array[i + 1];
    }
    (*n)--; // Уменьшаем размер массива

    // Перевыделяем память под дерево турниров
    int height = (int)ceil(log2(*n));
    int maxSize = 2 * (int)pow(2, height) - 1;
    *tree = realloc(*tree, maxSize * sizeof(int));

    // Перестраиваем дерево
    buildTournamentTree(array, *tree, 0, 0, *n - 1);
}

// Функция для поиска элемента в массиве
int findIndexInArray(int *array, int n, int value) {
    for (int i = 0; i < n; i++) {
        if (array[i] == value) {
            return i; // Возвращаем индекс элемента в исходном массиве
        }
    }
    return -1; // Элемент не найден
}
// Функция для вывода элементов массива
void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int array[100] = {5, 3, 9, 1, 6, 4, 8, 7}; // Увеличен размер для демонстрации вставки
    int n = 8; // Начальный размер массива
    int height = (int)ceil(log2(n));
    int maxSize = 2 * (int)pow(2, height) - 1;
    int *tree = (int *)malloc(maxSize * sizeof(int));

    clock_t start, end;
    double cpu_time_used;

    // Измерение времени для функции buildTournamentTree
    start = clock();
    buildTournamentTree(array, tree, 0, 0, n - 1);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения Построения дерева турниров: %f секунд\n", cpu_time_used);

    printf("Массив: ");
    printArray(array, n);
    printf("Дерево турниров: ");
    printArray(tree, maxSize);
    printf("Победитель: %d\n", tree[0]); // Корень дерева содержит победителя

    // Задаем параметры для вставки и удаления
    int insertPosition = 2;
    int insertValue = 10;
    int deletePosition = 2;
    int searchValue = 9; // Измените это значение для тестирования

    // Измерение времени для функции insertElement
    start = clock();
    insertElement(array, &tree, 0, n - 1, insertPosition, insertValue, &n);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения Вставки элемента: %f секунд\n", cpu_time_used);

    printf("Массив после вставки: ");
    printArray(array, n);
    printf("Дерево турниров после вставки: ");
    printArray(tree, maxSize); // Вывод дерева турниров после вставки

    // Измерение времени для функции deleteElement
    start = clock();
    deleteElement(array, &tree, 0, n - 1, deletePosition, &n);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения Удаления элемента: %f секунд\n", cpu_time_used);

    printf("Массив после удаления: ");
    printArray(array, n);
    printf("Дерево турниров после удаления: "); // Вывод дерева турниров после удаления
    printArray(tree, maxSize);

    // Измерение времени для функции findIndexInArray
    start = clock();
    int searchIndex = findIndexInArray(array, n, searchValue);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Время выполнения Поиска в массиве: %f секунд\n", cpu_time_used);

    if (searchIndex != -1) {
        printf("Элемент %d найден в позиции %d в исходном массиве.\n", searchValue, searchIndex);
    } else {
        printf("Элемент %d не найден в исходном массиве.\n", searchValue);
    }

    free(tree);
    return 0;
}

