#include <iostream>
#include <random>
#include <utility>

// 1) Функция быстрого возведения в степень по модулю
long long fast_power_mod(long long a, long long x, long long p) {
    long long result = 1;
    a = a % p;

    while (x > 0) {
        if (x % 2 == 1) {
            result = (result * a) % p;
        }
        a = (a * a) % p;
        x = x / 2;
    }

    return result;
}

// 2) Функция теста простоты Ферма
bool fermat_primality_test(long long n, int k = 5) {
    if (n <= 1 || n == 4) return false;
    if (n <= 3) return true;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long long> dis(2, n - 2);

    for (int i = 0; i < k; i++) {
        long long a = dis(gen);
        if (fast_power_mod(a, n - 1, n) != 1) {
            return false;
        }
    }

    return true;
}

// 3) Функция расширенного алгоритма Евклида
std::pair<long long, std::pair<long long, long long>> extended_euclidean(long long a, long long b) {
    if (b == 0) {
        return {a, {1, 0}};
    }

    auto result = extended_euclidean(b, a % b);
    long long gcd = result.first;
    long long x1 = result.second.first;
    long long y1 = result.second.second;

    long long x = y1;
    long long y = x1 - (a / b) * y1;

    return {gcd, {x, y}};
}

// Функция для генерации случайного числа в диапазоне
long long generate_random(long long min, long long max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long long> dis(min, max);
    return dis(gen);
}

// Функция для генерации простого числа
long long generate_prime(long long min, long long max) {
    long long candidate;
    do {
        candidate = generate_random(min, max);
    } while (!fermat_primality_test(candidate));
    return candidate;
}

// Основная функция с меню
int main() {
    int choice;

    do {
        std::cout << "\n=== Криптографическая библиотека ===" << std::endl;
        std::cout << "1. Быстрое возведение в степень по модулю" << std::endl;
        std::cout << "2. Тест простоты Ферма" << std::endl;
        std::cout << "3. Расширенный алгоритм Евклида" << std::endl;
        std::cout << "4. Выход" << std::endl;
        std::cout << "Выберите опцию: ";
        std::cin >> choice;

        switch (choice) {
            case 1: {
                long long a, x, p;
                std::cout << "Введите a, x, p: ";
                std::cin >> a >> x >> p;
                long long result = fast_power_mod(a, x, p);
                std::cout << a << "^" << x << " mod " << p << " = " << result << std::endl;
                break;
            }

            case 2: {
                int sub_choice;
                std::cout << "1. Ввести число с клавиатуры" << std::endl;
                std::cout << "2. Сгенерировать случайное число" << std::endl;
                std::cout << "3. Сгенерировать простое число" << std::endl;
                std::cout << "Выберите опцию: ";
                std::cin >> sub_choice;

                if (sub_choice == 1) {
                    long long n;
                    std::cout << "Введите число: ";
                    std::cin >> n;
                    bool is_prime = fermat_primality_test(n);
                    std::cout << n << (is_prime ? " - вероятно простое" : " - составное") << std::endl;
                } else if (sub_choice == 2) {
                    long long n = generate_random(2, 1000000000);
                    std::cout << "Сгенерировано число: " << n << std::endl;
                    bool is_prime = fermat_primality_test(n);
                    std::cout << n << (is_prime ? " - вероятно простое" : " - составное") << std::endl;
                } else if (sub_choice == 3) {
                    long long prime = generate_prime(100000000, 1000000000);
                    std::cout << "Сгенерировано простое число: " << prime << std::endl;
                }
                break;
            }

            case 3: {
                int sub_choice;
                std::cout << "1. Ввести a и b с клавиатуры" << std::endl;
                std::cout << "2. Сгенерировать a и b" << std::endl;
                std::cout << "3. Сгенерировать простые a и b" << std::endl;
                std::cout << "Выберите опцию: ";
                std::cin >> sub_choice;

                long long a, b;
                if (sub_choice == 1) {
                    std::cout << "Введите a и b: ";
                    std::cin >> a >> b;
                } else if (sub_choice == 2) {
                    a = generate_random(1, 1000000000);
                    b = generate_random(1, 1000000000);
                    std::cout << "Сгенерированы числа: a = " << a << ", b = " << b << std::endl;
                } else if (sub_choice == 3) {
                    a = generate_prime(100000000, 1000000000);
                    b = generate_prime(100000000, 1000000000);
                    std::cout << "Сгенерированы простые числа: a = " << a << ", b = " << b << std::endl;
                }

                auto result = extended_euclidean(a, b);
                long long gcd = result.first;
                long long x = result.second.first;
                long long y = result.second.second;

                std::cout << "НОД(" << a << ", " << b << ") = " << gcd << std::endl;
                std::cout << "Коэффициенты: x = " << x << ", y = " << y << std::endl;
                std::cout << "Проверка: " << a << "*" << x << " + " << b << "*" << y << " = " << (a*x + b*y) << std::endl;
                break;
            }

            case 4:
                std::cout << "Выход..." << std::endl;
                break;

            default:
                std::cout << "Неверный выбор!" << std::endl;
        }
    } while (choice != 4);

    return 0;
}
