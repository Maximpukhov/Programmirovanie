#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define M_PI 3.14159265358979323846

double func(double x, double y) {
    return exp(x - y);
}

double getrand(double min, double max, unsigned int *seed) {
    return min + (max - min) * ((double)rand_r(seed) / RAND_MAX);
}

int main(int argc, char **argv) {
    const int n_values[2] = {10000000, 100000000}; // n = 10^7 и n = 10^8
    for (int n_index = 0; n_index < 2; n_index++) {
        const int n = n_values[n_index];
        printf("Численное интегрирование методом Монте-Карло: n = %d\n", n);

        unsigned int main_seed = 12345;
        double t = omp_get_wtime();
        double s = 0.0;
        int in = 0;

        // Однопоточное вычисление
        for (int i = 0; i < n; i++) {
            double x = getrand(-1, 0, &main_seed);
            double y = getrand(0, 1, &main_seed);
            s += func(x, y);
            in++;
        }
        double res_single = M_PI * s / in;
        double time_single = omp_get_wtime() - t;
        printf("Однопоточный результат: %.12f, время: %.6f сек\n", res_single, time_single);

        // Многопоточные вычисления
        for (int num_threads = 2; num_threads <= 8; num_threads += 2) {
            omp_set_num_threads(num_threads);
            t = omp_get_wtime();
            s = 0.0;
            in = 0;

            #pragma omp parallel
            {
                double s_loc = 0;
                int in_loc = 0;
                unsigned int seed = main_seed;
                #pragma omp for nowait
                for (int i = 0; i < n; i++) {
                    double x = getrand(0, 1, &seed);
                    double y = getrand(2, 5, &seed);
                    s_loc += func(x, y);
                    in_loc++;
                }
                #pragma omp atomic
                s += s_loc;
                #pragma omp atomic
                in += in_loc;
            }
            double res = M_PI * s / in;
            double time_multi = omp_get_wtime() - t;
            double speedup = time_single / time_multi;
            printf("Результат: %.12f, n = %d, Количество потоков: %d, Время (сек.): %.6f, Ускорение: %.4lf\n", res, n, num_threads, time_multi, speedup);
        }
    }

    return 0;
}
