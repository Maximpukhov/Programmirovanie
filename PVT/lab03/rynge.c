#include <stdio.h>
#include <math.h>
#include <omp.h>

double func(double x) {
    return (sqrt(x*(3-x))/(x+1));
}

int main(int argc, char **argv) {
    const double eps = 1E-5;
    const double a = 1.0;
    const double b = 1.2;
    const int n0 = 100000000;
    printf("num integ: [%f, %f], n0 = %d, EPS = %f\n", a, b, n0, eps);
    double sq[2];
    double t, time_serial;

    // Вычисление времени в одном потоке
    omp_set_num_threads(1);
    t = omp_get_wtime();
    int n = n0, k;
    double delta = 1;
    for (k = 0; delta > eps; n *= 2, k ^= 1) {
        double h = (b - a) / n;
        double s = 0.0;
        sq[k] = 0;
        for (int i = 0; i < n; i++) {
            s += func(a + h * (i + 0.5));
        }
        sq[k] += s * h;
        if (n > n0)
            delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
    }
    time_serial = omp_get_wtime() - t;
    printf("Single-threaded result: %.12f; time: %.6f sec\n", sq[k ^ 1], time_serial);

    // Цикл по разным количествам потоков
    for (int num_threads = 2; num_threads <= 8; num_threads += 2) {
        omp_set_num_threads(num_threads);
        t = omp_get_wtime();
        #pragma omp parallel
        {
            int n = n0, k;
            double delta = 1;
            for (k = 0; delta > eps; n *= 2, k ^= 1) {
                double h = (b - a) / n;
                double s = 0.0;
                sq[k] = 0;

                #pragma omp barrier

                #pragma omp for nowait
                for (int i = 0; i < n; i++)
                    s += func(a + h * (i + 0.5));

                #pragma omp atomic
                sq[k] += s * h;

                #pragma omp barrier
                if (n > n0)
                    delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
            }

            #pragma omp master
            printf("Result: %.12f; Rule Runge: EPS %e, n %d\n", sq[k ^ 1], eps, n / 2);
        }
        t = omp_get_wtime() - t;
        double speedup = time_serial / t;
        printf("Number of threads: %d, Time (sec.): %.6f, Speedup: %.3lf\n", num_threads, t, speedup);
    }
    return 0;
}
