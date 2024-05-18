#include <stdio.h>
#include <math.h>
#include <omp.h>

double func(double x) {
    return (pow(x,4))/(0.5*x*x+x+6);
}

int main(int argc, char **argv) {
    const double eps = 1E-5;
    const double a = 0.4;
    const double b = 1.5;
    const int n0 = 100000000;
    printf("num integ: [%f, %f], n0 = %d, EPS = %f\n", a, b, n0, eps);
    double sq[2];
    double t;
    double time_pos = 3.652000;
    double speedup;
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

                // Ждем, пока все потоки обнулят sq[k] и s
                #pragma omp barrier

                #pragma omp for nowait
                for (int i = 0; i < n; i++)
                    s += func(a + h * (i + 0.5));

                #pragma omp atomic
                sq[k] += s * h;

                // Ждем, пока все потоки обновят sq[k]
                #pragma omp barrier
                if (n > n0)
                    delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
            }

            #pragma omp master
            printf("res Pi: %.12f; rule Runge: EPS %e, n %d\n", sq[k] * sq[k], eps, n / 2);
        }
        t = omp_get_wtime() - t;
        speedup = time_pos/t;
        printf("num  threads: %d, time (sec.): %.6f, speedup: %.3lf\n", num_threads, t,speedup);
    }
    return 0;
}
