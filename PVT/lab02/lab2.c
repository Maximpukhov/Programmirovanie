#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>
#include <sys/time.h>

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
                for (int j = 0; j < n; j++)
                    c[i] += a[i * n + j] * b[j];
        }
    }
}

void matrix_vector_product_serial(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

double wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

void run_parallel(int m, int n, int num_threads)
{
    double *a, *b, *c;

    // Allocate memory for 2-d array a[m, n]
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t_serial = wtime();
    matrix_vector_product_serial(a, b, c, m, n);
    t_serial = wtime() - t_serial;

    omp_set_num_threads(num_threads);
    double t_parallel = wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t_parallel = wtime() - t_parallel;

    double speedup = t_serial / t_parallel;
    printf("Elapsed time (serial) %.6f sec.\n", t_serial);
    printf("Elapsed time (parallel, %d threads): %.6f sec.\n", num_threads, t_parallel);
    printf("speedup: %.6lf\n",speedup);
    free(a);
    free(b);
    free(c);
}


int main(int argc, char **argv)
{
    int m[] = {15000, 20000, 25000};
    int n[] = {15000, 20000, 25000};
    int num_threads_values[] = {2, 4, 6, 8};

    for (int i = 0; i < 3; i++) {

        if (i>=1)
            printf("\n");
        printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m[i], n[i]);
        printf("Memory used: %" PRIu64 " MiB\n", ((m[i] * n[i] + m[i] + n[i]) * sizeof(double)) >> 20);

        for (int j = 0; j < 4; j++) {
            int num_threads = num_threads_values[j];
            run_parallel(m[i], n[i], num_threads);
        }
    }

    return 0;
}
