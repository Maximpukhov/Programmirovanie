#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<inttypes.h>
#include <stdlib.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
	for (int i = 0; i < m; i++) {
		c[i] = 0.0;
		for (int j = 0; j < n; j++)
			c[i] += a[i * n + j] * b[j];
	}
}

double run_serial(int m, int n)
{
	double *a, *b, *c;
	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;
	}
	for (int j = 0; j < n; j++)
		b[j] = j;
	double t = omp_get_wtime();
	matrix_vector_product(a, b, c, m, n);
	t = omp_get_wtime() - t;
	printf("Elapsed time (serial): %.6f sec.\n", t);
	free(a);
	free(b);
	free(c);
	return t;
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int j)
{
	#pragma omp parallel num_threads(j)
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

double run_parallel(int m, int n, int j)
{
	double *a, *b, *c;
	// Allocate memory for 2-d array a[m, n]
	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;
	}
	for (int j = 0; j < n; j++)
		b[j] = j;
	double t = omp_get_wtime();
	matrix_vector_product_omp(a, b, c, m, n, j);
	t = omp_get_wtime() - t;
	printf("Elapsed time (parallel): %.6f sec.\n", t);
	free(a);
	free(b);
	free(c);

	return t;
}


int main(int argc, char **argv)
{
	FILE *f = fopen("result.csv", "w");
	for(int i = 15000; i<=25000; i+=5000)
    //for(int i = 15000; i<=15000; i+=5000)
	{
		for(int j = 2; j<=8;j+=2)
        //for(int j = 2; j<=2;j+=2)
		{
			int m = i;
			int n = i;
			double tp = 0;
			double ts = 0;
			printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
			printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);

			
				tp = run_parallel(m, n, j);
			
			ts = run_serial(m, n);
            // if(ts!= 0.0 && tp!= 0.0)
            // {
            double res = ts/tp;
            char string[50]; 
            sprintf(string, "%d;%.6f\n", j, res);
            fputs(string, f);
            //}
            // else
            // {
            //     return -1;
            // }
		}
	}
	fclose(f);
	return 0;
}
