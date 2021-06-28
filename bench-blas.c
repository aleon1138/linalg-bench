#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <time.h>

int N = 10000;
int DIMS[] = {4, 6, 8, 10, 14, 20, 27, 38, 52, 71, 98, 135, 186, 256};
int N_DIMS = sizeof(DIMS)/sizeof(DIMS[0]);


float * alloc(int size)
{
    float *x = (float*)malloc(size*sizeof(float));
    for (int i = 0; i < size; ++i) {
        x[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
    }
    return x;
}

long nanotime()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main()
{
    openblas_set_num_threads(1);

    for (int i = 0; i < N_DIMS; ++i) {
        int n = DIMS[i];

        float *a = alloc(n);
        float *b = alloc(n);

        long t0 = nanotime();
        for (int i = 0; i < N; ++i) {
            cblas_sdot(n, a, 1, b, 1);
        }
        long dt = (nanotime()-t0)/N;

        printf("cblas,dot,%d,%ld\n",n,dt);

        free(a);
        free(b);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        int n = DIMS[i];

        float *a = alloc(n*n);
        float *b = alloc(n);
        float *c = alloc(n);

        long t0 = nanotime();
        for (int i = 0; i < N; ++i) {
            cblas_sgemv(CblasColMajor, CblasNoTrans,n,n,1.0f,a,n,b,1, 0.0f,c,1);
        }
        long dt = (nanotime()-t0)/N;

        printf("cblas,gemv,%d,%ld\n",n,dt);

        free(a);
        free(b);
        free(c);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        int n = DIMS[i];

        float *a = alloc(n*n);
        float *b = alloc(n*n);
        float *c = alloc(n*n);

        long t0 = nanotime();
        for (int i = 0; i < N; ++i) {
            cblas_sgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, n,n,n, 1.0f,a,n, b,n, 0.0f,c,n);
        }
        long dt = (nanotime()-t0)/N;

        printf("cblas,gemm,%d,%ld\n",n,dt);

        free(a);
        free(b);
        free(c);
    }

    for (int i = 0; i < N_DIMS; ++i) {
        int n = DIMS[i];

        float *a = alloc(n);
        float *b = alloc(n);
        float *c = alloc(n*n);

        long t0 = nanotime();
        for (int i = 0; i < N; ++i) {
            cblas_sger(CblasColMajor, n,n, 1.0f,a,1, b,1, c,n);
        }
        long dt = (nanotime()-t0)/N;

        printf("cblas,ger,%d,%ld\n",n,dt);

        free(a);
        free(b);
        free(c);
    }
    return 0;
}
