
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)


__global__ void saxpy(float a, float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
} 

int main(int argc, char *argv[]) {

    int n = std::atoi(argv[1]);

    if (n == 0) {
        printf("SUM=0.0f\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] =  (i % 1024) - 512;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(2.0f, d_x, d_y, n);

    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += h_y[i];
    printf("SUM=%f\n", sum);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    free(h_x);
    free(h_y);
}