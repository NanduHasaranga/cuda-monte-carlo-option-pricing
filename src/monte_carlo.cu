#include <math.h>
#include "../include/monte_carlo.cuh"

__constant__ float d_S0, d_K, d_r, d_sigma, d_T;

void initConstants(float S0, float K, float r, float sigma, float T){
    cudaMemcpyToSymbol(d_S0, &S0, sizeof(float));
    cudaMemcpyToSymbol(d_K, &K, sizeof(float));
    cudaMemcpyToSymbol(d_r, &r, sizeof(float));
    cudaMemcpyToSymbol(d_sigma, &sigma, sizeof(float));
    cudaMemcpyToSymbol(d_T, &T, sizeof(float));

}

__global__ void initRNG(curandState *states, unsigned long seed, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void monteCarloKernel(curandState *states, float * payoffs, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        float Z = curand_normal(&states[id]);
        float ST = d_S0 * exp((d_r - 0.5f * d_sigma * d_sigma) * d_T + d_sigma *sqrt(d_T) * Z);
        payoffs[id] = fmaxf(ST - d_K, 0.0f);
    }
}

__global__ void reduceSum(float *input, float *output, int N){
    __shared__ float cache[BLOCK_SIZE];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tid;

    cache[tid] = (id < N) ? input[id] : 0.0f;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = cache[0];
    }
}

float cpuReduce(float * d_array, int N){
    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    float *d_intermediate;
    cudaMalloc((void**)&d_intermediate, blocks * sizeof(float));

    reduceSum<<<blocks, threads>>>(d_array, d_intermediate, N);
    cudaDeviceSynchronize();

    float *h_intermediate = new float[blocks];
    cudaMemcpy(h_intermediate, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;

    for(int i = 0; i < blocks; i++){
        sum += h_intermediate[i];
    }

    delete[] h_intermediate;
    cudaFree(d_intermediate);
    return sum;
}