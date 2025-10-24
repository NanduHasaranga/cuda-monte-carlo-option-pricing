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