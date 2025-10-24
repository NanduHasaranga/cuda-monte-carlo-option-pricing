#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

#include <curand_kernel.h>

#define NUM_SIMULATIONS 1000000
#define BLOCK_SIZE 256

void initConstants(float S0, float K, float r, float sigma, float T);

__global__ void initRNG(curandState *states, unsigned long seed, int N);

__global__ void monteCarloKernel(curandState *states, float * payoffs, int N);

__global__ void reduceSum(float *input, float *output, int N);

float cpuReduce(float * d_array, int N);

#endif