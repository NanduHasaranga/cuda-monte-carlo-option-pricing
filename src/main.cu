#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "../include/monte_carlo.cuh"

int main(){
    curandState *d_states;
    float *d_payoffs;
    // float *d_S0, *d_K, *d_r, *d_sigma, *d_T;

    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.10f;
    float sigma = 0.2f;
    float T = 1.0f;

    initConstants(S0, K, r, sigma, T);

    // cudaMalloc((void**)&d_S0, sizeof(float));
    // cudaMalloc((void**)&d_K, sizeof(float));
    // cudaMalloc((void**)&d_r, sizeof(float));
    // cudaMalloc((void**)&d_sigma, sizeof(float));
    // cudaMalloc((void**)&d_T, sizeof(float));

    // cudaMemcpy(d_S0, S0, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_K, K, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_r, r, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_sigma, sigma, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_T, T, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_states, NUM_SIMULATIONS * sizeof(curandState));
    cudaMalloc((void**)&d_payoffs, NUM_SIMULATIONS * sizeof(float));

    int blocks = (NUM_SIMULATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initRNG<<<blocks, BLOCK_SIZE>>>(d_states, 1234, NUM_SIMULATIONS);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA error: %s.\n", cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         

    cudaDeviceSynchronize();

    return 0;
}