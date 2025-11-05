#include "../include/black_scholes.h"
#include <cmath>

static float normalCDF(float x){
    return 0.5f * erfcf(-x / sqrtf(2.0f));
}

float blackScholesCall(float S0, float K, float r, float sigma, float T){
    float sqrtT = sqrtf(T);

    float d1 = (logf(S0 / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtT);

    float d2 = d1 - sigma * sqrtT;

    return S0 * normalCDF(d1) - K * expf(-r * T) * normalCDF(d2);
}