# Monte Carlo Option Pricing with CUDA

This project prices a **European call option** using a GPU-accelerated Monte Carlo simulation in CUDA, then compares the result with the analytical Black-Scholes price.

## Overview

The implementation performs the following steps:

1. Initializes option parameters ($S_0$, $K$, $r$, $\sigma$, $T$) in device constant memory.
2. Initializes one CURAND RNG state per simulation path.
3. Simulates terminal stock prices under Geometric Brownian Motion.
4. Computes call payoffs on the GPU.
5. Reduces payoffs to a sum and computes discounted expected payoff.
6. Computes Black-Scholes call price on CPU for comparison.

## Mathematical Model

For each simulation path, terminal stock price is generated as:

$$
S_T = S_0 \cdot \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right), \quad Z \sim \mathcal{N}(0,1)
$$

European call payoff:

$$
\max(S_T - K, 0)
$$

Monte Carlo estimator:

$$
C \approx e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
$$

## Project Structure

```
include/
	black_scholes.h      # Black-Scholes function declaration
	monte_carlo.cuh      # CUDA kernels, constants, and API declarations
src/
	black_scholes.cpp    # Analytical Black-Scholes implementation
	monte_carlo.cu       # RNG init, path simulation, and reduction kernels
	main.cu              # Program entry point and workflow
ReadME.md
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (for `nvcc` and CURAND)
- C++ compiler compatible with your CUDA Toolkit
- Linux environment (project currently developed in Linux)

Verify CUDA compiler:

```bash
nvcc --version
```

## Build

From the repository root, compile with:

```bash
nvcc -O3 -std=c++14 -I./include src/main.cu src/monte_carlo.cu src/black_scholes.cpp -o src/option_pricer
```

## Run

```bash
./src/option_pricer
```

Typical output format:

```text
European Call Option Price (Monte Carlo): <value>
Black-Scholes Call Price: <value>
```

Because Monte Carlo uses random sampling, the Monte Carlo value may vary slightly between runs.

## Current Default Parameters

Defined in `src/main.cu`:

- $S_0 = 100.0$
- $K = 100.0$
- $r = 0.10$
- $\sigma = 0.20$
- $T = 1.0$

Simulation configuration is defined in `include/monte_carlo.cuh`:

- `NUM_SIMULATIONS = 1000000`
- `BLOCK_SIZE = 256`

## How to Experiment

1. Change option parameters in `src/main.cu`.
2. Optionally change simulation size and block size in `include/monte_carlo.cuh`.
3. Rebuild and rerun.

Increasing `NUM_SIMULATIONS` usually improves estimator stability but increases runtime.

## Implementation Notes

- Option parameters are stored in CUDA constant memory for fast read access.
- CURAND states are initialized with a fixed seed (`1234`) in the current code.
- Reduction uses one kernel pass plus final host-side accumulation.
- Error checks are performed after kernel launches with `CUDA_CHECK(cudaDeviceSynchronize())`.

## Troubleshooting

- **`nvcc: command not found`**
	Install CUDA Toolkit and ensure CUDA bin directory is on your `PATH`.

- **CUDA runtime errors (launch/memory/device)**
	Rebuild, verify GPU availability with `nvidia-smi`, and confirm toolkit-driver compatibility.

- **Large deviation from Black-Scholes price**
	Increase `NUM_SIMULATIONS` and verify parameter consistency.

## Next Improvements

- Add command-line arguments for option and simulation parameters.
- Implement full GPU reduction (multi-stage) to avoid host final accumulation.
- Add confidence interval reporting for Monte Carlo estimate.
- Extend to put options and Greeks.