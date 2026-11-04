# FastLinAlg.jl

[![Build Status](https://github.com/ojwoodford/FastLinAlg.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/FastLinAlg.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/FastLinAlg.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/FastLinAlg.jl)

SIMD-accelerated, zero-allocation linear algebra primitives in Julia, avoiding BLAS thread wakeup and call overhead for small and medium vectors.

## Why FastLinAlg.jl?

Standard Julia linear algebra is built around standard BLAS and LAPACK libraries via `LinearAlgebra.jl`. While BLAS is heavily optimized for large-scale operations (such as dense matrix-matrix multiplication on large arrays), it introduces significant overhead in performance-critical inner loops involving small- to medium-sized data:

1. **BLAS Call and Thread-Checking Overhead**:
   Standard routines such as `LinearAlgebra.norm` invoke external C/Fortran BLAS subroutines (`dnrm2`). On every call, this incurs `ccall` overhead, foreign function interface boundary transitions, argument marshalling, and multithreading synchronization checks. For small to medium vectors, this administrative overhead exceeds the cost of the actual computation. Native Julia SIMD execution via `LoopVectorization.@turbo` executes directly in CPU registers without leaving the Julia runtime.

2. **Over-Defensive Multi-Pass Scaling in Standard Norms**:
   `LinearAlgebra.norm` uses BLAS `dnrm2` (or `generic_norm2`), which executes defensive multi-pass scaling (e.g., Blue's algorithm) with extra conditional branches and normalization factors to protect against extreme underflow or overflow. When data magnitudes are well within normal IEEE-754 dynamic range, this defensive handling adds unnecessary overhead. `fast_norm` and `sqnorm` compute norms directly via SIMD vectorization in a single pass, delivering **10x–17x speedups** over `LinearAlgebra.norm`.

## Performance Evaluation

A benchmark script is provided in [`examples/speed_test.jl`](examples/speed_test.jl). It times all included functions against standard implementations across multiple input sizes using warmup cycles, minimum-time filtering over multiple trials, and compiler barriers to prevent constant folding.

To run the speed test:

```bash
julia --project=. examples/speed_test.jl
```

### Benchmark Results

Results collected on an Apple Silicon (M-series) CPU:

#### 1. Euclidean Norm (`fast_norm` vs `LinearAlgebra.norm`)

| Vector Size | Standard (`LinearAlgebra.norm`) | FastLinAlg (`fast_norm`) | Speedup | Time Reduction |
|:-----------:|:-------------------------------:|:------------------------:|:-------:|:--------------:|
| 4 | ~5.3 ns | ~1.9 ns | ~2.8x | ~64.7% |
| 16 | ~12.8 ns | ~2.8 ns | ~4.6x | ~78.2% |
| 64 | ~60.7 ns | ~5.3 ns | ~11.6x | ~91.3% |
| 256 | ~276.0 ns | ~17.2 ns | ~16.1x | ~93.8% |
| 1024 | ~1.23 μs | ~71.0 ns | ~17.4x | ~94.2% |
| 4096 | ~5.08 μs | ~312.3 ns | ~16.3x | ~93.9% |

#### 2. Squared Euclidean Norm (`sqnorm` vs `LinearAlgebra.dot(a, a)`)

| Vector Size | Standard (`dot(a, a)`) | FastLinAlg (`sqnorm`) | Speedup | Time Reduction |
|:-----------:|:----------------------:|:---------------------:|:-------:|:--------------:|
| 4 | ~9.3 ns | ~1.9 ns | ~5.0x | ~80.1% |
| 16 | ~13.0 ns | ~2.8 ns | ~4.7x | ~78.6% |
| 64 | ~14.9 ns | ~11.1 ns | ~1.3x | ~25.5% |
| 256 | ~39.5 ns | ~15.8 ns | ~2.5x | ~60.2% |
| 1024 | ~151.5 ns | ~75.2 ns | ~2.0x | ~50.4% |
| 4096 | ~588.6 ns | ~334.0 ns | ~1.8x | ~43.3% |

### Key Takeaways

- **Over 16x Faster Norm Computation**: `fast_norm` avoids multi-pass underflow/overflow scaling in BLAS `dnrm2`, reducing Euclidean norm latency by over 93% on vectors of length 64 to 4096.
- **Zero Foreign Function Overhead**: `sqnorm` avoids BLAS library call and threading setup costs, offering ~2x–5x lower latency for small vectors and consistently lower time across all dimensions.

## Externally Available Methods

### `sqnorm`

```julia
sqnorm(x::Number) -> Number
sqnorm(vec::AbstractVector) -> eltype(vec)
```

Computes the squared L2 norm $x^2$ (for a `Number`) or $\sum_i v_i^2$ (for an `AbstractVector`) using `LoopVectorization.@turbo`.

### `fast_norm`

```julia
fast_norm(vec::AbstractVector) -> Real
```

Computes the Euclidean L2 norm $\sqrt{\text{sqnorm}(\text{vec})}$ directly in Julia without calling BLAS.

## Example

```julia
using FastLinAlg

a = randn(100)

s = sqnorm(a)
n = fast_norm(a)
```
