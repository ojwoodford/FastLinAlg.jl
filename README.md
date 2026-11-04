# FastLinAlg.jl

[![Build Status](https://github.com/ojwoodford/FastLinAlg.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/FastLinAlg.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/FastLinAlg.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/FastLinAlg.jl)

SIMD-accelerated, zero-allocation linear algebra primitives in Julia, avoiding BLAS thread wakeup and call overhead for small and medium vectors and matrices.

## Why FastLinAlg.jl?

Standard Julia linear algebra is built around standard BLAS and LAPACK libraries via `LinearAlgebra.jl`, and sparse linear algebra via `SparseArrays.jl`. While BLAS is heavily optimized for large-scale operations (such as dense matrix-matrix multiplication on large arrays), it introduces significant overhead in performance-critical inner loops involving small- to medium-sized data:

1. **BLAS Call and Thread-Checking Overhead**:
   Standard routines such as `LinearAlgebra.dot` and `LinearAlgebra.norm` invoke external C/Fortran BLAS subroutines (`ddot`, `dnrm2`). On every call, this incurs `ccall` overhead, foreign function interface boundary transitions, argument marshalling, and multithreading synchronization checks. For small to medium vectors ($N \le 1000$), this administrative overhead exceeds the cost of the actual computation. Native Julia SIMD execution via `LoopVectorization.@turbo` executes directly in CPU registers without leaving the Julia runtime, delivering up to **3.9x speedups** on small vector dot products.

2. **Over-Defensive Multi-Pass Scaling in Standard Norms**:
   `LinearAlgebra.norm` uses BLAS `dnrm2` (or `generic_norm2`), which executes defensive multi-pass scaling (e.g., Blue's algorithm) with extra conditional branches and normalization factors to protect against extreme underflow or overflow. When data magnitudes are well within normal IEEE-754 dynamic range, this defensive handling adds unnecessary overhead. `fast_norm` and `sqnorm` compute norms directly via SIMD vectorization in a single pass, delivering **10x–17x speedups** over `LinearAlgebra.norm`.

3. **Inefficient Quadratic Forms ($b^T A b$)**:
   Computing quadratic forms $b^T A b$ is common across many linear algebra workloads, but standard implementations have notable performance drawbacks:
   - **Dense Matrices**: Standard `dot(b, A, b)` in `LinearAlgebra` is single-threaded and lacks unrolled vectorization. `fast_bAb` uses `@tturbo` to leverage multithreaded SIMD execution across CPU cores, achieving up to **2.4x speedup**.
   - **Static Arrays**: `fast_bAb` compiles directly into unrolled SIMD instructions via `@turbo`, running **2.0x–2.5x faster** than standard static array dot products.
   - **Sparse Matrices**: `SparseArrays.dot(b, A, b)` scales each non-zero matrix element by $b_i$ individually, executing $O(\text{nnz})$ scalar multiplications with $b$. `fast_bAb` factors out $b_i$ per column—first computing the column inner product with $b$ and then multiplying by $b_i$ once per column ($O(N)$ multiplications)—achieving a **2.2x speedup** on sparse systems.

## Performance Evaluation

A benchmark script is provided in [`examples/speed_test.jl`](examples/speed_test.jl). It times all included functions against standard implementations across multiple input sizes using warmup cycles, minimum-time filtering over multiple trials, and compiler barriers to prevent constant folding.

To run the speed test:

```bash
julia --project=. examples/speed_test.jl
```

### Benchmark Results

Results collected on an Apple Silicon (M-series) CPU:

#### 1. Vector Dot Product (`fast_dot` vs `LinearAlgebra.dot`)

| Vector Size | Standard (`LinearAlgebra.dot`) | FastLinAlg (`fast_dot`) | Speedup | Time Reduction |
|:-----------:|:------------------------------:|:-----------------------:|:-------:|:--------------:|
| 4 | ~9.6 ns | ~2.5 ns | ~3.9x | ~74.2% |
| 16 | ~13.0 ns | ~7.7 ns | ~1.7x | ~40.9% |
| 64 | ~14.6 ns | ~8.9 ns | ~1.6x | ~38.7% |
| 256 | ~41.5 ns | ~29.7 ns | ~1.4x | ~28.4% |
| 1024 | ~145.8 ns | ~113.0 ns | ~1.3x | ~22.5% |
| 4096 | ~568.5 ns | ~448.3 ns | ~1.3x | ~21.2% |

#### 2. Euclidean Norm (`fast_norm` vs `LinearAlgebra.norm`)

| Vector Size | Standard (`LinearAlgebra.norm`) | FastLinAlg (`fast_norm`) | Speedup | Time Reduction |
|:-----------:|:-------------------------------:|:------------------------:|:-------:|:--------------:|
| 4 | ~5.3 ns | ~1.9 ns | ~2.8x | ~64.7% |
| 16 | ~12.8 ns | ~2.8 ns | ~4.6x | ~78.2% |
| 64 | ~60.7 ns | ~5.3 ns | ~11.6x | ~91.3% |
| 256 | ~276.0 ns | ~17.2 ns | ~16.1x | ~93.8% |
| 1024 | ~1.23 μs | ~71.0 ns | ~17.4x | ~94.2% |
| 4096 | ~5.08 μs | ~312.3 ns | ~16.3x | ~93.9% |

#### 3. Squared Euclidean Norm (`sqnorm` vs `LinearAlgebra.dot(a, a)`)

| Vector Size | Standard (`dot(a, a)`) | FastLinAlg (`sqnorm`) | Speedup | Time Reduction |
|:-----------:|:----------------------:|:---------------------:|:-------:|:--------------:|
| 4 | ~9.3 ns | ~1.9 ns | ~5.0x | ~80.1% |
| 16 | ~13.0 ns | ~2.8 ns | ~4.7x | ~78.6% |
| 64 | ~14.9 ns | ~11.1 ns | ~1.3x | ~25.5% |
| 256 | ~39.5 ns | ~15.8 ns | ~2.5x | ~60.2% |
| 1024 | ~151.5 ns | ~75.2 ns | ~2.0x | ~50.4% |
| 4096 | ~588.6 ns | ~334.0 ns | ~1.8x | ~43.3% |

#### 4. Quadratic Form $b^T A b$ (`fast_bAb` vs `LinearAlgebra.dot(b, A, b)`)

**Dense Matrix & Vector (`Matrix{Float64}`)**

| Matrix Size | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |
|:-----------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|
| 8 × 8 | ~17.5 ns | ~13.0 ns | ~1.3x | ~25.5% |
| 32 × 32 | ~135.5 ns | ~88.8 ns | ~1.5x | ~34.4% |
| 128 × 128 | ~2.03 μs | ~1.34 μs | ~1.5x | ~34.0% |
| 512 × 512 | ~40.33 μs | ~16.65 μs | ~2.4x | ~58.7% |

**Static Arrays (`SMatrix` & `SVector`)**

| Array Size | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |
|:----------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|
| 2 × 2 | ~1.9 ns | ~1.6 ns | ~1.2x | ~16.6% |
| 4 × 4 | ~3.8 ns | ~1.9 ns | ~2.0x | ~50.4% |
| 8 × 8 | ~12.7 ns | ~5.6 ns | ~2.3x | ~56.2% |
| 12 × 12 | ~24.8 ns | ~9.9 ns | ~2.5x | ~59.9% |

**Sparse Matrix & Vector (`SparseMatrixCSC`, 5% Density)**

| Matrix Size | Non-zeros (nnz) | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |
|:-----------:|:---------------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|
| 50 × 50 | 101 | ~92.9 ns | ~98.0 ns | ~0.9x | -5.5% |
| 200 × 200 | 1914 | ~1.80 μs | ~792.0 ns | ~2.3x | ~55.9% |
| 1000 × 1000 | 49852 | ~46.49 μs | ~20.98 μs | ~2.2x | ~54.9% |

### Key Takeaways

- **Over 16x Faster Norm Computation**: `fast_norm` avoids multi-pass underflow/overflow scaling in BLAS `dnrm2`, reducing Euclidean norm latency by over 93% on vectors of length 64 to 4096.
- **Zero Foreign Function Overhead**: `fast_dot` and `sqnorm` avoid BLAS library call and threading setup costs, offering ~2x–5x lower latency for small vectors and consistently lower time across all dimensions.
- **Fast Quadratic Forms**: `fast_bAb` achieves ~2.2x–2.5x speedups across static arrays, dense multithreaded systems, and sparse CSC matrices by using optimal SIMD loop structures and factoring out per-column scalar multiplications.

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

### `fast_dot`

```julia
fast_dot(a::AbstractVector, b::AbstractVector) -> promote_type(eltype(a), eltype(b))
```

Computes the inner product $\sum_i a_i b_i$ of vectors `a` and `b` using `LoopVectorization.@turbo`.

### `fast_bAb`

```julia
fast_bAb(A::Matrix, b::Vector) -> eltype(b)
fast_bAb(A::StaticArray, b::StaticArray) -> eltype(b)
fast_bAb(A::SparseMatrixCSC, b::Vector) -> eltype(b)
```

Computes the quadratic form $b^T A b$ in a single pass with zero allocations:
- For `Matrix` and `Vector`: multithreaded SIMD via `LoopVectorization.@tturbo`.
- For `StaticArray`: single-threaded SIMD via `LoopVectorization.@turbo`.
- For `SparseMatrixCSC` and `Vector`: direct traversal over compressed sparse columns.

## Example

```julia
using FastLinAlg, StaticArrays, SparseArrays

a = randn(100)
b = randn(100)

s = sqnorm(a)
n = fast_norm(a)
d = fast_dot(a, b)

A_dense = randn(100, 100)
q1 = fast_bAb(A_dense, b)

A_sparse = sprandn(100, 100, 0.1)
q2 = fast_bAb(A_sparse, b)
```
