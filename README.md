# FastNorm.jl

[![Build Status](https://github.com/ojwoodford/FastNorm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/FastNorm.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/FastNorm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/FastNorm.jl)

High-performance norms and normalization methods in Julia that trade a small amount of numerical precision for significant speedups and reduced memory passes.

## Why FastNorm.jl?

Standard Julia linear algebra is built around standard BLAS and LAPACK libraries via `LinearAlgebra.jl`. While standard implementations are designed to guarantee full IEEE-754 precision and protect against extreme floating-point underflow or overflow, they introduce significant overhead in performance-critical inner loops involving small- to medium-sized data:

1. **BLAS Call and Thread-Checking Overhead**:
   Standard routines such as `LinearAlgebra.norm` invoke external C/Fortran BLAS subroutines (`dnrm2`). On every call, this incurs `ccall` overhead, foreign function interface (FFI) boundary transitions, argument marshalling, and multithreading synchronization checks. For small- to medium-sized vectors, this administrative overhead dwarfs the actual floating-point work. FastNorm executes native Julia SIMD loops directly in CPU registers without leaving the Julia runtime.

2. **Over-Defensive Multi-Pass Scaling in Standard Norms**:
   `LinearAlgebra.norm` uses BLAS `dnrm2` (or `generic_norm2`), which executes defensive multi-pass scaling (e.g., Blue's algorithm) with extra conditional branches and scaling factors to protect against extreme underflow or overflow. When data magnitudes are well within normal IEEE-754 dynamic range, this defensive handling adds unnecessary latency. `fast_norm` and `sqnorm` compute norms directly in a single pass, delivering **up to 17x speedups** over `LinearAlgebra.norm`.

3. **In-Place Unit-Norm Normalization (`norm1!!`)**:
   Standard normalization to unit Euclidean norm (`vec ./= norm(vec)`) incurs the full multi-pass scaling overhead of BLAS `dnrm2` followed by elementwise array division. `norm1!!` uses single-pass SIMD `sqnorm` and normalizes the vector in-place (or returns a new static vector) using reciprocal multiplication. To prevent division by zero or NaN creation on degenerate vectors without artificially restricting floating-point dynamic range, `norm1!!` defaults to `minsqnorm = floatmin(T)` (the smallest positive normal number for type `T`), delivering **up to 7.7x speedups**.

4. **Single-Pass Mean and Normalizer (`vecmeannorm`)**:
   Standard calculations of the mean $\bar{x}$ and norm of deviations $\sqrt{\sum_i (x_i - \bar{x})^2}$ require two sequential memory passes over the vector: pass 1 computes the mean, and pass 2 computes the squared differences from the mean. `vecmeannorm` computes both the mean and the normalization factor in a **single pass** using the algebraic identity:
   $$\sum_{i=1}^N (x_i - \bar{x})^2 = \sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i$$
   Clamping with the default minimum normalization constant `floatmin(T)` protects against division by zero or NaN results when variance is zero. Halving the memory traffic and leveraging SIMD vectorization yields **up to 3.3x speedups**.

5. **Two-Pass In-Place Zero-Mean Unit-Norm Normalization (`mean0norm1!!`)**:
   Normalizing a vector to have zero mean and unit Euclidean norm ($\mu = 0$, $\|v\|_2 = 1$) conventionally requires 4 memory passes:
   1. Compute mean (pass 1: read vector)
   2. Subtract mean (pass 2: write vector)
   3. Compute Euclidean norm via BLAS (pass 3: read vector)
   4. Divide by norm (pass 4: write vector)

   `mean0norm1!!` uses `vecmeannorm` to compute both the mean and the normalization factor in a single read pass, and then broadcasts the affine transformation $(v_i - \bar{x}) \cdot \text{scale}$ in a second pass. Memory is traversed only twice rather than four times. For mutable arrays (such as standard `Vector` or views), the operation modifies elements in-place with **zero allocations**; for immutable arrays (such as `StaticArrays.SVector`), a new normalized static vector is returned. Clamped with `floatmin(T)` by default, it delivers **up to 8.8x speedups** over standard multi-pass normalization pipelines.

## Performance Evaluation

A benchmark script is provided in [`examples/speed_test.jl`](examples/speed_test.jl). It benchmarks all FastNorm functions against standard implementations across multiple input sizes using warmup cycles, minimum-time filtering over multiple trials, and compiler barriers to prevent constant folding.

To run the speed test:

```bash
julia --project=. examples/speed_test.jl
```

### Benchmark Results

Results collected on an Apple Silicon (M-series) CPU:

#### 1. Euclidean Norm (`fast_norm` vs `LinearAlgebra.norm`)

| Vector Size | Standard (`LinearAlgebra.norm`) | FastNorm (`fast_norm`) | Speedup | Time Reduction |
|:-----------:|:-------------------------------:|:------------------------:|:-------:|:--------------:|
| 4 | ~5.4 ns | ~1.9 ns | ~2.9x | ~65.2% |
| 16 | ~12.7 ns | ~2.9 ns | ~4.4x | ~77.4% |
| 64 | ~62.5 ns | ~5.3 ns | ~11.9x | ~91.6% |
| 256 | ~275.4 ns | ~17.1 ns | ~16.1x | ~93.8% |
| 1024 | ~1.26 μs | ~71.5 ns | ~17.6x | ~94.3% |
| 4096 | ~5.13 μs | ~312.2 ns | ~16.4x | ~93.9% |

#### 2. Squared Euclidean Norm (`sqnorm` vs `LinearAlgebra.dot(a, a)`)

| Vector Size | Standard (`dot(a, a)`) | FastNorm (`sqnorm`) | Speedup | Time Reduction |
|:-----------:|:----------------------:|:---------------------:|:-------:|:--------------:|
| 4 | ~9.3 ns | ~1.9 ns | ~5.0x | ~80.0% |
| 16 | ~13.1 ns | ~2.8 ns | ~4.7x | ~78.7% |
| 64 | ~14.9 ns | ~9.0 ns | ~1.7x | ~40.0% |
| 256 | ~39.0 ns | ~15.3 ns | ~2.6x | ~60.9% |
| 1024 | ~138.3 ns | ~67.2 ns | ~2.1x | ~51.4% |
| 4096 | ~542.2 ns | ~311.9 ns | ~1.7x | ~42.5% |

#### 3. Unit-Norm Normalization (`norm1!!` vs Standard `./= norm`)

| Vector Size | Standard (`./= norm`) | FastNorm (`norm1!!`) | Speedup | Time Reduction |
|:-----------:|:---------------------:|:----------------------:|:-------:|:--------------:|
| 4 | ~14.7 ns | ~15.6 ns | ~0.9x | -5.5% |
| 16 | ~27.2 ns | ~19.3 ns | ~1.4x | ~29.0% |
| 64 | ~106.6 ns | ~23.1 ns | ~4.6x | ~78.3% |
| 256 | ~378.4 ns | ~55.2 ns | ~6.9x | ~85.4% |
| 1024 | ~1.45 μs | ~197.9 ns | ~7.3x | ~86.4% |
| 4096 | ~5.76 μs | ~744.4 ns | ~7.7x | ~87.1% |

#### 4. Mean and Normalizer (`vecmeannorm` vs Standard 2-Pass)

| Vector Size | Standard (2-pass) | FastNorm (`vecmeannorm`) | Speedup | Time Reduction |
|:-----------:|:-----------------:|:--------------------------:|:-------:|:--------------:|
| 4 | ~4.7 ns | ~3.4 ns | ~1.4x | ~27.1% |
| 16 | ~10.4 ns | ~4.4 ns | ~2.4x | ~57.9% |
| 64 | ~20.8 ns | ~8.2 ns | ~2.5x | ~60.6% |
| 256 | ~65.6 ns | ~25.8 ns | ~2.5x | ~60.6% |
| 1024 | ~279.4 ns | ~84.9 ns | ~3.3x | ~69.6% |
| 4096 | ~1.05 μs | ~324.1 ns | ~3.2x | ~69.2% |

#### 5. Zero-Mean Unit-Norm Normalization (`mean0norm1!!` vs Standard Multi-Pass)

| Vector Size | Standard (Multi-pass) | FastNorm (`mean0norm1!!`) | Speedup | Time Reduction |
|:-----------:|:---------------------:|:---------------------------:|:-------:|:--------------:|
| 4 | ~26.1 ns | ~24.9 ns | ~1.1x | ~4.8% |
| 16 | ~47.9 ns | ~27.1 ns | ~1.8x | ~43.5% |
| 64 | ~133.9 ns | ~32.0 ns | ~4.2x | ~76.1% |
| 256 | ~442.2 ns | ~64.5 ns | ~6.9x | ~85.4% |
| 1024 | ~1.69 μs | ~202.1 ns | ~8.3x | ~88.0% |
| 4096 | ~6.59 μs | ~748.1 ns | ~8.8x | ~88.6% |

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

Computes the Euclidean L2 norm $\sqrt{\text{sqnorm}(\text{vec})}$ directly in native Julia without calling BLAS.

### `norm1!!`

```julia
norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T -> V
norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> AbstractVector{T}
```

Normalizes a vector to unit Euclidean norm ($\|v\|_2 \approx 1$). Computes the squared norm via `sqnorm` in a single pass, then scales the vector in a second pass. For mutable vectors (`Vector`, views of mutable arrays), the operation modifies `vec` in-place (`vec === norm1!!(vec)`). For immutable arrays (`StaticArrays.SVector`), it returns a new normalized vector. Clamping to `minsqnorm` (defaulting to `floatmin(T)`) prevents division by zero and NaN generation for zero vectors.

### `vecmeannorm`

```julia
vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where T -> Tuple{T, T}
vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> Tuple{T, T}
```

Computes the mean $\bar{x}$ and normalization factor $1 / \sqrt{\max(\sum (x_i - \bar{x})^2, \text{minsqnorm})}$ in a single memory pass over `vec`. Returns `(vecmean, vecnorm)`. Clamping to `minsqnorm` (defaulting to `floatmin(T)`) guards against division by zero for zero-variance inputs.

### `mean0norm1!!`

```julia
mean0norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T -> V
mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> AbstractVector{T}
```

Normalizes a vector to zero mean and unit Euclidean norm ($\mu = 0, \|v\|_2 = 1$). Uses `vecmeannorm` to compute the mean and normalizer in one pass, then applies the scaling in a second pass. For mutable vectors (`Vector`, views of mutable arrays), the operation is in-place and returns the modified array (`vec === mean0norm1!!(vec)`). For immutable arrays (`StaticArrays.SVector`), it returns a new normalized vector. Defaults to `minsqnorm = floatmin(T)`.

## Example

```julia
using FastNorm, StaticArrays

# Standard dense vectors
a = randn(100)
s = sqnorm(a)
n = fast_norm(a)

# In-place unit-norm normalization
norm1!!(a)

# Single-pass mean and normalizer
mu, normalizer = vecmeannorm(a)

# In-place zero-mean unit-norm normalization
mean0norm1!!(a)

# Immutable static vectors
svec = SVector{4, Float64}(1.0, 2.0, 3.0, 4.0)
svec_unit = norm1!!(svec)
svec_norm = mean0norm1!!(svec)
```
