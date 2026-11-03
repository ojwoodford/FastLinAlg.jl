# FastLinAlg.jl

SIMD-accelerated, zero-allocation linear algebra primitives in Julia, avoiding BLAS thread wakeup overhead for small and medium vectors and matrices.

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

### `Base.cumsum!`

```julia
cumsum!(A::AbstractVector) -> A
```

Single-argument in-place cumulative sum that mutates `A` so `A[k] = sum(A[1:k])` and returns `A`.

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

x = [1, 2, 3, 4]
cumsum!(x) # [1, 3, 6, 10]
```
