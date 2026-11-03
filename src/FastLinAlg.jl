module FastLinAlg

using LoopVectorization, SparseArrays, StaticArrays

export sqnorm, fast_norm, fast_dot, fast_bAb

"""
    sqnorm(x::Number)
    sqnorm(vec::AbstractVector)

Computes the squared Euclidean norm (`x * x` for a scalar, or `sum(abs2, vec)` for a vector)
using SIMD vectorization via `LoopVectorization.@turbo`.
"""
sqnorm(x::Number) = @fastmath x * x
function sqnorm(vec::AbstractVector)
    total = zero(eltype(vec))
    @turbo for i in eachindex(vec)
        total += vec[i] * vec[i]
    end
    return total
end

"""
    fast_norm(vec::AbstractVector)

Computes the Euclidean 2-norm `sqrt(sqnorm(vec))` in native SIMD Julia without calling multi-threaded BLAS.
"""
fast_norm(vec::AbstractVector) = sqrt(sqnorm(vec))

"""
    fast_dot(a::AbstractVector, b::AbstractVector)

Computes the dot product of vectors `a` and `b` using `LoopVectorization.@turbo` without BLAS thread wakeup overhead.
"""
function fast_dot(a::AbstractVector, b::AbstractVector)
    total = zero(promote_type(eltype(a), eltype(b)))
    @turbo for i in eachindex(a, b)
        total += a[i] * b[i]
    end
    return total
end

"""
    fast_bAb(A::Matrix, b::Vector)
    fast_bAb(A::StaticArray, b::StaticArray)
    fast_bAb(A::SparseMatrixCSC, b::Vector)

Computes the quadratic form `b' * A * b` without allocating intermediate vectors.
Uses `@tturbo` for dense `Matrix`, `@turbo` for `StaticArray`, and direct CSC non-zero iteration for `SparseMatrixCSC`.
"""
function fast_bAb(A::Matrix, b::Vector)
    total = zero(eltype(b))
    @tturbo for i in eachindex(b)
        subtotal = zero(eltype(b))
        for j in eachindex(b)
            subtotal += A[j,i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::StaticArray, b::StaticArray)
    total = zero(eltype(b))
    @turbo for i in eachindex(b)
        subtotal = zero(eltype(b))
        for j in eachindex(b)
            subtotal += A[j,i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::SparseMatrixCSC, b::Vector)
    total = zero(eltype(b))
    for i in eachindex(b)
        coltotal = zero(eltype(b))
        for j in nzrange(A, i)
            @inbounds @fastmath coltotal += A.nzval[j] * b[A.rowval[j]]
        end
        @inbounds @fastmath coltotal *= b[i]
        total += coltotal
    end
    return total
end

"""
    Base.cumsum!(A::AbstractVector)

Single-argument in-place cumulative sum overwriting `A` with `cumsum(A)` and returning `A`.
"""
function Base.cumsum!(A::AbstractVector)
    total = zero(eltype(A))
    @inbounds for ind in eachindex(A)
        @fastmath total += A[ind]
        A[ind] = total
    end
    return A
end

end # module FastLinAlg
