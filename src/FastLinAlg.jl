module FastLinAlg

using LoopVectorization, PrecompileTools

export sqnorm, fast_norm

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

include("precompile.jl")

end # module FastLinAlg
