# Speed test comparing FastLinAlg primitives to standard Julia / LinearAlgebra implementations
using FastLinAlg
using LinearAlgebra
using SparseArrays
using StaticArrays
using Random

"""
    bench_call(f::F, iters::Int, trials::Int=5) where F

Time execution of `f()` across `trials` independent runs of `iters` iterations each,
returning the minimum time per call in nanoseconds.
"""
@noinline function bench_call(f::F, iters::Int, trials::Int=5) where F
    # Warmup
    Base.inferencebarrier(f())
    best = Inf
    for _ in 1:trials
        t0 = time_ns()
        for _ in 1:iters
            Base.inferencebarrier(f())
        end
        t = (time_ns() - t0) / iters
        best = min(best, t)
    end
    return best
end

@noinline function bench_cumsum_inplace!(bufs::Vector{Vector{Float64}}, iters::Int, K::Int)
    t0 = time_ns()
    @inbounds for i in 1:iters
        Base.inferencebarrier(cumsum!(bufs[(i & (K - 1)) + 1]))
    end
    return (time_ns() - t0) / iters
end

@noinline function bench_cumsum_2arg!(out::Vector{Float64}, bufs::Vector{Vector{Float64}}, iters::Int, K::Int)
    t0 = time_ns()
    @inbounds for i in 1:iters
        Base.inferencebarrier(Base.cumsum!(out, bufs[(i & (K - 1)) + 1]))
    end
    return (time_ns() - t0) / iters
end

@noinline function bench_cumsum_allocating(x::Vector{Float64}, iters::Int)
    t0 = time_ns()
    for _ in 1:iters
        Base.inferencebarrier(cumsum(x))
    end
    return (time_ns() - t0) / iters
end

function format_time(t_ns::Float64)
    if t_ns < 1000
        return string(round(t_ns, digits=2), " ns")
    elseif t_ns < 1_000_000
        return string(round(t_ns / 1000, digits=2), " μs")
    else
        return string(round(t_ns / 1_000_000, digits=2), " ms")
    end
end

"""
    run_speed_test(trials=5)

Run comprehensive benchmarks comparing all FastLinAlg functions against standard implementations
and print Markdown-formatted performance summary tables.
"""
function run_speed_test(trials::Int=5)
    Random.seed!(42)

    println("=" ^ 88)
    println("FastLinAlg.jl Speed Test: FastLinAlg vs Standard Julia / LinearAlgebra")
    println("Trials per test: $trials (taking minimum time per call)")
    println("=" ^ 88)

    # -------------------------------------------------------------------------
    # 1. Vector Dot Product: fast_dot vs LinearAlgebra.dot
    # -------------------------------------------------------------------------
    println("\n### 1. Vector Dot Product: `fast_dot` vs `LinearAlgebra.dot`\n")
    println("| Vector Size | Standard (`LinearAlgebra.dot`) | FastLinAlg (`fast_dot`) | Speedup | Time Reduction |")
    println("|:-----------:|:------------------------------:|:-----------------------:|:-------:|:--------------:|")

    for n in [4, 16, 64, 256, 1024, 4096]
        a = randn(n)
        b = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        @assert fast_dot(a, b) ≈ dot(a, b)

        t_std = bench_call(() -> dot(a, b), iters, trials)
        t_fast = bench_call(() -> fast_dot(a, b), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # -------------------------------------------------------------------------
    # 2. Euclidean Norm: fast_norm vs LinearAlgebra.norm
    # -------------------------------------------------------------------------
    println("\n### 2. Euclidean Norm: `fast_norm` vs `LinearAlgebra.norm`\n")
    println("| Vector Size | Standard (`LinearAlgebra.norm`) | FastLinAlg (`fast_norm`) | Speedup | Time Reduction |")
    println("|:-----------:|:-------------------------------:|:------------------------:|:-------:|:--------------:|")

    for n in [4, 16, 64, 256, 1024, 4096]
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        @assert fast_norm(a) ≈ norm(a)

        t_std = bench_call(() -> norm(a), iters, trials)
        t_fast = bench_call(() -> fast_norm(a), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # -------------------------------------------------------------------------
    # 3. Squared Euclidean Norm: sqnorm vs LinearAlgebra.dot(a, a)
    # -------------------------------------------------------------------------
    println("\n### 3. Squared Euclidean Norm: `sqnorm` vs `LinearAlgebra.dot(a, a)`\n")
    println("| Vector Size | Standard (`dot(a, a)`) | FastLinAlg (`sqnorm`) | Speedup | Time Reduction |")
    println("|:-----------:|:----------------------:|:---------------------:|:-------:|:--------------:|")

    for n in [4, 16, 64, 256, 1024, 4096]
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        @assert sqnorm(a) ≈ dot(a, a)

        t_std = bench_call(() -> dot(a, a), iters, trials)
        t_fast = bench_call(() -> sqnorm(a), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # -------------------------------------------------------------------------
    # 4. Quadratic Form b' * A * b: fast_bAb vs LinearAlgebra.dot(b, A, b)
    # -------------------------------------------------------------------------
    println("\n### 4. Quadratic Form (b' * A * b): `fast_bAb` vs `LinearAlgebra.dot(b, A, b)`\n")

    println("#### 4a. Dense Matrix & Vector (`Matrix{Float64}`)\n")
    println("| Matrix Size | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |")
    println("|:-----------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|")

    for n in [8, 32, 128, 512]
        A = randn(n, n)
        b = randn(n)
        iters = max(100, 100_000 ÷ (n * n ÷ 10 + 1))

        # Verification
        @assert fast_bAb(A, b) ≈ dot(b, A, b)

        t_std = bench_call(() -> dot(b, A, b), iters, trials)
        t_fast = bench_call(() -> fast_bAb(A, b), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n × $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    println("\n#### 4b. Static Arrays (`SMatrix` & `SVector`)\n")
    println("| Array Size | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |")
    println("|:----------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|")

    for n in [2, 4, 8, 12]
        A = SMatrix{n, n}(randn(n, n))
        b = SVector{n}(randn(n))
        iters = 50_000

        # Verification
        @assert fast_bAb(A, b) ≈ dot(b, A, b)

        t_std = bench_call(() -> dot(b, A, b), iters, trials)
        t_fast = bench_call(() -> fast_bAb(A, b), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n × $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    println("\n#### 4c. Sparse Matrix & Vector (`SparseMatrixCSC`, 5% Density)\n")
    println("| Matrix Size | Non-zeros (nnz) | Standard (`dot(b, A, b)`) | FastLinAlg (`fast_bAb`) | Speedup | Time Reduction |")
    println("|:-----------:|:---------------:|:-------------------------:|:-----------------------:|:-------:|:--------------:|")

    for n in [50, 200, 1000]
        A = sprandn(n, n, 0.05)
        b = randn(n)
        iters = max(100, 50_000 ÷ n)

        # Verification
        @assert fast_bAb(A, b) ≈ dot(b, A, b)

        t_std = bench_call(() -> dot(b, A, b), iters, trials)
        t_fast = bench_call(() -> fast_bAb(A, b), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n × $n | $(nnz(A)) | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # -------------------------------------------------------------------------
    # 5. In-place Cumulative Sum: Base.cumsum!(A) vs Base implementations
    # -------------------------------------------------------------------------
    println("\n### 5. In-place Cumulative Sum: `cumsum!(A)` vs Standard Implementations\n")
    println("| Vector Size | Allocating (`cumsum(x)`) | 2-Arg In-place (`cumsum!(out, x)`) | FastLinAlg 1-Arg (`cumsum!(x)`) | Speedup vs `cumsum` | Time Reduction |")
    println("|:-----------:|:------------------------:|:----------------------------------:|:-------------------------------:|:--------------------:|:--------------:|")

    for n in [8, 64, 512, 4096]
        x = randn(n)
        K = 64
        iters = max(1000, 200_000 ÷ n)
        bufs_fast = [copy(x) for _ in 1:K]
        bufs_base = [copy(x) for _ in 1:K]
        out = similar(x)

        # Verification
        test_buf = copy(x)
        cumsum!(test_buf)
        @assert test_buf ≈ cumsum(x)

        t_alloc = minimum([bench_cumsum_allocating(x, iters) for _ in 1:trials])
        t_2arg = minimum([bench_cumsum_2arg!(out, bufs_base, iters, K) for _ in 1:trials])
        t_fast = minimum([bench_cumsum_inplace!(bufs_fast, iters, K) for _ in 1:trials])

        speedup = t_alloc / t_fast
        reduction = (1.0 - t_fast / t_alloc) * 100.0
        println("| $n | $(format_time(t_alloc)) | $(format_time(t_2arg)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_speed_test()
end
