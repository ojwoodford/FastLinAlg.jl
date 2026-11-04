# Speed test comparing FastLinAlg primitives to standard Julia / LinearAlgebra implementations
using FastLinAlg
using LinearAlgebra
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
    # 1. Euclidean Norm: fast_norm vs LinearAlgebra.norm
    # -------------------------------------------------------------------------
    println("\n### 1. Euclidean Norm: `fast_norm` vs `LinearAlgebra.norm`\n")
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
    # 2. Squared Euclidean Norm: sqnorm vs LinearAlgebra.dot(a, a)
    # -------------------------------------------------------------------------
    println("\n### 2. Squared Euclidean Norm: `sqnorm` vs `LinearAlgebra.dot(a, a)`\n")
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

    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_speed_test()
end
