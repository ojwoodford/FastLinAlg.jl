using FastLinAlg, LinearAlgebra, StaticArrays, Test

@testset "FastLinAlg.jl" begin
    # sqnorm, fast_norm
    a = randn(1001)
    @test sqnorm(-3.0) == 9.0
    @test sqnorm(4) == 16
    @test sqnorm(a) ≈ dot(a, a)
    @test sqnorm(Float64[]) == 0.0
    @test sqnorm(SVector(3.0, 4.0)) == 25.0

    @test fast_norm(a) ≈ norm(a)
    @test fast_norm(Float64[]) == 0.0
    @test fast_norm(SVector(3.0, 4.0)) == 5.0
end
