using PrecompileTools

@setup_workload begin
    # Float64 data
    v_dense64 = [1.0, 2.0, 3.0, 4.0]
    v_view64 = view(v_dense64, 1:3)
    v_svec64 = SVector{4, Float64}(1.0, 2.0, 3.0, 4.0)

    # Float32 data
    v_dense32 = Float32[1.0, 2.0, 3.0, 4.0]
    v_view32 = view(v_dense32, 1:3)
    v_svec32 = SVector{4, Float32}(1.0f0, 2.0f0, 3.0f0, 4.0f0)

    @compile_workload begin
        # 1. Scalar functions (Float64 and Float32)
        sqnorm(3.0)
        sqnorm(3.0f0)

        # 2. Vector sqnorm (Float64 and Float32)
        sqnorm(v_dense64)
        sqnorm(v_view64)
        sqnorm(v_svec64)
        sqnorm(v_dense32)
        sqnorm(v_view32)
        sqnorm(v_svec32)

        # 3. fast_norm (Float64 and Float32)
        fast_norm(v_dense64)
        fast_norm(v_view64)
        fast_norm(v_svec64)
        fast_norm(v_dense32)
        fast_norm(v_view32)
        fast_norm(v_svec32)

        # 4. vecmeannorm (Float64 and Float32)
        vecmeannorm(v_dense64)
        vecmeannorm(v_view64)
        vecmeannorm(v_svec64)
        vecmeannorm(v_dense64, 1e-8)
        vecmeannorm(v_dense64; minsqnorm=1e-8)

        vecmeannorm(v_dense32)
        vecmeannorm(v_view32)
        vecmeannorm(v_svec32)
        vecmeannorm(v_dense32, 1e-8)
        vecmeannorm(v_dense32; minsqnorm=1e-8)

        # 5. mean0norm1!! (Float64 and Float32)
        mean0norm1!!(v_dense64)
        mean0norm1!!(v_view64)
        mean0norm1!!(v_svec64)
        mean0norm1!!(v_dense64, 1e-8)
        mean0norm1!!(v_dense64; minsqnorm=1e-8)

        mean0norm1!!(v_dense32)
        mean0norm1!!(v_view32)
        mean0norm1!!(v_svec32)
        mean0norm1!!(v_dense32, 1e-8)
        mean0norm1!!(v_dense32; minsqnorm=1e-8)

        # 6. norm1!! (Float64 and Float32)
        norm1!!(v_dense64)
        norm1!!(v_view64)
        norm1!!(v_svec64)
        norm1!!(v_dense64, 1e-8)
        norm1!!(v_dense64; minsqnorm=1e-8)

        norm1!!(v_dense32)
        norm1!!(v_view32)
        norm1!!(v_svec32)
        norm1!!(v_dense32, 1e-8)
        norm1!!(v_dense32; minsqnorm=1e-8)
    end
end
