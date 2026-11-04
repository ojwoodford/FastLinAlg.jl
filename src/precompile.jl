using PrecompileTools

@setup_workload begin
    # Float64 data
    v_dense64 = [1.0, 2.0, 3.0, 4.0]
    v_view64 = view(v_dense64, 1:3)

    # Float32 data
    v_dense32 = Float32[1.0, 2.0, 3.0, 4.0]
    v_view32 = view(v_dense32, 1:3)

    @compile_workload begin
        # 1. Scalar functions (Float64 and Float32)
        sqnorm(3.0)
        sqnorm(3.0f0)

        # 2. Vector sqnorm (Float64 and Float32)
        sqnorm(v_dense64)
        sqnorm(v_view64)
        sqnorm(v_dense32)
        sqnorm(v_view32)

        # 3. fast_norm (Float64 and Float32)
        fast_norm(v_dense64)
        fast_norm(v_view64)
        fast_norm(v_dense32)
        fast_norm(v_view32)
    end
end
