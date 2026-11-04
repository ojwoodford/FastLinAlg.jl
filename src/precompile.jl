using PrecompileTools

@setup_workload begin
    # Float64 data
    v_dense64_1 = [1.0, 2.0, 3.0, 4.0]
    v_dense64_2 = [0.5, 1.5, 2.5, 3.5]
    v_view64_1 = view(v_dense64_1, 1:3)
    v_view64_2 = view(v_dense64_2, 1:3)

    m_dense64 = [1.0 0.5; 0.5 2.0]
    m_vec64 = [1.0, 2.0]

    m_sparse64 = SparseArrays.spzeros(Float64, 3, 3)
    m_sparse64[1, 1] = 2.0
    m_sparse64[1, 2] = 0.5
    m_sparse64[2, 1] = 0.5
    m_sparse64[2, 2] = 3.0
    m_sparse64[3, 3] = 1.0
    v_sparse64 = [1.0, 2.0, 1.5]

    # Float32 data
    v_dense32_1 = Float32[1.0, 2.0, 3.0, 4.0]
    v_dense32_2 = Float32[0.5, 1.5, 2.5, 3.5]
    v_view32_1 = view(v_dense32_1, 1:3)
    v_view32_2 = view(v_dense32_2, 1:3)

    m_dense32 = Float32[1.0 0.5; 0.5 2.0]
    m_vec32 = Float32[1.0, 2.0]

    m_sparse32 = SparseArrays.spzeros(Float32, 3, 3)
    m_sparse32[1, 1] = 2.0f0
    m_sparse32[1, 2] = 0.5f0
    m_sparse32[2, 1] = 0.5f0
    m_sparse32[2, 2] = 3.0f0
    m_sparse32[3, 3] = 1.0f0
    v_sparse32 = Float32[1.0, 2.0, 1.5]

    # Int data (for cumsum! only)
    v_int = [1, 2, 3, 4]

    @compile_workload begin
        # 1. Scalar functions (Float64 and Float32)
        sqnorm(3.0)
        sqnorm(3.0f0)

        # 2. Vector sqnorm (Float64 and Float32)
        sqnorm(v_dense64_1)
        sqnorm(v_view64_1)
        sqnorm(v_dense32_1)
        sqnorm(v_view32_1)

        # 3. fast_norm (Float64 and Float32)
        fast_norm(v_dense64_1)
        fast_norm(v_view64_1)
        fast_norm(v_dense32_1)
        fast_norm(v_view32_1)

        # 4. fast_dot (Float64 and Float32)
        fast_dot(v_dense64_1, v_dense64_2)
        fast_dot(v_view64_1, v_view64_2)
        fast_dot(v_dense32_1, v_dense32_2)
        fast_dot(v_view32_1, v_view32_2)

        # 5. fast_bAb (Float64 and Float32 dense and sparse matrices)
        fast_bAb(m_dense64, m_vec64)
        fast_bAb(m_sparse64, v_sparse64)
        fast_bAb(m_dense32, m_vec32)
        fast_bAb(m_sparse32, v_sparse32)

        # 6. Base.cumsum! (Int only)
        cumsum!(v_int)
    end
end
