# GPU tests for the Philox4x32-10 CUDA extension (ext/RandomDataStreamsCUDAExt.jl).
#
# Included from runtests.jl, so `Pkg.test()` always loads CUDA.jl (it is a
# dependency of test/Project.toml) and runs these on any machine with a
# functional CUDA device. On a machine without one -- CI included -- CUDA.jl
# still precompiles, `CUDA.functional()` reports `false`, and the testset
# below is skipped with an @info rather than silently passing nothing.
#
# Run just this file directly, without the rest of the suite, with:
#
#   julia --project=test -e 'include("test/test_cuda.jl")'

using RandomDataStreams
using Random
using Test
using CUDA

# `normcdfinv` moved to the CUDACore subpackage in CUDA.jl 6; resolve it the way
# the extension does.
const _normcdfinv = isdefined(CUDA, :normcdfinv) ? CUDA.normcdfinv : CUDA.CUDACore.normcdfinv

# Applied through a kernel rather than broadcast: in CUDA.jl 6 the intrinsic has
# no host method, so broadcasting it fails at eltype inference.
function _normcdfinv_kernel!(out, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(out) && (@inbounds out[i] = _normcdfinv(u[i]))
    return nothing
end

function device_normcdfinv(u::AbstractVector{Float64})
    d = CuArray(u)
    out = similar(d)
    @cuda threads = 256 blocks = cld(length(d), 256) _normcdfinv_kernel!(out, d)
    return Array(out)
end

if !CUDA.functional()
    @info "CUDA is not functional on this machine -- skipping GPU tests" CUDA.functional()
else

@testset "rand! on the GPU is the CPU fill, bit for bit" begin
    families = [
        ("Philox4x32-10",   PhiloxRNG),
        ("Philox4x64-10",   Philox4x64RNG),
        ("Threefry4x32-20", Threefry4x32RNG),
        ("Threefry4x64-20", Threefry4x64RNG),
    ]
    for (name, G) in families
        @testset "$name" begin
            # `pre` 32-bit draws put the generator at every offset within a
            # block; odd offsets make a Float64 of a 32-bit family straddle two
            for T in (Float64, Float32, Float16), pre in 0:4, n in (1, 2, 3, 5, 257, 10_001)
                start = G(20260927)
                for _ in 1:pre
                    rand(start, UInt32)
                end
                cpu, gpu = copy(start), copy(start)
                cpu_out = rand!(cpu, Vector{T}(undef, n))
                gpu_out = rand!(gpu, CUDA.zeros(T, n))
                @test Array(gpu_out) == cpu_out
                # the host object is left exactly where the CPU fill leaves it
                @test get_state(gpu)[1] == get_state(cpu)[1]      # counter
                @test get_state(gpu)[4] == get_state(cpu)[4]      # index in block
                @test rand(gpu, UInt64) == rand(cpu, UInt64)
            end

            # across a carry into the high half of the counter, and across the
            # wrap of the whole 128-bit counter
            for c in ((UInt128(3) << 64) - UInt128(2), typemax(UInt128) - UInt128(1)), T in (Float64, Float32)
                cpu = G(11); cpu.ctr = c
                gpu = copy(cpu)
                @test Array(rand!(gpu, CUDA.zeros(T, 40))) == rand!(cpu, Vector{T}(undef, 40))
                @test get_state(gpu)[1] == get_state(cpu)[1]
            end
        end
    end

    @testset "does not depend on the launch configuration" begin
        # Each element is computed from (key, counter, index) alone, so the
        # number of threads per block cannot change a draw.
        ext = Base.get_extension(RandomDataStreams, :RandomDataStreamsCUDAExt)
        rng = PhiloxRNG(UInt32[3, 4])
        rand(rng, UInt32)                                  # start mid-block
        hi, lo, w0 = RandomDataStreams._draw_position(rng)
        n = 5_003
        ref = rand!(copy(rng), Vector{Float64}(undef, n))
        for t in (32, 96, 256, 1024)
            A = CUDA.zeros(Float64, n)
            @cuda threads = t blocks = cld(n, t) ext._fill_kernel!(A, typeof(rng), rng.key, hi, lo, w0, n, Val(isodd(w0)))
            @test Array(A) == ref
        end
    end

    @testset "values land in (0, 1)" begin
        for T in (Float64, Float32, Float16)
            v = Array(rand!(PhiloxRNG((UInt32(11), UInt32(22))), CUDA.zeros(T, 50_000)))
            @test all(0 .< v .< 1)
            @test 0.48 < sum(Float64.(v)) / length(v) < 0.52
        end
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG((UInt32(11), UInt32(22)))
        rand!(rng, CUDA.zeros(Float64, 0))
        @test get_state(rng)[1] == 0
    end
end

@testset "PhiloxRNG CUDA randn! (Box-Muller)" begin
    key = (UInt32(33), UInt32(44))

    @testset "statistics of a large draw" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 200_000)
        randn!(rng, out)
        v = Array(out)
        @test all(isfinite, v)
        @test abs(sum(v) / length(v)) < 0.02                        # mean ~ 0
        @test abs(sum(v .^ 2) / length(v) - 1.0) < 0.05              # variance ~ 1
    end

    @testset "deterministic for a fixed key and counter" begin
        a = PhiloxRNG(key); out_a = CUDA.zeros(Float64, 37)
        b = PhiloxRNG(key); out_b = CUDA.zeros(Float64, 37)
        randn!(a, out_a)
        randn!(b, out_b)
        @test Array(out_a) == Array(out_b)
    end

    @testset "advances the counter exactly like rand! does" begin
        for n in (1, 2, 3, 41)
            a = PhiloxRNG(key); randn!(a, CUDA.zeros(Float64, n))
            b = PhiloxRNG(key); rand!(b, CUDA.zeros(Float64, n))
            @test get_state(a)[1] == get_state(b)[1] == cld(n, 2)
        end
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float64, 10)
        @test_throws ArgumentError randn!(rng, out)

        reset_substream!(rng)
        @test randn!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 0)
        randn!(rng, out)
        @test get_state(rng)[1] == 0
    end
end

@testset "PhiloxRNG CUDA randn_inversion! (inversion, comparison branch)" begin
    key = (UInt32(111), UInt32(222))

    @testset "matches normcdfinv of a direct block computation" begin
        n = 4001
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, n)
        randn_inversion!(rng, out)
        v = Array(out)

        u = Float64[]
        b = 0
        while length(u) < n
            ctr = philox4x32_counter(UInt64(0), UInt64(b))
            blk = RandomDataStreams.philox(ctr, key, Val(10))
            lo = (UInt64(blk[2]) << 32) | UInt64(blk[1])
            hi = (UInt64(blk[4]) << 32) | UInt64(blk[3])
            for w in (lo, hi)
                push!(u, open01(w))
            end
            b += 1
        end
        expected = device_normcdfinv(u[1:n])
        @test v ≈ expected
    end

    @testset "statistics of a large draw" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 200_000)
        randn_inversion!(rng, out)
        v = Array(out)
        @test all(isfinite, v)
        @test abs(sum(v) / length(v)) < 0.02
        @test abs(sum(v .^ 2) / length(v) - 1.0) < 0.05
    end

    @testset "deterministic for a fixed key and counter" begin
        a = PhiloxRNG(key); out_a = CUDA.zeros(Float64, 37)
        b = PhiloxRNG(key); out_b = CUDA.zeros(Float64, 37)
        randn_inversion!(a, out_a)
        randn_inversion!(b, out_b)
        @test Array(out_a) == Array(out_b)
    end

    @testset "antithetic symmetry: Phi^-1(1 - u) == -Phi^-1(u)" begin
        # The whole point of this function: componentwise reflection of the
        # underlying uniform negates the corresponding normal, exactly, with
        # no dependence on any other uniform -- see its docstring. Checked
        # directly against the device intrinsic it is built on, over uniforms
        # kept away from the exact 0/1 endpoints (where the identity is still
        # true, in the sense that -Inf and +Inf are negatives of each other,
        # but isapprox is not going to agree).
        u = Array(CUDA.rand(Float64, 100_000))
        u = clamp.(u, 1.0e-12, 1.0 - 1.0e-12)
        z    = device_normcdfinv(u)
        zbar = device_normcdfinv(1.0 .- u)
        @test zbar ≈ -z
    end

    @testset "advances the counter exactly like randn! does" begin
        for n in (1, 2, 3, 41)
            a = PhiloxRNG(key); randn_inversion!(a, CUDA.zeros(Float64, n))
            b = PhiloxRNG(key); randn!(b, CUDA.zeros(Float64, n))
            @test get_state(a)[1] == get_state(b)[1] == cld(n, 2)
        end
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float64, 10)
        @test_throws ArgumentError randn_inversion!(rng, out)

        reset_substream!(rng)
        @test randn_inversion!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 0)
        randn_inversion!(rng, out)
        @test get_state(rng)[1] == 0
    end
end

@testset "PhiloxRNG CUDA randn_inversion! (Float32, comparison branch)" begin
    key = (UInt32(112), UInt32(223))

    @testset "statistics of a large draw" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 200_000)
        randn_inversion!(rng, out)
        v = Array(out)
        @test all(isfinite, v)
        @test abs(sum(v) / length(v)) < 0.02
        @test abs(sum(v .^ 2) / length(v) - 1.0) < 0.05
    end

    @testset "advances the counter four outputs per block" begin
        for n in (1, 2, 3, 4, 5, 83)
            rng = PhiloxRNG(key)
            randn_inversion!(rng, CUDA.zeros(Float32, n))
            @test get_state(rng)[1] == cld(n, 4)
        end
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float32, 10)
        @test_throws ArgumentError randn_inversion!(rng, out)

        reset_substream!(rng)
        @test randn_inversion!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 0)
        randn_inversion!(rng, out)
        @test get_state(rng)[1] == 0
    end
end

# A launch grid smaller than the kernel's index range leaves the tail of the
# array untouched, and a buffer that starts as zeros hides it: the statistics
# tests above only notice through a wrong variance. A NaN sentinel makes "every
# element was written" an exact check, for every GPU fill in the extension.
@testset "PhiloxRNG CUDA fills write every element" begin
    key = (UInt32(114), UInt32(225))
    fills = [
        ("rand! Float64",             Float64, rand!),
        ("rand! Float32",             Float32, rand!),
        ("rand! Float16",             Float16, rand!),
        ("randn! Float64",            Float64, randn!),
        ("randn! Float32",            Float32, randn!),
        ("randn_inversion! Float64",  Float64, randn_inversion!),
        ("randn_inversion! Float32",  Float32, randn_inversion!),
        ("randn_polar! Float64",      Float64, randn_polar!),
    ]
    # Sizes chosen to be awkward: not a multiple of 2, 4, 256 or 1024, and one
    # large enough that a quarter-sized grid is many blocks short.
    for (name, T, f) in fills, n in (1, 3, 1000, 100_003, 1_000_001)
        out = CUDA.fill(T(NaN), n)
        f(PhiloxRNG(key), out)
        @test all(isfinite, Array(out))
    end
end

@testset "PhiloxRNG CUDA randn_polar! (accept-reject, comparison branch)" begin
    key = (UInt32(113), UInt32(224))

    @testset "statistics of a large draw" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 200_000)
        randn_polar!(rng, out)
        v = Array(out)
        @test all(isfinite, v)
        @test abs(sum(v) / length(v)) < 0.02
        @test abs(sum(v .^ 2) / length(v) - 1.0) < 0.05
    end

    @testset "deterministic for a fixed key and counter" begin
        a = PhiloxRNG(key); out_a = CUDA.zeros(Float64, 37)
        b = PhiloxRNG(key); out_b = CUDA.zeros(Float64, 37)
        randn_polar!(a, out_a)
        randn_polar!(b, out_b)
        @test Array(out_a) == Array(out_b)
    end

    @testset "reserves _POLAR_BUDGET blocks per output pair" begin
        budget = Base.get_extension(RandomDataStreams, :RandomDataStreamsCUDAExt)._POLAR_BUDGET
        for n in (1, 2, 3, 41)
            rng = PhiloxRNG(key)
            randn_polar!(rng, CUDA.zeros(Float64, n))
            @test get_state(rng)[1] == cld(n, 2) * budget
        end
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float64, 10)
        @test_throws ArgumentError randn_polar!(rng, out)

        reset_substream!(rng)
        @test randn_polar!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 0)
        randn_polar!(rng, out)
        @test get_state(rng)[1] == 0
    end
end

@testset "PhiloxRNG CUDA randn! (Float32, Box-Muller)" begin
    key = (UInt32(77), UInt32(88))

    @testset "statistics of a large draw" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 200_000)
        randn!(rng, out)
        v = Array(out)
        @test all(isfinite, v)
        @test abs(sum(v) / length(v)) < 0.02
        @test abs(sum(v .^ 2) / length(v) - 1.0) < 0.05
    end

    @testset "advances the counter four outputs per block" begin
        for n in (1, 2, 3, 4, 5, 83)
            rng = PhiloxRNG(key)
            randn!(rng, CUDA.zeros(Float32, n))
            @test get_state(rng)[1] == cld(n, 4)
        end
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float32, 10)
        @test_throws ArgumentError randn!(rng, out)

        reset_substream!(rng)
        @test randn!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 0)
        randn!(rng, out)
        @test get_state(rng)[1] == 0
    end
end

@testset "philox4x32_10 / philox4x32_counter as a per-thread kernel primitive" begin
    # A minimal user kernel: thread i draws one Philox4x32 block from counter
    # (i, 0) under a fixed key, with no stream object and no shared state --
    # exactly the pattern advanced Monte Carlo kernels use these functions for.
    function _kernel!(out, key)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= length(out)
            ctr = philox4x32_counter(UInt64(i - 1), UInt64(0))
            blk = philox4x32_10(ctr, key)
            @inbounds out[i] = blk[1]
        end
        return nothing
    end

    key = (UInt32(5), UInt32(9))
    n = 1000
    out = CUDA.zeros(UInt32, n)
    @cuda threads = 256 blocks = cld(n, 256) _kernel!(out, key)

    expected = [RandomDataStreams.philox(philox4x32_counter(UInt64(i - 1), UInt64(0)), key, Val(10))[1]
                for i in 1:n]
    @test Array(out) == expected
end

@testset "open01 computes the same value on the device" begin
    # The conversion's one floating-point operation is a subtraction whose
    # exact result is representable, so IEEE 754 fixes every bit of it.
    function _open01_kernel!(out, words)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        i <= length(out) && (@inbounds out[i] = open01(words[i]))
        return nothing
    end
    for W in (UInt32, UInt64)
        words = vcat(W[0, 1, typemax(W), typemax(W) - 1, typemax(W) >> 1], rand(PhiloxRNG(UInt32[1, 2]), W, 10_000))
        out = CUDA.zeros(W === UInt32 ? Float32 : Float64, length(words))
        @cuda threads = 256 blocks = cld(length(words), 256) _open01_kernel!(out, CuArray(words))
        @test Array(out) == open01.(words)
    end
end

end # CUDA.functional()
