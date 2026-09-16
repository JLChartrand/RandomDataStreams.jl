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

if !CUDA.functional()
    @info "CUDA is not functional on this machine -- skipping GPU tests" CUDA.functional()
else

@testset "PhiloxRNG CUDA fill" begin
    key = (UInt32(11), UInt32(22))

    @testset "matches the CPU path bit for bit" begin
        for n in (1, 2, 3, 100, 10_001)          # 1 and odd n exercise the tail pairing
            cpu = PhiloxRNG(key)
            cpu_out = Vector{Float64}(undef, n)
            rand!(cpu, cpu_out)

            gpu = PhiloxRNG(key)
            gpu_out = CUDA.zeros(Float64, n)
            rand!(gpu, gpu_out)

            @test Array(gpu_out) == cpu_out
            # both paths must advance the counter by the same number of blocks
            @test get_state(gpu)[1] == get_state(cpu)[1]
        end
    end

    @testset "values land in [0, 1)" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 50_000)
        rand!(rng, out)
        v = Array(out)
        @test all(0.0 .<= v .< 1.0)
        @test 0.48 < sum(v) / length(v) < 0.52
    end

    @testset "continues from where the GPU fill left off" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 20)
        rand!(rng, out)
        after_gpu = rand(rng)                     # next scalar draw, on the CPU

        ref = PhiloxRNG(key)
        ref_out = Vector{Float64}(undef, 20)
        rand!(ref, ref_out)
        after_cpu = rand(ref)
        @test after_gpu == after_cpu
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)                                 # one scalar draw -> mid-block
        out = CUDA.zeros(Float64, 10)
        @test_throws ArgumentError rand!(rng, out)

        reset_substream!(rng)                     # realigned -> works again
        @test rand!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float64, 0)
        rand!(rng, out)
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

@testset "PhiloxRNG CUDA fill (Float32, native word-per-output)" begin
    key = (UInt32(55), UInt32(66))

    @testset "matches a direct block computation, four outputs per block" begin
        for n in (1, 3, 4, 5, 1000, 4001)         # exercise every within-block offset
            rng = PhiloxRNG(key)
            out = CUDA.zeros(Float32, n)
            rand!(rng, out)
            v = Array(out)

            expected = Float32[]
            b = 0
            while length(expected) < n
                ctr = philox4x32_counter(UInt64(0), UInt64(b))
                blk = RandomDataStreams.philox(ctr, key, Val(10))
                append!(expected, close_open01.(blk))
                b += 1
            end
            @test v == expected[1:n]
            @test get_state(rng)[1] == cld(n, 4)
        end
    end

    @testset "values land in [0, 1)" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 50_000)
        rand!(rng, out)
        v = Array(out)
        @test all(0.0f0 .<= v .< 1.0f0)
        @test 0.48 < sum(v) / length(v) < 0.52
    end

    @testset "refuses a mid-block generator" begin
        rng = PhiloxRNG(key)
        rand(rng)
        out = CUDA.zeros(Float32, 10)
        @test_throws ArgumentError rand!(rng, out)

        reset_substream!(rng)
        @test rand!(rng, out) === out
    end

    @testset "empty array is a no-op" begin
        rng = PhiloxRNG(key)
        out = CUDA.zeros(Float32, 0)
        rand!(rng, out)
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

end # CUDA.functional()
