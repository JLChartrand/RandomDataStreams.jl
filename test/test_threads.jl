using RandomDataStreams
using Test
using Base.Threads

# Parallel use is the reason this package exists, and it was the one claim the
# suite never checked. Two things are verified here.
#
#   1. Streams carry no shared state, so drawing from one stream per thread
#      gives exactly what drawing from them one after another gives. The serial
#      run is the oracle: any hidden global, cache or aliasing between streams
#      would show up as a mismatch.
#
#   2. `next_stream!(gen, n)` hands out the same streams as n successive calls.
#      It exists so that the safe pattern -- take the streams serially, then
#      parallelise over them -- is shorter than the unsafe one.
#
# Concurrency is only genuinely exercised when Julia is started with more than
# one thread. The assertions hold either way; with `-t 1` they degenerate into
# a second serial run, which is why CI sets JULIA_NUM_THREADS.

const THREAD_FAMILIES = Pair{String,Any}[
    "MRG32k3a"        => () -> MRG32k3aGen(12345),
    "MRG63k3a"        => () -> MRG63k3aGen(12345),
    "Xoshiro256pp"    => () -> Xoshiro256ppGen(12345),
    "Xoroshiro128ss"  => () -> Xoroshiro128ssGen(12345),
    "PCG64"           => () -> PCG64Gen(12345),
    "PCG64DXSM"       => () -> PCG64DXSMGen(12345),
    "Philox4x32-10"   => () -> PhiloxGen(12345),
    "Philox4x64-10"   => () -> Philox4x64Gen(12345),
    "Threefry4x64-20" => () -> Threefry4x64Gen(12345),
]

const NSTREAMS = 8
const NDRAWS   = 10_000

@testset "one stream per thread matches the serial run" begin
    nthreads() == 1 && @info "running with one thread: the parallel assertions " *
                             "still hold but no concurrency is exercised " *
                             "(set JULIA_NUM_THREADS to test it properly)."

    for (name, make) in THREAD_FAMILIES
        @testset "$name" begin
            gen = make()
            serial = [sum(rand(rng) for _ in 1:NDRAWS)
                      for rng in next_stream!(gen, NSTREAMS)]

            gen = make()
            rngs = next_stream!(gen, NSTREAMS)
            parallel = Vector{Float64}(undef, NSTREAMS)
            @threads for t in 1:NSTREAMS
                s = 0.0
                r = rngs[t]
                for _ in 1:NDRAWS
                    s += rand(r)
                end
                parallel[t] = s
            end

            @test parallel == serial

            # Distinct streams, not the same one handed out twice: a factory
            # that forgot to advance its seed would pass the test above.
            @test length(unique(serial)) == NSTREAMS
        end
    end
end

@testset "next_stream!(gen, n) is n successive calls" begin
    for (name, make) in THREAD_FAMILIES
        @testset "$name" begin
            g1 = make()
            batch = next_stream!(g1, 4)

            g2 = make()
            one_at_a_time = [next_stream!(g2) for _ in 1:4]

            @test [rand(r) for r in batch] == [rand(r) for r in one_at_a_time]

            # The generator is left in the same place either way, so a batch
            # call can be followed by more calls without a gap or an overlap.
            @test rand(next_stream!(g1)) == rand(next_stream!(g2))
        end
    end

    gen = MRG32k3aGen(12345)
    @test isempty(next_stream!(gen, 0))
    @test_throws ArgumentError next_stream!(gen, -1)
end

@testset "generator objects are documented as not thread-safe" begin
    # Not a race detector: a data race is not deterministic, so asserting that
    # concurrent `next_stream!` produces collisions would be a flaky test. What
    # is checked is that the warning a user needs is actually in the docstring
    # they will read.
    doc = string(@doc next_stream!)
    @test occursin("thread", lowercase(doc))
    @test occursin("race", lowercase(doc))
end
