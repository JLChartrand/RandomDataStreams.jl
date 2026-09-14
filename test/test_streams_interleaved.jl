# Dependence BETWEEN streams, not the quality of each stream in isolation.
#
# L'Ecuyer, Nadeau-Chamard, Chen & Lebar (2021), Sec. 6: "For RNGs with multiple
# streams and substreams (including splittable, counter-based, etc.), it is also
# important to test the dependence between those streams. For that, one can
# construct sequences that take a few values from each stream for a certain
# number of streams, in a round-robin fashion [...] a power of 2 from 16 to 1024
# [streams], [...] a fixed number of values per stream (from 1 to 8)."
#
# For a counter-based generator this also exercises the key schedule
# (`stream_key`): a schedule that hands out structurally related keys shows up
# here, not in a battery run on a single stream.

using RandomDataStreams
using Test

@isdefined(check_smallcrush) || include("testu01_support.jl")


# TestU01 calls the callback from C, so it has to be a `Function` whose return
# type infers to Float64; a mutable functor behind a closure gives both.
mutable struct RoundRobin{R}
    streams::Vector{R}
    s::Int
    c::Int
    per::Int
end

function (g::RoundRobin)()
    v = rand(g.streams[g.s], Float64)
    g.c += 1
    if g.c == g.per
        g.c = 0
        g.s = g.s == length(g.streams) ? 1 : g.s + 1
    end
    return v
end

function interleaved(gen_init, nstreams::Int, per_stream::Int)
    gen = gen_init()
    g = RoundRobin([next_stream!(gen) for _ in 1:nstreams], 1, 0, per_stream)
    return () -> g()
end

@testset "TestU01 SmallCrush on interleaved streams" begin
    generators = [
        ("MRG32k3a", MRG32k3aGen),
        ("MRG63k3a", MRG63k3aGen),
        ("Xoshiro256+", () -> Xoshiro256plusGen([0x01, 0x02, 0x03, 0x04])),
        ("Philox4x32-10", PhiloxGen),
        ("Philox4x64-10", Philox4x64Gen),
        ("Threefry4x64-20", Threefry4x64Gen),
        ("Threefry4x32-20", Threefry4x32Gen),
        ("PCG64", () -> PCG64Gen(20260830)),
        ("PCG64DXSM", () -> PCG64DXSMGen(20260830)),
    ]

    for (name, gen_init) in generators
        @testset "$name" begin
            f = interleaved(gen_init, 64, 1)
            # `TU01.Gen` refuses a callback that does not infer as Float64,
            # which is what would otherwise crash the C side.
            check_smallcrush(() -> TU01.Gen(f, name), "interleaved / $name")
        end
    end
end
