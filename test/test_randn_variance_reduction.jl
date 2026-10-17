# Why `randn` draws by inversion: the mathematical claim behind the choice,
# checked on the CPU against the alternatives.
#
# Antithetic variates and randomized quasi-Monte Carlo manipulate the *uniform*
# stream componentwise -- reflecting one coordinate (u -> 1 - u), or replacing
# it with a low-discrepancy point -- and rely on the uniform -> normal transform
# to carry that structure through unchanged in every other coordinate. That
# only holds for a transform that is (a) monotone and (b) applied one uniform to
# one normal, with no cross-talk between coordinates. Inversion,
# Z = Phi^-1(U), is exactly that, and is what `randn` computes, as `norminv`.
#
# The two alternatives this file checks against are neither. Box-Muller, which
# `randn_boxmuller!` runs on the GPU, mixes a *pair* of uniforms nonlinearly
# (one sets a radius, the other an angle), so reflecting one of them does not
# negate one output. The standard library's Ziggurat, which `randn` used before,
# takes the sign, the layer and the abscissa from different bits of one word:
# complementing the word -- which is what reflecting the uniform does to it,
# see floats.jl -- does not negate the normal either. docs/src/streams.md
# tabulates the comparison.

using RandomDataStreams
using Random
using Test

# A generator that hands the standard library either the words of a Xoshiro or
# their complements -- the words of the reflected uniforms, since
# 1 - _u01(w) == _u01(~w). For the Ziggurat counterexample below.
struct Words <: AbstractRNG
    r::Xoshiro
    complement::Bool
end
Random.rng_native_52(::Words) = UInt64
Random.rand(w::Words, ::Random.SamplerType{UInt64}) =
    w.complement ? ~rand(w.r, UInt64) : rand(w.r, UInt64)

@testset "why inversion, for antithetic/RQMC normal variates" begin

    @testset "norminv: known quantiles" begin
        # Phi^-1 of each Float64 u, rounded from 256-bit MPFR arithmetic
        # (bisection then Newton on Phi = erfc/2), at a point of each of AS 241's
        # three regions, their boundaries and the extremes of the uniform grid.
        for (u, z) in ((0.5, 0.0),
                       (0.975, 1.9599639845400538),
                       (0.025, -1.9599639845400543),
                       (0.075, -1.439531470938456),
                       (1.0e-10, -6.361340902404057),
                       (exp(-25.0), -6.657904643501103),
                       (2.0^-53, -8.209536151601387),
                       (1 - 2.0^-53, 8.209536151601387))
            @test isapprox(norminv(u), z; rtol = 2e-15, atol = 1e-300)
        end
        @test norminv(0.0) == -Inf && norminv(1.0) == Inf
        @test isnan(norminv(-0.5)) && isnan(norminv(1.5)) && isnan(norminv(NaN))
        # the single-precision map, to its own accuracy
        @test isapprox(norminv(0.975f0), Float32(norminv(Float64(0.975f0))); rtol = 4e-7)
    end

    @testset "norminv(1 - u) == -norminv(u), exactly, on the uniform grid" begin
        # The word generators' uniforms are odd multiples of 2^-53, a grid
        # symmetric about 1/2 on which 1 - u is exact and u - 1/2 too, and
        # norminv works from |u - 1/2| and its sign -- so the identity is exact,
        # all the way to the extremes, not merely to the fit's accuracy.
        rng = Xoshiro256p(UInt64[1, 2, 3, 4])
        for _ in 1:200_000
            u = rand(rng)
            @test norminv(1 - u) == -norminv(u)
        end
        for k in 0:999
            u = (2k + 1) * 2.0^-53
            @test norminv(1 - u) == -norminv(u)
        end
        # and in single precision, on the Float32 grid of `randn32`
        for k in 0:(2^23 - 1)
            u = Float32((2k + 1) * 2.0^-24)
            norminv(1f0 - u) == -norminv(u) || @test false
        end
    end

    @testset "an antithetic stream of uniforms gives the negated normals" begin
        for mk in (() -> next_stream!(PhiloxGen(7)), () -> next_stream!(PCG64Gen(7)),
                   () -> next_stream!(Xoshiro256ppGen(7)))
            u = rand(mk(), 1000)
            @test norminv.(1 .- u) == -randn(mk(), 1000)
        end
    end

    @testset "inversion: reflecting ONE coordinate of a many-uniform draw touches only that coordinate" begin
        # The property antithetic/RQMC schemes actually need: reflecting a
        # single uniform in a vector must not perturb the normals built from
        # the other coordinates. True by construction for a componentwise
        # transform (each Z_i is a function of U_i alone).
        rng = Xoshiro256p(UInt64[5, 6, 7, 8])
        u = rand(rng, 8)
        z = norminv.(u)
        for i in eachindex(u)
            u2 = copy(u); u2[i] = 1 - u2[i]
            z2 = norminv.(u2)
            for j in eachindex(u)
                @test z2[j] == (j == i ? -z[j] : z[j])
            end
        end
    end

    @testset "Ziggurat: complementing the word does NOT negate the normal" begin
        # The standard library's randn, on the words of a Xoshiro and on their
        # complements (`Words`, above).
        negated = 0
        for seed in 1:1000
            z  = Random.randn(Words(Xoshiro(seed), false))
            zc = Random.randn(Words(Xoshiro(seed), true))
            zc == -z && (negated += 1)
        end
        @test negated == 0
    end

    @testset "Box-Muller: reflecting one coordinate does NOT negate one output" begin
        # z1 = r*cos(theta) with r = sqrt(-2*log(1-u1)), theta = 2*pi*u2 --
        # the transform `randn_boxmuller!` uses on the GPU.
        bm_z1(u1, u2) = sqrt(-2 * log(1 - u1)) * cos(2 * pi * u2)

        rng = Xoshiro256p(UInt64[9, 10, 11, 12])
        mismatches = 0
        n = 1000
        for _ in 1:n
            u1, u2 = rand(rng), rand(rng)
            z1 = bm_z1(u1, u2)
            z1_reflect_u1 = bm_z1(1 - u1, u2)          # reflect only u1, as a componentwise scheme would
            isapprox(z1_reflect_u1, -z1; atol = 1e-9) && (mismatches += 1)
        end
        # Exact equality only at the measure-zero fixed point u1 == 1 - u1;
        # generically it fails every time.
        @test mismatches == 0
    end

    @testset "Box-Muller: reflecting BOTH coordinates does not negate the pair either" begin
        # Even the more generous reflection -- negate the whole underlying
        # pair, not just one coordinate -- fails, because reflecting u1
        # changes r nonlinearly while reflecting u2 only mirrors theta, and
        # cos(2pi - theta) = cos(theta) keeps the first output's sign.
        bm(u1, u2) = (r = sqrt(-2 * log(1 - u1)); theta = 2 * pi * u2; (r * cos(theta), r * sin(theta)))

        rng = Xoshiro256p(UInt64[13, 14, 15, 16])
        mismatches = 0
        n = 1000
        for _ in 1:n
            u1, u2 = rand(rng), rand(rng)
            z1, z2 = bm(u1, u2)
            z1r, z2r = bm(1 - u1, 1 - u2)
            (isapprox(z1r, -z1; atol = 1e-9) && isapprox(z2r, -z2; atol = 1e-9)) && (mismatches += 1)
        end
        @test mismatches == 0
    end

end
