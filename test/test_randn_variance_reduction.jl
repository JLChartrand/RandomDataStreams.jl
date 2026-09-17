# Why branch philox-gpu-randn-test exists: a portable (no GPU, no CUDA.jl)
# check of the actual mathematical claim behind preferring inversion over
# Box-Muller (and accept-reject in general) for normal variates that feed
# antithetic variates or RQMC.
#
# Those variance-reduction techniques manipulate the *uniform* stream
# componentwise -- reflecting one coordinate (u -> 1 - u), or replacing it
# with a low-discrepancy point -- and rely on the uniform -> normal transform
# to carry that structure through unchanged in every other coordinate. That
# only holds for a transform that is (a) monotonic and (b) applied one
# uniform to one normal, with no cross-talk between coordinates. Inversion,
# Z = Phi^-1(U), is exactly that. Box-Muller is neither: it mixes a *pair* of
# uniforms nonlinearly (one sets a radius, the other an angle), so reflecting
# just one of them does not simply negate one output -- it changes an output
# that is not even "at" that uniform, in an unrelated way. The GPU kernels
# that put this into practice (ext/RandomDataStreamsCUDAExt.jl's
# `randn_inversion!`, `randn!`, `randn_polar!`) need CUDA and a real device
# to exercise (see test_cuda.jl); the underlying claim does not, since it is
# a fact about the two transforms themselves, independent of how their
# uniform inputs were generated.
#
# `ltqnorm` below is Peter Acklam's rational approximation to Phi^-1
# (2003; https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/),
# used here only as a portable, CPU-only, self-contained stand-in for the
# CUDA `normcdfinv` intrinsic `randn_inversion!` calls on the GPU -- not
# exported, not part of the package's public API. Its two branches are
# mirror images of each other by construction (the upper-tail branch is
# exactly the negated lower-tail formula), so it satisfies the antithetic
# identity in the test below by construction, not by coincidence -- which
# is itself part of the point: *any* correct implementation of Phi^-1 must
# satisfy it, since it is a property of the mathematical function, not of
# one particular polynomial fit.

function ltqnorm(p::Float64)
    a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02;  a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02;  b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01;  b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00;  c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03;  d2 = 3.224671290700398e-01;  d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    elseif p <= p_high
        q = p - 0.5; r = q * q
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    end
end

@testset "why inversion, for antithetic/RQMC normal variates" begin

    @testset "ltqnorm sanity: known quantiles" begin
        # Standard reference values (R's qnorm / any statistical table),
        # checked to Acklam's own advertised bound (relative error < 1.5e-9).
        for (p, expected) in ((0.5, 0.0), (0.975, 1.9599639845400545),
                              (0.995, 2.5758293035489004), (0.025, -1.9599639845400545),
                              (0.99, 2.3263478740408408))
            @test isapprox(ltqnorm(p), expected; atol = 3e-9)
        end
    end

    @testset "inversion: Phi^-1(1 - u) == -Phi^-1(u), to near machine precision" begin
        # Margin kept at 1e-6 (not all the way to 0/1): Acklam's rational fit
        # has a *relative* error bound in p-space, which the inverse function
        # amplifies out in x-space as p approaches the endpoints (the density
        # in the denominator shrinks faster than the fit's error does) --
        # a property of this one polynomial approximation used for testing,
        # not of Phi^-1 itself, whose exact identity holds all the way to the
        # endpoints (where it reads -Inf == -(Inf)).
        rng = Xoshiro256p(UInt64[1, 2, 3, 4])
        for _ in 1:200_000
            u = rand(rng) * (1 - 2e-6) + 1e-6
            @test isapprox(ltqnorm(1 - u), -ltqnorm(u); atol = 1e-8)
        end
    end

    @testset "inversion: reflecting ONE coordinate of a many-uniform draw touches only that coordinate" begin
        # The property antithetic/RQMC schemes actually need: reflecting a
        # single uniform in a vector must not perturb the normals built from
        # the other coordinates. True by construction for a componentwise
        # transform (each Z_i is a function of U_i alone); asserted here as a
        # regression net on `ltqnorm` itself, and as documentation of exactly
        # what "componentwise" is buying.
        rng = Xoshiro256p(UInt64[5, 6, 7, 8])
        u = rand(rng, 8)
        z = ltqnorm.(u)
        for i in eachindex(u)
            u2 = copy(u); u2[i] = 1 - u2[i]
            z2 = ltqnorm.(u2)
            for j in eachindex(u)
                if j == i
                    @test isapprox(z2[j], -z[j]; atol = 1e-9)
                else
                    @test z2[j] == z[j]
                end
            end
        end
    end

    @testset "Box-Muller: reflecting one coordinate does NOT negate one output" begin
        # z1 = r*cos(theta) with r = sqrt(-2*log(1-u1)), theta = 2*pi*u2 --
        # the transform this package's GPU `randn!` (Box-Muller) uses.
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
        # changes r nonlinearly while reflecting u2 only rotates theta by pi,
        # and cos/sin of (theta + pi) is a sign flip that does not compose
        # with the mismatched r' the way it would need to.
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
