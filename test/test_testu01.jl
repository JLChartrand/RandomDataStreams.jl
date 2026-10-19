
using RandomDataStreams
using Test

@isdefined(check_smallcrush) || include("testu01_support.jl")

# TestU01 calls back into Julia through a C function pointer, so the callback
# must be inferred as Float64: the generator tables below have an abstract
# element type, and a closure written over one of their entries directly would
# return Any. Going through this barrier specialises on the runtime type and
# closes over something concrete.
single_stream(rng) = () -> rand(rng, Float64)


@testset "battery guard" begin
    # The platform failure is skipped; anything else must not be swallowed.
    # Verified here rather than trusted, since the skip branch is unreachable
    # on the platforms where the suite normally runs.
    @test run_battery(() -> error("cfunction: closures are not supported on this platform"),
                      "guard") === nothing
    @test run_battery(() -> 42, "guard") == 42
    @test_throws ErrorException run_battery(() -> error("boom"), "guard")
end

@testset "the callback hands TestU01 what a native generator would" begin
    # TestU01's own MRG32k3a and the package's give the same numbers from the
    # published state, one from inside the library and one through the Julia
    # callback every battery here goes through, so their SmallCrush statistics
    # must agree double for double. A difference would mean the path itself
    # changes what the batteries see. The native generator needs no callback,
    # and runs where the closures do not.
    native = TU01.withgen(TU01.ulec_mrg32k3a()) do g
        silently(() -> TU01.smallcrush(g))
    end
    @test length(native) == 15
    ours = check_smallcrush(() -> TU01.Gen(single_stream(MRG32k3a()), "MRG32k3a()"),
                            "SmallCrush / MRG32k3a()")
    ours === nothing || @test [s.p for s in native] == [s.p for s in ours]
    # and TestU01's AES, the reference the CollisionOver calibration uses
    aes = TU01.withgen(TU01.aes(UInt8.(1:16))) do g
        silently(() -> TU01.smallcrush(g))
    end
    @test length(aes) == 15 && isempty(TU01.suspects(aes))
end

@testset "CollisionOver on small alphabets: the corrected law" begin
    # At q = 0 every clump is one collision and the recursion is the Poisson
    # law, which can be checked term by term.
    λ = 3.5
    pois(n) = exp(-λ) * λ^n / Float64(factorial(big(n)))
    for c in 0:15
        left, right = TU01.polya_aeppli_tails(c, λ, 0)
        @test left ≈ sum(pois, 0:c) rtol = 1e-12
        @test right ≈ sum(pois, c:60) rtol = 1e-12
    end
    # At q > 0, the mean it was given and the variance clumping predicts.
    μ, q = 50.0, 1 / 4
    cdf = [TU01.polya_aeppli_tails(c, μ, q)[1] for c in 0:600]
    pmf = diff([0.0; cdf])
    m = sum((0:600) .* pmf)
    @test sum(pmf) ≈ 1 rtol = 1e-12
    @test m ≈ μ rtol = 1e-10
    @test sum(((0:600) .- m) .^ 2 .* pmf) ≈ μ * (1 + q) / (1 - q) rtol = 1e-10

    # Pinned to the reanalysis of the archived BigCrush campaigns, which
    # computes the corrected law another way -- a Poisson mixture of negative
    # binomials, with SciPy (reanalyze.py, papers repository, bd50b70) -- and to
    # the p-values TestU01 returned for the same counts, in those runs'
    # pvalues.tsv.
    for (count, d, t, testu01, corrected) in [
            (1500, 4, 21, 0.00015403264362272764, 0.0026767217370104613),  # Xoshiro512pp, single
            (1243, 4, 21, 0.9995407424782214, 0.994996584218327),          # MRG32k3a, 16 x 8
            (1506, 8, 14, 8.3050441977342e-5, 0.00046289324023464324),     # Philox4x32-10, 16 x 8
            (1242, 8, 14, 0.999584584588777, 0.9984335349022355),          # Xoroshiro128p, 64 x 1
            (1482, 2^21, 2, 0.0008585563014616375, 0.0008585607752520025)] # Threefry4x32-20, 256 x 2
        @test TU01.collisionover_p(count, 30, 20_000_000, d, t; q = 0) ≈ testu01 atol = 1e-9
        @test TU01.collisionover_p(count, 30, 20_000_000, d, t) ≈ corrected atol = 1e-12
    end

    # The correction never makes a class more severe, at any count a run can
    # plausibly give. replay.jl relies on it: it starts from the tests TestU01
    # singled out and drops the ones the correction clears, which can only be
    # right if the correction never singles out a test TestU01 did not.
    severity = Dict(:pass => 0, :suspect => 1, :fail => 2)
    worse = [(d, c) for (d, t) in ((2^21, 2), (2^14, 3), (64, 7), (8, 14), (4, 21))
             for c in 0:3000
             if severity[TU01.classify(TU01.collisionover_p(c, 30, 20_000_000, d, t))] >
                severity[TU01.classify(TU01.collisionover_p(c, 30, 20_000_000, d, t; q = 0))]]
    @test isempty(worse)
end

@testset "CollisionOver on small alphabets: the counts, from the log" begin
    # Two CollisionOver blocks as TestU01 prints them, and between them another
    # test that prints a collision count of its own, which must not be taken.
    block(r, d, t, count) = """
        smultin_MultinomialOver test:
        -----------------------------------------------
           N = 30,  n = 20000000,  r = $(lpad(r, 2)),   d = $d,   t = $(lpad(t, 2)),
               Sparse =   TRUE

        -----------------------------------------------
        Results of CollisionOver test:

        POISSON approximation                 :
        Expected number of collisions = N*Mu  :     1364.24
        Observed number of collisions         :     $count
        p-value of test                       :    0.59

        """
    permut = """
        smultin_Multinomial test:
        -----------------------------------------------
           N = 20,  n = 20000000,  r =  0,   t = 14,
        -----------------------------------------------
        Test Results for Collisions

        Expected number of collisions = N*Mu  :    45879.47
        Observed number of collisions         :    45775

        """
    log = tempname()
    write(log, block(0, 4, 21, 1500) * permut * block(28, 4, 21, 1243))
    @test TU01.collision_counts(log) ==
          [(N = 30, n = 20_000_000, r = 0, d = 4, t = 21, count = 1500),
           (N = 30, n = 20_000_000, r = 28, d = 4, t = 21, count = 1243)]
    @test TU01.corrected_tests(:bigcrush, log) ==
          Dict(11 => TU01.collisionover_p(1500, 30, 20_000_000, 4, 21),
               12 => TU01.collisionover_p(1243, 30, 20_000_000, 4, 21))
    @test isempty(TU01.corrected_tests(:crush, log))

    # What a repeat battery of tests 11 and 12 returns, with TestU01's p-values
    # for those counts: corrected in place, judged on the corrected p-value.
    stats = [TU01.Statistic(0, "CollisionOver, t = 21", 0.00015403264362272764),
             TU01.Statistic(1, "CollisionOver, t = 21", 0.9995407424782214)]
    fixed = TU01.correct(:bigcrush, stats, log)
    @test [s.p for s in fixed] == [s.p for s in stats]
    @test fixed[1].corrected ≈ 0.0026767217370104613 atol = 1e-12
    @test fixed[2].corrected ≈ 0.994996584218327 atol = 1e-12
    @test TU01.classify(stats[1]) === :suspect && TU01.classify(fixed[1]) === :pass
    @test isempty(TU01.suspects(fixed)) && length(TU01.suspects(stats)) == 2
    # Crush is not corrected, and a statistic of another test is left alone.
    @test TU01.correct(:crush, stats, log) == stats
    other = [TU01.Statistic(0, "SerialOver, r = 0", 0.5); stats]
    @test TU01.correct(:bigcrush, other, log)[1] === other[1]
    # A log that does not account for the statistics is refused, not guessed at.
    @test_throws ErrorException TU01.correct(:bigcrush, stats[1:1], log)        # one block too many
    @test_throws ErrorException TU01.correct(:bigcrush, reverse(stats), log)   # paired wrong
    quiet = tempname()
    write(quiet, permut)
    @test_throws ErrorException TU01.correct(:bigcrush, stats, quiet)
    rm(log); rm(quiet)
end

@testset "TestU01 SmallCrush Validation" begin
    # TestU01 tests take a bit of time, but validate the generator's statistical quality
    
    generators = [
        ("MRG32k3a", MRG32k3aGen),
        ("MRG63k3a", MRG63k3aGen),
        ("Xoshiro256+", () -> Xoshiro256plusGen([0x01, 0x02, 0x03, 0x04])),
        ("Philox4x32-10", PhiloxGen),
        ("Philox4x64-10", Philox4x64Gen),
        ("Threefry4x64-20", Threefry4x64Gen),
        ("Threefry4x32-20", Threefry4x32Gen),
        ("PCG64", () -> PCG64Gen(20260830)),
        ("PCG64DXSM", () -> PCG64DXSMGen(20260830))
    ]

    for (name, gen_init) in generators
        @testset "$name" begin
            rng = next_stream!(gen_init())          # Float64 in (0, 1)
            check_smallcrush(() -> TU01.Gen(single_stream(rng), name),
                             "SmallCrush / $name")
        end
    end
end
