
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
