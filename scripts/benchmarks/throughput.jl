#!/usr/bin/env julia

# Reproducible throughput benchmark for every generator of the package.
#
#     julia -O3 scripts/benchmarks/throughput.jl
#
# Measurement method, fixed deliberately -- naive loops give answers that differ
# by nearly a factor of two, so the method has to be stated with the numbers:
#
#   * BenchmarkTools, minimum time over the samples. The minimum is the right
#     statistic here: the work per draw is constant, so everything above the
#     minimum is interference from the rest of the machine.
#
#   * Scalar draws are accumulated with `xor` on the raw bits. An accumulator is
#     needed so the loop cannot be optimised away, and `xor` has a one-cycle
#     latency: a floating-point `+` chain has four, which is longer than a draw
#     from the fastest generators here and would measure the adder instead.
#
#   * Nothing is written to memory in the scalar loop. Storing each draw to an
#     array makes the measurement one of memory bandwidth: on this machine the
#     same xoshiro generator reads 867 M/s with an accumulator and 481 M/s
#     storing into an 8 MB vector.
#
#   * The bulk figures use `rand!` into a 4096-element vector, small enough to
#     stay in cache, so they measure generation and not the memory system.
#
# On a hybrid CPU -- Intel 12th generation and later, with performance and
# efficiency cores -- pin the process before quoting anything, or the scheduler
# will move it between core types mid-run and the minimum will come from
# whichever core happened to be fastest:
#
#     taskset -c 0 julia -O3 scripts/benchmarks/throughput.jl
#
# and say in the write-up which core type was used. A faster hybrid chip gives
# *less* trustworthy numbers than a uniform one unless this is done.

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, Random, BenchmarkTools, Printf

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
const RDS = RandomDataStreams

const N_SCALAR = 100_000        # draws per benchmark sample
const N_BULK = 4096             # elements per rand! call: 32 KB, cache resident

# Seeds are expanded through splitmix64 from one fixed integer: a low-entropy
# state such as all-ones takes a while to mix and is a poor starting point for
# a battery, while a fixed integer keeps the runs reproducible.
seed_words(n::Int) = RandomDataStreams._splitmix_words(UInt64(12345), n)

@inline _bits(x::Float64) = reinterpret(UInt64, x)
@inline _bits(x::UInt64) = x
@inline _bits(x::UInt32) = UInt64(x)

function xor_loop(rng, ::Type{T}, n) where {T}
    acc = zero(UInt64)
    for _ in 1:n
        acc ⊻= _bits(rand(rng, T))
    end
    return acc
end

function xor_loop_f64(rng, n)
    acc = zero(UInt64)
    for _ in 1:n
        acc ⊻= reinterpret(UInt64, rand(rng))
    end
    return acc
end

"Millions of draws per second, from the minimum sample time."
mdraws(b, n) = n / (minimum(b).time * 1e-9) / 1e6

# Blocks must be produced from independent counters, the way the generator
# actually uses them. Feeding each block back as the next counter measures the
# latency of the cipher rather than its throughput, and understates it by
# enough to make the block rate look lower than the draw rate built on it.
function block_loop(f, ctr::NTuple{4,W}, key, n) where {W}
    acc = zero(W)
    for i in 1:n
        blk = f((W(i), ctr[2], ctr[3], ctr[4]), key)
        acc ⊻= blk[1] ⊻ blk[2] ⊻ blk[3] ⊻ blk[4]
    end
    return acc
end

function main()
    println("RandomDataStreams throughput")
    println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
    println(provenance_line())
    println("BenchmarkTools, minimum of the samples; ", N_SCALAR,
            " draws per scalar sample, ", N_BULK, " per rand! call")
    println(repeat("-", 78), "\n")

    generators = [
        ("MRG32k3a",        () -> MRG32k3a()),
        ("MRG63k3a",        () -> MRG63k3a()),
        ("Xoroshiro128p",   () -> Xoroshiro128p(seed_words(2))),
        ("Xoroshiro128ss",  () -> Xoroshiro128ss(seed_words(2))),
        ("Xoroshiro128pp",  () -> Xoroshiro128pp(seed_words(2))),
        ("Xoshiro256p",     () -> Xoshiro256p(seed_words(4))),
        ("Xoshiro256ss",    () -> Xoshiro256ss(seed_words(4))),
        ("Xoshiro256pp",    () -> Xoshiro256pp(seed_words(4))),
        ("Xoshiro512p",     () -> Xoshiro512p(seed_words(8))),
        ("Xoshiro512ss",    () -> Xoshiro512ss(seed_words(8))),
        ("Xoshiro512pp",    () -> Xoshiro512pp(seed_words(8))),
        ("PCG64",           () -> PCG64(seed_words(2))),
        ("PCG64DXSM",       () -> PCG64DXSM(seed_words(2))),
        ("Philox4x32-10",   () -> next_stream!(PhiloxGen())),
        ("Philox4x64-10",   () -> next_stream!(Philox4x64Gen())),
        ("Threefry4x32-20", () -> next_stream!(Threefry4x32Gen())),
        ("Threefry4x64-20", () -> next_stream!(Threefry4x64Gen())),
    ]

    println("Scalar draws: millions per second, and nanoseconds per draw\n")
    @printf("%-18s %9s %6s %9s %6s %9s %6s\n", "generator",
            "Float64", "ns", "UInt64", "ns", "UInt32", "ns")
    for (name, mk) in generators
        rng = mk()
        f = mdraws(@benchmark(xor_loop_f64($rng, $N_SCALAR)), N_SCALAR)
        u64 = mdraws(@benchmark(xor_loop($rng, UInt64, $N_SCALAR)), N_SCALAR)
        u32 = mdraws(@benchmark(xor_loop($rng, UInt32, $N_SCALAR)), N_SCALAR)
        @printf("%-18s %9.0f %6.2f %9.0f %6.2f %9.0f %6.2f\n", name,
                f, 1000 / f, u64, 1000 / u64, u32, 1000 / u32)
    end

    println("\nArray fill with rand!, millions of elements per second:\n")
    @printf("%-18s %12s %12s %12s\n", "generator", "Float64", "UInt64", "UInt32")
    for (name, mk) in generators
        rng = mk()
        vf = Vector{Float64}(undef, N_BULK)
        v64 = Vector{UInt64}(undef, N_BULK)
        v32 = Vector{UInt32}(undef, N_BULK)
        f = mdraws(@benchmark(rand!($rng, $vf)), N_BULK)
        u64 = mdraws(@benchmark(rand!($rng, $v64)), N_BULK)
        u32 = mdraws(@benchmark(rand!($rng, $v32)), N_BULK)
        @printf("%-18s %12.0f %12.0f %12.0f\n", name, f, u64, u32)
    end

    println("\nRaw bijection rate, millions of 4-word blocks per second:\n")
    z32 = ntuple(_ -> UInt32(0), 4)
    z64 = ntuple(_ -> UInt64(0), 4)
    bijections = [
        ("philox4x32-10",   (c, k) -> RDS.philox(c, k),   z32, (UInt32(1), UInt32(2))),
        ("philox4x64-10",   (c, k) -> RDS.philox(c, k),   z64, (UInt64(1), UInt64(2))),
        ("threefry4x32-20", (c, k) -> RDS.threefry(c, k), z32, z32),
        ("threefry4x64-20", (c, k) -> RDS.threefry(c, k), z64, z64),
    ]
    for (name, f, ctr, key) in bijections
        n = 10_000
        b = @benchmark block_loop($f, $ctr, $key, $n)
        @printf("%-18s %12.1f\n", name, mdraws(b, n))
    end

    println("\nFor reference, the idiom this method deliberately avoids:")
    let rng = Xoshiro256p(seed_words(4))
        b = @benchmark rand($rng)
        @printf("  @benchmark rand(rng) on Xoshiro256p: %.2f ns/draw, i.e. %.0f M/s\n",
                minimum(b).time, 1000 / minimum(b).time)
        @printf("  the same generator in the loop above:  %.0f M/s\n",
                mdraws(@benchmark(xor_loop_f64($rng, $N_SCALAR)), N_SCALAR))
        println("  A single timed call cannot resolve a sub-nanosecond operation;")
        println("  BenchmarkTools' own per-sample overhead is the same order.")
    end

    println("\nAllocations (must be zero):")
    for (name, mk) in generators
        rng = mk()
        v = Vector{Float64}(undef, 64)
        xor_loop_f64(rng, 10); rand!(rng, v)                # compile first
        @printf("  %-18s scalar %d B / 1000 draws, rand! %d B\n", name,
                @allocated(xor_loop_f64(rng, 1000)), @allocated(rand!(rng, v)))
    end
end

main()
