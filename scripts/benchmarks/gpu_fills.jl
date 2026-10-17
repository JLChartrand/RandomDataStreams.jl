#!/usr/bin/env julia

# Device fill rates of every counter-based generator, for every fill the CUDA
# extension provides, against the memory bound and CUDA.jl's own generators.
#
#     julia -O3 scripts/benchmarks/gpu_fills.jl [results.csv]
#
# Needs a functional CUDA device; prints a message and exits otherwise.
#
# Measurement method:
#
#   * One fill of 2^26 elements per call into a preallocated CuArray, timed
#     with `CUDA.@elapsed` over enough calls to last 0.2 s, after a warm-up
#     call. Rates are elements per second; the GB/s column is the bytes
#     written, against which `fill!` gives the memory bound.
#
#   * A fresh generator per call, so a fill never starts mid-block unless the
#     row says so -- the kernel's own cost, not the bookkeeping of a stream.
#
#   * The references are `fill!` (no generation at all), CUDA.jl's default
#     `rand!`/`randn!` with no generator argument (the cuRAND library), and
#     GPUArrays' generator, `CUDA.default_rng()`.
#
#   * Small fills are launch-bound: their time per call is mostly the host's,
#     so it moves with the CPU the process runs on. On a hybrid CPU pin the
#     process (`taskset -c 0 julia -O3 ...`) before quoting them, as for
#     throughput.jl.
#
# A card with a low power limit varies by 10 to 20% from one run to the next;
# compare rows of one run, and repeat a run before quoting a figure.

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, Random, CUDA, Printf

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
include(joinpath(@__DIR__, "csv_util.jl"))

const N = 2^26

function seconds_per_call(f, A; mintime = 0.2)
    f(A); CUDA.synchronize()
    once = CUDA.@elapsed f(A)
    reps = max(3, ceil(Int, mintime / once))
    return (CUDA.@elapsed for _ in 1:reps; f(A); end) / reps
end

function row!(rows, commit, group, label, T, s)
    @printf("%-14s %-30s %-8s %8.2f ms   %6.2f Gvalues/s   %6.1f GB/s\n",
            group, label, T, 1e3s, N / s / 1e9, N * sizeof(T) / s / 1e9)
    push!(rows, (commit, group, label, string(T), string(s), string(N / s / 1e9)))
end

function main()
    csv_path = isempty(ARGS) ? nothing : ARGS[1]
    rows = NTuple{6,String}[]
    println("RandomDataStreams GPU fills")
    println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
    println(provenance_line())
    if !CUDA.functional()
        println("\nCUDA is not functional on this machine -- nothing to benchmark.")
        return
    end
    println("GPU: ", CUDA.name(CUDA.device()), ", CUDA.jl ", pkgversion(CUDA))
    println("2^26 elements per fill\n")
    commit = provenance().commit

    for T in (Float64, Float32)
        A = CuArray{T}(undef, N)
        row!(rows, commit, "reference", "fill! (memory bound)", T, seconds_per_call(a -> fill!(a, zero(T)), A))
        row!(rows, commit, "reference", "CUDA.jl rand!, no rng (cuRAND)", T, seconds_per_call(a -> rand!(a), A))
        row!(rows, commit, "reference", "CUDA.jl randn!, no rng (cuRAND)", T, seconds_per_call(a -> randn!(a), A))
        row!(rows, commit, "reference", "GPUArrays rand!", T, seconds_per_call(a -> rand!(CUDA.default_rng(), a), A))
        row!(rows, commit, "reference", "GPUArrays randn!", T, seconds_per_call(a -> randn!(CUDA.default_rng(), a), A))
    end
    println()

    for (name, G) in (("Philox4x32", PhiloxRNG), ("Philox4x64", Philox4x64RNG),
                      ("Threefry4x32", Threefry4x32RNG), ("Threefry4x64", Threefry4x64RNG))
        for T in (Float64, Float32, Float16)
            A = CuArray{T}(undef, N)
            row!(rows, commit, name, "rand!", T, seconds_per_call(a -> rand!(G(1), a), A))
            row!(rows, commit, name, "randn! (inversion)", T, seconds_per_call(a -> randn!(G(1), a), A))
            row!(rows, commit, name, "randexp! (inversion)", T, seconds_per_call(a -> randexp!(G(1), a), A))
            T === Float32 &&
                row!(rows, commit, name, "randn32! (Float32 inversion)", T, seconds_per_call(a -> randn32!(G(1), a), A))
            G === PhiloxRNG && T !== Float16 &&
                row!(rows, commit, name, "randn_boxmuller!", T, seconds_per_call(a -> randn_boxmuller!(G(1), a), A))
        end
        println()
    end

    println("launch-bound sizes, Philox4x32, Float32: microseconds per call")
    for m in (10^3, 10^5, 10^7)
        A = CuArray{Float32}(undef, m)
        for (label, f) in (("rand!", rand!), ("randn!", randn!), ("randn32!", randn32!))
            s = seconds_per_call(a -> f(PhiloxRNG(1), a), A; mintime = 0.1)
            @printf("  n = %-9d %-10s %8.1f µs\n", m, label, 1e6s)
            push!(rows, (commit, "small n=$m", label, "Float32", string(s), string(m / s / 1e9)))
        end
    end

    if csv_path !== nothing
        write_csv(csv_path, ["commit", "group", "fill", "eltype", "seconds", "gvalues_per_s"], rows)
        println("\nWrote ", csv_path)
    end
end

main()
