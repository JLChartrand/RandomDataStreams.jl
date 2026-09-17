#!/usr/bin/env julia

# CPU vs GPU throughput for a common Monte Carlo shape: generate N iid
# variates, transform them, reduce to one scalar. Philox4x32-10 drives both
# sides, so what is measured is where the work runs (a host loop vs. a device
# kernel plus a device-side reduction), not a difference in generator.
#
#     julia -O3 scripts/benchmarks/gpu_throughput.jl
#
# Needs a functional CUDA device (`CUDA.functional()`); prints a message and
# exits otherwise, rather than failing loudly on a CPU-only machine.
#
# Measurement method, same rationale as throughput.jl:
#
#   * BenchmarkTools, minimum over samples.
#
#   * Each sample is "fill, then reduce": `rand!`/`randn!` into a buffer, then
#     `sum` (or `sum(exp, ...)`, fused rather than materializing an
#     intermediate array). `sum` on a `CuArray` copies its scalar result back
#     to the host, which already forces the device to finish -- so the timed
#     expression needs no explicit `CUDA.synchronize()` to be honest about GPU
#     wall-clock time.
#
#   * N = 100_000 is deliberately modest, on purpose: a GPU kernel launch and
#     the final host sync cost a few fixed microseconds no matter how little
#     work is inside them, and at this size that overhead is not yet
#     negligible next to the generator itself. Do not read a GPU loss at this
#     N as "the CUDA extension is slow" -- it says where the break-even point
#     is, which is exactly what this script is for. See the bottom of this
#     file for other sizes worth trying.
#
# CPU and GPU draw from independent `PhiloxRNG` instances seeded with the same
# key. For the plain-uniform case their two buffers end up bit-identical after
# a call (rand! into a CuArray{Float64} is built to match the CPU fill
# exactly); for `exp` the two can differ in the last bit or two, since the CPU
# and CUDA math libraries don't promise identical rounding; for `randn!` they
# are expected to differ outright -- Box-Muller on the GPU, Ziggurat on the
# CPU, deliberately different algorithms (see ext/RandomDataStreamsCUDAExt.jl).
#
# Three GPU-only normal-variate rows (branch philox-gpu-randn-test) put
# `randn!` (Box-Muller), `randn_inversion!` (Phi^-1 via CUDA's `normcdfinv`),
# and `randn_polar!` (Marsaglia's polar method, an accept-reject algorithm)
# side by side against the *same* CPU Ziggurat baseline, so their relative
# GPU cost is comparable directly from this one table. Do not read the "CPU"
# column for those three as three different measurements -- it is the same
# `randn!` timing repeated, included only so each row's speedup ratio is
# self-contained. `randn_polar!`'s reservation of `_POLAR_BUDGET` Philox
# blocks per output pair (see its docstring) also means it advances the RNG
# much further per call than the other two for the same `N` -- irrelevant to
# a throughput number taken from a fresh generator each run, but relevant if
# reusing the same stream afterwards.
#
# Pass a path as the first argument to also write the table as a CSV
# (columns: commit,N,case,cpu_ms,gpu_ms,speedup). Plain
# `julia gpu_throughput.jl`, with no argument, prints only.

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, Random, CUDA, BenchmarkTools, Printf

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
include(joinpath(@__DIR__, "csv_util.jl"))

function run_case(N::Int, rows)
    println("N = $N draws, fill + reduce (BenchmarkTools minimum)\n")
    @printf("%-24s %12s %12s %10s\n", "case", "CPU (ms)", "GPU (ms)", "speedup")

    key = (UInt32(1), UInt32(2))
    cpu, gpu = PhiloxRNG(key), PhiloxRNG(key)
    cbuf = Vector{Float64}(undef, N)
    gbuf = CUDA.zeros(Float64, N)

    cases = [
        ("sum(u), u ~ U(0,1)",
         () -> (rand!(cpu, cbuf); sum(cbuf)),
         () -> (rand!(gpu, gbuf); sum(gbuf))),
        ("sum(exp(u)), u ~ U(0,1)",
         () -> (rand!(cpu, cbuf); sum(exp, cbuf)),
         () -> (rand!(gpu, gbuf); sum(exp, gbuf))),
        ("sum(n), n ~ N(0,1) [Box-Muller]",
         () -> (randn!(cpu, cbuf); sum(cbuf)),
         () -> (randn!(gpu, gbuf); sum(gbuf))),
        ("sum(n), n ~ N(0,1) [inversion]",
         () -> (randn!(cpu, cbuf); sum(cbuf)),
         () -> (randn_inversion!(gpu, gbuf); sum(gbuf))),
        ("sum(n), n ~ N(0,1) [polar/accept-reject]",
         () -> (randn!(cpu, cbuf); sum(cbuf)),
         () -> (randn_polar!(gpu, gbuf); sum(gbuf))),
    ]

    commit = provenance().commit
    for (name, cpu_f, gpu_f) in cases
        cpu_f(); gpu_f()                              # compile / warm up, untimed
        tc = minimum(@benchmark $cpu_f()).time / 1e6  # ns -> ms
        tg = minimum(@benchmark $gpu_f()).time / 1e6
        @printf("%-24s %12.3f %12.3f %10.2fx\n", name, tc, tg, tc / tg)
        push!(rows, (commit, string(N), name, string(tc), string(tg), string(tc / tg)))
    end
    println()
end

function main()
    csv_path = isempty(ARGS) ? nothing : ARGS[1]
    rows = Tuple{String,String,String,String,String,String}[]

    println("RandomDataStreams GPU throughput")
    println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
    println(provenance_line())

    if !CUDA.functional()
        println("\nCUDA is not functional on this machine -- nothing to benchmark.")
        return
    end

    println("GPU: ", CUDA.name(CUDA.device()))
    println(repeat("-", 78), "\n")

    run_case(100_000, rows)

    if csv_path !== nothing
        write_csv(csv_path, ["commit", "N", "case", "cpu_ms", "gpu_ms", "speedup"], rows)
        println("Wrote ", csv_path)
    end
end

main()
