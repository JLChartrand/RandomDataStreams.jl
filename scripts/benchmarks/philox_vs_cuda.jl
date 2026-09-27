#!/usr/bin/env julia

# One (contender, element type) case of the "big sum of exp(N(0,1))" GPU
# comparison: this package's Philox4x32-10 against CUDA.jl's own normal
# generators. Meant to be driven by philox_vs_cuda.sh, one process per case:
#
#     julia -O3 scripts/benchmarks/philox_vs_cuda.jl \
#         <contender> <Float32|Float64> <results.csv> <errors.txt>
#
# Contenders
#
#   philox_boxmuller  this package:  randn!(::PhiloxRNG, ::CuArray)
#   philox_inversion  this package:  randn_inversion!(::PhiloxRNG, ::CuArray)
#   curand            CUDA.jl:       Random.randn!(::CuArray) -- with no RNG
#                                    argument this is the cuRAND library
#                                    (CUDA.jl 6.x), i.e. what `CUDA.randn(T, n)`
#                                    gives you
#   gpuarrays         CUDA.jl:       Random.randn!(CUDA.default_rng(), ::CuArray)
#                                    -- GPUArrays' own Philox4x32-10 +
#                                    Box-Muller kernel, the closest rival to
#                                    this package's design
#   native            CUDA.jl 6.x:   Random.randn!(CUDA.cuRAND.native_rng(), _)
#                                    -- device-side Philox2x32 kernel. Its own
#                                    docstring says it exists for testing and
#                                    perf comparison, not production; included
#                                    because it is the one CUDA.jl RNG that
#                                    documents being a comparison target.
#
# Workload (per repetition): draw TOTAL variates in CHUNK-sized pieces into one
# reused device buffer and accumulate sum(exp, chunk) into a Float64 on the
# host. Chunked because the buffer for 2^30 Float64 draws is 8 GiB; the same
# chunking is applied to every contender, so it does not favour any of them.
# Two timings per case, so a slow generator can be told from a slow reduction:
#
#   fill    generate only (no reduction)
#   sumexp  generate + sum(exp, chunk), the workload asked for
#
# Both are the minimum and the median over REPS repetitions, after one untimed
# warm-up chunk (compilation).
#
# Sizes (env vars, all log2): BENCH_TOTAL_LOG2 (default 30), BENCH_CHUNK_LOG2
# (default 26, or the total if that is smaller), BENCH_REPS (5). The chunk is a power of two on purpose: cuRAND's normal
# generator needs a power-of-two length, and CUDA.jl 6.x silently pads any
# other length into a second, larger buffer and copies back -- which would
# charge that allocation and copy to cuRAND alone. If the chunk does not fit
# in free GPU memory it is halved until it does, and the chunk actually used is
# recorded in the CSV.
#
# Correctness cross-check: E[exp(Z)] = exp(1/2) for Z ~ N(0,1), and
# Var[exp(Z)] = e^2 - e, so the mean of all N draws has a known standard error.
# The CSV records the sample mean and its z-score. A wildly wrong generator
# (or one that silently returns zeros or NaN) shows up there; a merely
# fast-but-wrong one would otherwise look like a win. |z| > 5 is written to the
# errors file as a WARNING -- it is not an exception and does not fail the case.
#
# Errors. Anything that throws -- CUDA not functional, out of memory, a kernel
# fault, a method missing in this CUDA.jl version -- is appended to the errors
# file with the exception, its backtrace and the GPU state, and the process
# exits 3. It is one process per case because a CUDA fault can leave the
# context unusable ("sticky"), so a failure here must not be able to poison the
# next case.

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, Random, CUDA, Printf, Dates

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
include(joinpath(@__DIR__, "csv_util.jl"))

const SEED = 20260920
const CONTENDERS = ("philox_boxmuller", "philox_inversion", "curand", "gpuarrays", "native")
const ZSCORE_WARN = 5.0

const CSV_HEADER = ["contender", "eltype", "sumexp_gdraws_per_s", "fill_gdraws_per_s",
                    "sumexp_min_s", "sumexp_med_s", "fill_min_s", "fill_med_s",
                    "total_draws", "chunk", "reps", "mean", "z",
                    "commit", "dirty", "cuda_jl", "gpu"]

"Returns `gen!(buf)`: fill a device buffer with N(0,1) from the named contender."
function make_filler(name::AbstractString)
    if name == "philox_boxmuller"
        rng = PhiloxRNG((UInt32(1), UInt32(2)))
        return buf -> randn!(rng, buf)
    elseif name == "philox_inversion"
        rng = PhiloxRNG((UInt32(1), UInt32(2)))
        return buf -> randn_inversion!(rng, buf)
    elseif name == "curand"
        CUDA.seed!(SEED)
        return buf -> randn!(buf)
    elseif name == "gpuarrays"
        rng = CUDA.default_rng()
        Random.seed!(rng, SEED)
        return buf -> randn!(rng, buf)
    elseif name == "native"
        (isdefined(CUDA, :cuRAND) && isdefined(CUDA.cuRAND, :native_rng)) ||
            error("CUDA.cuRAND.native_rng does not exist in CUDA.jl $(pkgversion(CUDA)); " *
                  "it is a CUDA.jl 6.x API")
        rng = CUDA.cuRAND.native_rng()
        Random.seed!(rng, SEED)
        return buf -> randn!(rng, buf)
    else
        error("unknown contender \"$name\"; expected one of $(join(CONTENDERS, ", "))")
    end
end

"Draw `nchunks` chunks and return the Float64 sum of exp over all of them."
function stream_sumexp!(gen!, buf, nchunks::Int)
    acc = 0.0
    for _ in 1:nchunks
        gen!(buf)
        acc += Float64(sum(exp, buf))      # copies the scalar to the host: syncs
    end
    return acc
end

function stream_fill!(gen!, buf, nchunks::Int)
    for _ in 1:nchunks
        gen!(buf)
    end
    return nothing
end

"Wall-clock seconds of `f()`; `sync()` runs inside the timed region."
function timed(f, sync)
    t0 = time_ns()
    val = f()
    sync()
    return (time_ns() - t0) / 1e9, val
end

_median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

"""
    measure(gen!, buf, total, chunk, reps; sync) -> NamedTuple

Time the two workloads over `total` draws per repetition, in chunks of `chunk`.
`buf` is the (already allocated) device buffer, `gen!(buf)` fills it with
normals. Generic in the buffer so the loop and statistics can be checked on the
CPU without a GPU.
"""
function measure(gen!, buf, total::Int, chunk::Int, reps::Int; sync = () -> nothing)
    nchunks = total ÷ chunk
    gen!(buf); sum(exp, buf); sync()                 # warm up: compile, untimed

    fill_t = [first(timed(() -> stream_fill!(gen!, buf, nchunks), sync)) for _ in 1:reps]

    sum_t = Float64[]
    acc = 0.0
    for _ in 1:reps
        t, a = timed(() -> stream_sumexp!(gen!, buf, nchunks), sync)
        push!(sum_t, t)
        acc += a
    end

    n = Float64(nchunks * chunk) * reps               # draws behind `acc`
    mean_ = acc / n
    se = sqrt(exp(2.0) - exp(1.0)) / sqrt(n)
    return (fill_min = minimum(fill_t), fill_med = _median(fill_t),
            sumexp_min = minimum(sum_t), sumexp_med = _median(sum_t),
            mean = mean_, z = (mean_ - exp(0.5)) / se)
end

"Halve `chunk` until a buffer of it fits comfortably in `free` bytes."
function fit_chunk(chunk::Int, T::Type, free::Integer; floor_ = 2^16)
    while chunk > floor_ && chunk * sizeof(T) > free ÷ 4
        chunk ÷= 2
    end
    return chunk
end

function record_error(path::AbstractString, label::AbstractString, err, bt; warning = false)
    first_entry = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        println(io, repeat("=", 78))
        println(io, warning ? "WARNING  " : "ERROR    ", label, "   ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, provenance_line())
        println(io, "Julia ", VERSION, ", CUDA.jl ", pkgversion(CUDA))
        if warning
            println(io, err)
        else
            println(io, "exception type: ", typeof(err))
            showerror(io, err, bt)
            println(io)
        end
        try
            if CUDA.functional()
                println(io, "GPU: ", CUDA.name(CUDA.device()),
                        "; free ", Base.format_bytes(CUDA.free_memory()),
                        " of ", Base.format_bytes(CUDA.total_memory()))
            else
                println(io, "GPU: CUDA.functional() == false")
            end
        catch e
            println(io, "GPU state unavailable: ", sprint(showerror, e))
        end
        if first_entry && !warning
            # Once per file: the toolchain listing is long and identical each time.
            println(io, "\n--- CUDA.versioninfo() ---")
            try
                CUDA.versioninfo(io)
            catch e
                println(io, "CUDA.versioninfo() failed: ", sprint(showerror, e))
            end
        end
        println(io)
    end
end

function run_case(name::String, T::Type, csv_path::String, err_path::String)
    total_log2 = parse(Int, get(ENV, "BENCH_TOTAL_LOG2", "30"))
    # The chunk defaults to 2^26 but never past the total, so lowering only
    # BENCH_TOTAL_LOG2 (a smoke test) does not trip the validation below.
    chunk_log2 = parse(Int, get(ENV, "BENCH_CHUNK_LOG2", string(min(26, total_log2))))
    reps       = parse(Int, get(ENV, "BENCH_REPS", "5"))
    (16 <= chunk_log2 <= total_log2 && reps >= 1) ||
        error("need 16 <= BENCH_CHUNK_LOG2 <= BENCH_TOTAL_LOG2 and BENCH_REPS >= 1; got " *
              "chunk=$chunk_log2 total=$total_log2 reps=$reps")

    println("RandomDataStreams: philox vs CUDA.jl randn -- $name, $T")
    println(provenance_line())
    println("Julia ", VERSION, ", CUDA.jl ", pkgversion(CUDA))

    CUDA.functional(true)          # throws with the reason CUDA is unusable
    gpu = CUDA.name(CUDA.device())
    println("GPU: ", gpu)

    chunk = fit_chunk(2^chunk_log2, T, CUDA.free_memory())
    chunk == 2^chunk_log2 || println("chunk reduced to 2^", Int(log2(chunk)),
                                     " to fit free GPU memory")
    total = max(2^total_log2, chunk)
    println("total = $total draws per repetition, chunk = $chunk ($(Base.format_bytes(chunk * sizeof(T)))), reps = $reps\n")

    gen! = make_filler(name)
    buf = CUDA.zeros(T, chunk)
    r = measure(gen!, buf, total, chunk, reps; sync = CUDA.synchronize)

    gd(t) = total / t / 1e9
    @printf("%-18s %-8s  sum(exp): %8.3f Gdraws/s (min %.4f s, med %.4f s)\n",
            name, string(T), gd(r.sumexp_min), r.sumexp_min, r.sumexp_med)
    @printf("%-18s %-8s  fill    : %8.3f Gdraws/s (min %.4f s, med %.4f s)\n",
            "", "", gd(r.fill_min), r.fill_min, r.fill_med)
    @printf("mean of exp(n) = %.6f, expected %.6f, z = %+.2f\n", r.mean, exp(0.5), r.z)

    if !isfinite(r.z) || abs(r.z) > ZSCORE_WARN
        record_error(err_path, "$name $T", "mean of exp(n) = $(r.mean), expected $(exp(0.5)), " *
                     "z = $(r.z): this contender's output does not look like N(0,1). " *
                     "The throughput row is kept in the CSV, but do not trust it.", nothing;
                     warning = true)
    end

    p = provenance()
    append_csv(csv_path, CSV_HEADER,
               (name, string(T), gd(r.sumexp_min), gd(r.fill_min),
                r.sumexp_min, r.sumexp_med, r.fill_min, r.fill_med,
                total, chunk, reps, r.mean, r.z,
                p.commit, p.dirty, pkgversion(CUDA), replace(gpu, "," => ";")))
end

function main()
    if length(ARGS) != 4
        println(stderr, "usage: philox_vs_cuda.jl <contender> <Float32|Float64> <results.csv> <errors.txt>")
        exit(2)
    end
    name, tname, csv_path, err_path = ARGS
    T = get(Dict("Float32" => Float32, "Float64" => Float64), tname, nothing)
    if T === nothing
        println(stderr, "eltype must be Float32 or Float64, got \"$tname\"")
        exit(2)
    end
    try
        run_case(name, T, csv_path, err_path)
    catch err
        bt = catch_backtrace()
        try
            record_error(err_path, "$name $tname", err, bt)
        catch e2
            println(stderr, "could not write $err_path: ", sprint(showerror, e2))
        end
        println(stderr, "\nFAILED: $name $tname -- ", sprint(showerror, err),
                "\n(full details in $err_path)")
        exit(3)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
