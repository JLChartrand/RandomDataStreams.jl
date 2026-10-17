#!/usr/bin/env julia
#
# Calibrate a battery's CollisionOver tests against generators that are not in
# question: replicate tests 3 to 12 of BigCrush -- smarsa_CollisionOver, ten
# parameter sets -- many times, and record for every replication the number of
# collisions TestU01 counted and the p-value it gave.
#
#     julia scripts/testu01/calibrate.jl --generator=aes --job=3 \
#           --tests=3:2,4:2,9:6,10:6,11:6,12:6 --out=<dir>
#     julia scripts/testu01/calibrate.jl --generator=Philox4x32-10 --job=0 \
#           --tests=11:14,12:14 --out=<dir>
#     julia scripts/testu01/calibrate.jl --generator=ulec-mrg32k3a --tests=11:1 --out=<dir>
#     julia scripts/testu01/calibrate.jl --generator=MRG32k3a --stream=1 --tests=11:1 --out=<dir>
#
# Why. Across the BigCrush campaigns of October 2026, CollisionOver with
# t = 21 gave far more small p-values than chance allows, on generators of
# every family, while the same test with t = 2 to 14 did not. All ten of
# BigCrush's CollisionOver tests count collisions among n = 2e7 overlapping
# t-tuples in d^t = 2^42 cells, N = 30 times, and compare the total with a
# Poisson law of mean 30 n^2 / (2 d^t). They differ only in the alphabet d.
# With overlapping tuples, a collision between the tuples at i and j is
# followed by one at i+1 and j+1 whenever the next symbols agree, which they
# do with probability 1/d: collisions come in runs, and the variance of their
# number is (1 + 1/d)/(1 - 1/d) times the Poisson one. Negligible for d = 2^21,
# 9/7 for d = 8 (t = 14), 5/3 for d = 4 (t = 21).
#
# The generators. `aes` is TestU01's own AES in counter mode, called by the
# battery without any Julia in between; job j uses the 128-bit key whose
# little-endian bytes are those of j + 1, and counter 0. A package generator
# goes through the callback every campaign used, on stream 4097 + j unless
# --stream says otherwise -- past every stream the campaigns and their replays
# drew from. `ulec-mrg32k3a` is TestU01's MRG32k3a from the published state,
# whose numbers are exactly those of the package's `MRG32k3a()` (stream 1):
# the two must count the same collisions, which checks the callback path on
# this very test.
#
# Output, in <out>/<generator>-job<j>/: the TestU01 log, and collisions.tsv,
# one line per replication. Nothing is classified here; the analysis that was
# fixed before the run reads those files.

include("validate.jl")       # the generator table, TU01, provenance (PROV)

# BigCrush's CollisionOver tests, by test number: (r, d, t), with N = 30 and
# n = 2e7 for all of them, as BigCrush's own log prints them.
const COLLISION_TESTS = Dict(3 => (0, 2097152, 2), 4 => (9, 2097152, 2),
                             5 => (0, 16384, 3), 6 => (16, 16384, 3),
                             7 => (0, 64, 7), 8 => (24, 64, 7),
                             9 => (0, 8, 14), 10 => (27, 8, 14),
                             11 => (0, 4, 21), 12 => (28, 4, 21))
const FIRST_STREAM = 4097

function parse_calibration_args(args)
    opts = (generator = "", job = 0, stream = 0, tests = Pair{Int,Int}[], out = "",
            battery = "bigcrush")
    for a in args
        startswith(a, "--") || error("unexpected argument: $a")
        k, _, v = partition(a)
        if k == "generator"
            opts = merge(opts, (generator = v,))
        elseif k == "job"
            opts = merge(opts, (job = parse(Int, v),))
        elseif k == "stream"
            opts = merge(opts, (stream = parse(Int, v),))
        elseif k == "out"
            opts = merge(opts, (out = v,))
        elseif k == "tests"
            spec = Pair{Int,Int}[]
            for item in split(v, ',')
                t, r = parse.(Int, split(item, ':'))
                haskey(COLLISION_TESTS, t) || error("test $t is not one of BigCrush's CollisionOver tests (3 to 12)")
                r >= 1 || error("test $t: at least one replication")
                push!(spec, t => r)
            end
            opts = merge(opts, (tests = spec,))
        else
            error("unknown option: $a")
        end
    end
    isempty(opts.generator) && error("--generator is required: aes, ulec-mrg32k3a or a package generator")
    isempty(opts.tests) && error("--tests is required, as test:replications[,...]")
    isempty(opts.out) && error("--out is required")
    opts.job >= 0 || error("--job must be non-negative")
    return opts
end

aes_key(job::Int) = collect(reinterpret(UInt8, [UInt128(job + 1)]))   # little-endian

"The generator of one job, and a line saying exactly which one it is."
function calibration_gen(opts)
    g = opts.generator
    if g == "aes"
        key = aes_key(opts.job)
        return TU01.aes(key), "TestU01 AES-128, counter mode from 0, key " * bytes2hex(key) * " (bytes of job + 1, little-endian)"
    elseif g == "ulec-mrg32k3a"
        return TU01.ulec_mrg32k3a(), "TestU01 ulec_CreateMRG32k3a, state 12345 x 6"
    end
    any(p -> first(p) == g, GENERATOR_STREAMS) || error("unknown generator: $g")
    s = opts.stream > 0 ? opts.stream : FIRST_STREAM + opts.job
    sg = stream_gen(g)
    for _ in 1:(s - 1)
        next_stream!(sg)
    end
    return TU01.Gen(_single_stream(next_stream!(sg)), g), "package $g, stream $s, through the Julia callback"
end

const PARAMS = r"N = 30,\s+n = 20000000,\s+r =\s*(\d+),\s+d =\s*(\d+),\s+t =\s*(\d+)"
const OBSERVED = r"Observed number of collisions\s*:\s*(\d+)"

"(r, d, t, collisions) of every CollisionOver replication in a log, in order."
function collision_counts(log)
    out = NTuple{4,Int}[]
    cur = nothing
    for line in eachline(log)
        m = match(PARAMS, line)
        if m !== nothing
            cur = parse.(Int, (m[1], m[2], m[3]))
            continue
        end
        m = match(OBSERVED, line)
        if m !== nothing && cur !== nothing
            push!(out, (cur..., parse(Int, m[1])))
            cur = nothing
        end
    end
    return out
end

function calibrate(args)
    opts = parse_calibration_args(args)
    gen, what = calibration_gen(opts)
    tag = string(replace(opts.generator, r"[^\w-]" => "_"), "-job", opts.job)
    dir = joinpath(opts.out, tag)
    mkpath(dir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    log = joinpath(dir, "$(opts.battery)-collisionover-$tag-$stamp.log")
    rep = zeros(Cint, TU01.NTESTS[Symbol(opts.battery)] + 1)
    for (t, r) in opts.tests
        rep[t + 1] = r                           # rep[i] in C is rep[i+1] here
    end
    println("CollisionOver calibration -- ", what)
    println("tests ", join(("$t x $r" for (t, r) in opts.tests), ", "), "  ->  ", log)
    flush(stdout)

    stats = TU01.Statistic[]
    elapsed = TU01.withgen(gen) do g
        open(log, "w") do io
            println(io, "# ", opts.battery, " / collisionover / ", opts.generator)
            println(io, "# Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
            println(io, "# ", provenance_line(PROV))
            println(io, "# started ", now())
            println(io, "# generator: ", what)
            println(io, "# tests (number x replications): ", join(("$t x $r" for (t, r) in opts.tests), ", "))
            flush(io)
            @elapsed redirect_stdout(io) do
                try
                    TU01.line_buffer_stdout()
                    TU01.verbose!(true)
                    stats = TU01.repeat_battery(Symbol(opts.battery), g, rep)
                finally
                    TU01.flush_c_stdout()
                end
            end
        end
    end

    counts = collision_counts(log)
    length(counts) == length(stats) ||
        error("$(length(counts)) collision counts in the log for $(length(stats)) statistics")
    # RepeatBigCrush runs the tests in increasing order, each its replications
    # in a row, so the counts follow the same order as `opts.tests` sorted.
    order = [t for (t, r) in sort(opts.tests) for _ in 1:r]
    open(joinpath(dir, "collisions.tsv"), "w") do io
        println(io, "generator\tjob\ttest\tr\td\tt\treplication\tcollisions\tp\tname\tcommit\tdirty")
        seen = Dict{Int,Int}()
        for (i, (t, (r, d, tt, c))) in enumerate(zip(order, counts))
            (r, d, tt) == COLLISION_TESTS[t] ||
                error("replication $i: the log shows (r, d, t) = ($r, $d, $tt), not test $t's $(COLLISION_TESTS[t])")
            seen[t] = get(seen, t, 0) + 1
            println(io, join((opts.generator, opts.job, t, r, d, tt, seen[t], c, stats[i].p,
                              stats[i].name, PROV.commit, PROV.dirty), '\t'))
        end
    end
    @printf("done in %.0f s: %d replications\n", elapsed, length(stats))
end

if abspath(PROGRAM_FILE) == @__FILE__
    calibrate(ARGS)
end
