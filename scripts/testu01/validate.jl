#!/usr/bin/env julia

# On-demand TestU01 validation.
#
# This is NOT part of `Pkg.test()`, and must not become part of it: the package
# test suite verifies that an installation works, and runs SmallCrush only.
# Crush takes hours and BigCrush the better part of a day per generator, which
# is a validation exercise -- for a release, or for a paper -- not an
# installation check.
#
#     julia scripts/testu01/validate.jl --list
#     julia scripts/testu01/validate.jl --battery=crush --generator=Xoshiro256p
#     julia scripts/testu01/validate.jl --battery=alphabit --suite=bits
#
# Options
#   --battery=<name>                      default smallcrush; see BATTERIES
#   --suite=single|interleaved|bits|all   default single
#   --generator=<name>|all                default all
#   --streams=<n>                         streams for the interleaved suite (64)
#   --values=<n>                          values per stream, interleaved (1)
#   --bits=<n>                            bits for alphabit/rabbit (2^30)
#   --out=<dir>                           default testu01-results
#   --list                                print the plan and exit
#   --quiet                               summary only, no per-test reports
#
# Each run writes three things: the battery's own TestU01 report, to its own log
# file; one line per statistic in `pvalues.tsv`, taken from `bbattery_pVal[]`
# rather than from the printed report, so the p-values are the doubles the
# library computed and can be classified without parsing anything; and one line
# in `summary.tsv` recording the run, its cost and how many statistics it
# singled out.

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, Random, Printf, Dates

# The TestU01 layer is shared with the test suite, which is why it lives under
# `test/`: `Pkg.test()` has to be self-contained, and a script may reach
# anywhere in the checkout it belongs to.
include(joinpath(@__DIR__, "..", "..", "test", "tu01.jl"))

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
const PROV = provenance()

# Generators ------------------------------------------------------------------

# Seeds are expanded through splitmix64 from one fixed integer: a low-entropy
# state such as all-ones takes a while to mix and is a poor starting point for
# a battery, while a fixed integer keeps the runs reproducible.
seed_words(n::Int) = RandomDataStreams._splitmix_words(UInt64(12345), n)

# Every generator the package ships, each with the stream generator that
# produces it, so that all three suites cover the same set.
const GENERATOR_STREAMS = Pair{String,Any}[
    "MRG32k3a"        => MRG32k3aGen,
    "MRG63k3a"        => MRG63k3aGen,
    "Xoroshiro128p"   => () -> Xoroshiro128pGen(seed_words(2)),
    "Xoroshiro128ss"  => () -> Xoroshiro128ssGen(seed_words(2)),
    "Xoroshiro128pp"  => () -> Xoroshiro128ppGen(seed_words(2)),
    "Xoshiro256p"     => () -> Xoshiro256plusGen(seed_words(4)),
    "Xoshiro256ss"    => () -> Xoshiro256ssGen(seed_words(4)),
    "Xoshiro256pp"    => () -> Xoshiro256ppGen(seed_words(4)),
    "Xoshiro512p"     => () -> Xoshiro512pGen(seed_words(8)),
    "Xoshiro512ss"    => () -> Xoshiro512ssGen(seed_words(8)),
    "Xoshiro512pp"    => () -> Xoshiro512ppGen(seed_words(8)),
    "Philox4x32-10"   => PhiloxGen,
    "Philox4x64-10"   => Philox4x64Gen,
    "Threefry4x32-20" => Threefry4x32Gen,
    "Threefry4x64-20" => Threefry4x64Gen,
    "PCG64"           => () -> PCG64Gen(seed_words(2)),
    "PCG64DXSM"       => () -> PCG64DXSMGen(seed_words(2)),
]

const GENERATORS = GENERATOR_STREAMS

# Suites ----------------------------------------------------------------------
#
# single       the U(0,1) output, what the batteries were designed for
# interleaved  several streams round-robin, testing the streams against each
#              other rather than each on its own (L'Ecuyer et al. 2021, Sec. 6)
# bits         the UInt32 integer path, which the Crush batteries never see:
#              they examine the 30 most significant bits of the U(0,1) outputs

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

# TestU01 calls back into Julia through a C function pointer, so the callback
# must be a Function whose return type infers to Float64 -- UInt32 for the bit
# path. The generator tables have an abstract element type, so everything below
# goes through a function barrier: called with a value of unknown type, these
# specialise on its runtime type and close over something concrete. Without the
# barrier the callback returns Any, which `TU01.Gen` refuses rather than letting
# it reach the C side and crash there.
_single_stream(rng) = () -> rand(rng, Float64)
_bit_stream(rng) = () -> rand(rng, UInt32)
_interleaved(streams::Vector, per::Int) =
    let g = RoundRobin(streams, 1, 0, per)
        () -> g()
    end

stream_gen(name::String) =
    GENERATOR_STREAMS[findfirst(p -> first(p) == name, GENERATOR_STREAMS)].second()
make_rng(name::String) = next_stream!(stream_gen(name))

"""
    build(suite, name, opts) -> TU01.Gen

The live TestU01 generator for one run. Free it with `TU01.free!`, or let
`TU01.withgen` do it: the library keeps exactly one at a time.
"""
function build(suite::String, name::String, opts)
    if suite == "single"
        return TU01.Gen(_single_stream(make_rng(name)), name)
    elseif suite == "interleaved"
        g = stream_gen(name)
        return TU01.Gen(_interleaved([next_stream!(g) for _ in 1:opts.streams], opts.values), name)
    elseif suite == "bits"
        return TU01.bitgen(_bit_stream(make_rng(name)), name)
    end
    error("unknown suite: $suite")
end

"""
    closure_unsupported(err) -> Bool

Whether `err` is the one platform failure that stops everything: the callback is
built with `@cfunction` over a closure, which needs an executable trampoline,
and Apple Silicon forbids memory that is both writable and executable. It is
caught at the real call rather than predicted by a probe -- a closure over a
constant compiles to a singleton, needs no trampoline, and succeeds on exactly
the platform where the real one fails.
"""
closure_unsupported(err) = occursin("closures are not supported", sprint(showerror, err))

# Each entry runs one battery over a live generator and returns its statistics.
# The bit-level batteries take the number of bits to consume, which is why they
# are given the options rather than the generator alone.
#
# Crush and BigCrush examine the 30 most significant bits of a U(0,1) output and
# were never meant for an integer stream; Rabbit and Alphabit are the batteries
# that were. They are worth running on any of the three suites -- unif01 extracts
# bits from a double as readily as from a word -- but the `bits` suite is where
# they answer a question nothing else here does.
const BATTERIES = Dict(
    "smallcrush" => ((g, o) -> TU01.smallcrush(g),     "minutes"),
    "crush"      => ((g, o) -> TU01.crush(g),          "a few hours"),
    "bigcrush"   => ((g, o) -> TU01.bigcrush(g),       "most of a day"),
    "alphabit"   => ((g, o) -> TU01.alphabit(g, o.bits), "under a minute at 2^30 bits"),
    "rabbit"     => ((g, o) -> TU01.rabbit(g, o.bits),   "about ten minutes at 2^30 bits"),
)

"Batteries that consume a fixed number of bits rather than running to a fixed size."
const BIT_BATTERIES = ("alphabit", "rabbit")

# Per-test reports are on for a validation run -- every test with its parameters
# and its p-value -- both because that is the canonical TestU01 artefact to
# archive and because it is what makes a battery running for hours observable:
# the log fills as the tests complete instead of staying empty until the end.
# `--quiet` leaves only the closing summary.

# Driver ----------------------------------------------------------------------

function parse_args(args)
    opts = (battery = "smallcrush", suite = "single", generator = "all",
            streams = 64, values = 1, bits = 2.0^30, out = "testu01-results",
            list = false, quiet = false)
    for a in args
        if a == "--list"
            opts = merge(opts, (list = true,))
        elseif a == "--quiet"
            opts = merge(opts, (quiet = true,))
        elseif startswith(a, "--")
            k, _, v = partition(a)
            k == "battery"   ? (opts = merge(opts, (battery = lowercase(v),))) :
            k == "suite"     ? (opts = merge(opts, (suite = lowercase(v),))) :
            k == "generator" ? (opts = merge(opts, (generator = v,))) :
            k == "streams"   ? (opts = merge(opts, (streams = parse(Int, v),))) :
            k == "values"    ? (opts = merge(opts, (values = parse(Int, v),))) :
            k == "bits"      ? (opts = merge(opts, (bits = parse(Float64, v),))) :
            k == "out"       ? (opts = merge(opts, (out = v,))) :
            error("unknown option: $a")
        else
            error("unexpected argument: $a")
        end
    end
    haskey(BATTERIES, opts.battery) ||
        error("unknown battery: $(opts.battery) (have $(join(sort(collect(keys(BATTERIES))), ", ")))")
    opts.bits > 0 || error("--bits must be positive")
    return opts
end

function partition(a)
    body = a[3:end]
    i = findfirst('=', body)
    i === nothing && error("option needs a value: $a")
    return body[1:i-1], '=', body[i+1:end]
end

function plan(opts)
    names = opts.generator == "all" ? first.(GENERATORS) : [opts.generator]
    for n in names
        any(p -> first(p) == n, GENERATORS) || error("unknown generator: $n")
    end
    suites = opts.suite == "all" ? ["single", "interleaved", "bits"] : [opts.suite]
    return [(n, s) for s in suites for n in names]
end

function main(args)
    opts = parse_args(args)
    runs = plan(opts)
    battery, duration = BATTERIES[opts.battery]

    println("TestU01 validation -- ", opts.battery)
    println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
    println(provenance_line(PROV))
    println(length(runs), " run(s), each taking ", duration, " for this battery")
    opts.battery in BIT_BATTERIES && @printf("%.0f bits per run (--bits)\n", opts.bits)
    println()
    for (n, s) in runs
        @printf("  %-16s %s\n", n, s)
    end
    if opts.list
        println("\n--list given, nothing run.")
        return
    end

    mkpath(opts.out)
    index = joinpath(opts.out, "summary.tsv")
    isfile(index) || open(io -> println(io,
        "timestamp\tbattery\tsuite\tgenerator\tseconds\tlog\tjulia\tcpu\t" *
        "commit\tdirty\tpkgversion\tstreams\tvalues\tbits\tstatistics\tsuspect\tfailed\tminp"),
        index, "w")
    # One line per statistic, for every run in this directory: the exact
    # p-values, which the printed report only ever shows in its own formatting
    # ("eps", "1 -  3.0e-9"). Classifying against 1e-10 is not something to do
    # on text.
    pvals = joinpath(opts.out, "pvalues.tsv")
    isfile(pvals) || open(io -> println(io,
        "timestamp\tbattery\tsuite\tgenerator\tindex\tname\tp\tclass\tlog"), pvals, "w")

    println()
    for (name, suite) in runs
        stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
        log = joinpath(opts.out, "$(opts.battery)-$(suite)-$(replace(name, r"[^\w]" => "_"))-$stamp.log")
        @printf("--> %-16s %-12s -> %s\n", name, suite, log)
        flush(stdout)

        gen = try
            build(suite, name, opts)
        catch err
            if closure_unsupported(err)
                println("\nThis platform cannot build a C callback from a closure,")
                println("which TestU01 needs to call back into Julia. Nothing can run here.")
                exit(1)
            end
            println("    SKIPPED: ", sprint(showerror, err))
            continue
        end

        stats = TU01.Statistic[]
        elapsed = TU01.withgen(gen) do g
            open(log, "w") do io
                println(io, "# ", opts.battery, " / ", suite, " / ", name)
                println(io, "# Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
                println(io, "# ", provenance_line(PROV))
                println(io, "# started ", now())
                suite == "interleaved" &&
                    println(io, "# ", opts.streams, " streams, ", opts.values, " value(s) each, round-robin")
                # A replay of a bit-level battery has to consume the same number
                # of bits, so the run records it where replay.jl can read it.
                opts.battery in BIT_BATTERIES &&
                    @printf(io, "# bits = %.0f\n", opts.bits)
                flush(io)
                # TestU01 writes its report from C. Redirecting the descriptor is
                # not enough on its own: C stdout is block-buffered when it is not
                # a terminal, so without the fflush the report is still in the C
                # buffer when the descriptor is restored and lands on the console
                # instead of in the log.
                @elapsed redirect_stdout(io) do
                    try
                        TU01.line_buffer_stdout()
                        TU01.verbose!(!opts.quiet)
                        stats = battery(g, opts)
                    finally
                        TU01.flush_c_stdout()
                    end
                end
            end
        end

        bad = TU01.suspects(stats)
        failed = TU01.failures(stats)
        # How far the most extreme statistic sits from the middle: min(p, 1-p)
        # over the run, so one number says whether anything happened at all.
        extreme = isempty(stats) ? NaN : minimum(s -> min(s.p, 1 - s.p), stats)

        open(pvals, "a") do io
            for st in stats
                println(io, join((now(), opts.battery, suite, name, st.index, st.name,
                                  st.p, TU01.classify(st.p), basename(log)), '\t'))
            end
        end
        open(index, "a") do io
            println(io, join((now(), opts.battery, suite, name, round(elapsed, digits = 1),
                              log, VERSION, Sys.CPU_NAME, PROV.commit, PROV.dirty,
                              PROV.version,
                              suite == "interleaved" ? opts.streams : "",
                              suite == "interleaved" ? opts.values : "",
                              opts.battery in BIT_BATTERIES ? @sprintf("%.0f", opts.bits) : "",
                              length(stats), length(bad), length(failed), extreme), '\t'))
        end

        @printf("    done in %.1f s -- %d statistics, ", elapsed, length(stats))
        if !isempty(failed)
            @printf("%d FAILED outright, %d singled out\n", length(failed), length(bad))
        elseif !isempty(bad)
            @printf("%d singled out (replay them: they are what replay.jl is for)\n", length(bad))
        else
            println("none singled out")
        end
    end

    println("\nSummary index: ", index)
    println("Every p-value:  ", pvals)
    println()
    println("A statistic outside [1e-3, 1-1e-3] is *suspect*, which is a printing")
    println("threshold and not a verdict: replay it with replay.jl before calling it")
    println("anything. One outside [1e-10, 1-1e-10] is a failure outright. The test")
    println("numbers replay.jl needs are in the log's closing summary; the p-values")
    println("themselves are in pvalues.tsv, exactly as the library computed them.")
end

# Run the driver only when this file is the program. campaign.jl invokes it as
# a script, exactly as before; replay.jl includes it for the generator table,
# the suites and the TestU01 plumbing, and must not trigger a battery by doing
# so.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
