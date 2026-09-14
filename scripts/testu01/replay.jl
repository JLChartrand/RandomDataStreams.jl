#!/usr/bin/env julia
#
# Re-apply only the tests a battery singled out, on output the first run never
# saw. This is the step TestU01's authors ask for and the batteries themselves
# provide for; see `docs/src/validation.md` for the classification rule it
# serves.
#
#     julia scripts/testu01/replay.jl --log=<battery log> [--reps=3]
#     julia scripts/testu01/replay.jl --battery=crush --suite=bits \
#           --generator=Xoshiro256ss --tests=19 --reps=5
#
# A bit-level battery (alphabit, rabbit) also needs the number of bits the
# original run consumed: --log reads it from the log's header, otherwise pass
# --bits.
#
# Why this exists. A summary report lists every p-value outside
# [0.001, 0.999], which is `gofw_Suspectp` and nothing more than a printing
# threshold: over a whole campaign a handful of those are expected from chance
# alone. The TestU01 guide draws the real line elsewhere -- "when a p-value is
# extremely close to 0 or to 1 (for example, if it is less than 1e-10), one can
# obviously conclude that the generator fails the test. If the p-value is
# suspicious but failure is not clear enough, (p = 0.0005, for example), then
# the test can be replicated independently until either failure becomes obvious
# or suspicion disappears." Everything between those two cases has to be
# replayed before it can be called anything.
#
# Two details make the replay honest rather than decorative.
#
# The output must be disjoint. `seed_words` is a fixed seed, deliberately, so
# that a run can be reproduced exactly -- which means re-running a suite the
# ordinary way reproduces the same p-values and settles nothing. `--skip`
# advances the stream generator past everything the original run consumed, so
# the replay is the independent replication the guide asks for and not the same
# arithmetic performed twice.
#
# Only the flagged tests are re-run. `bbattery_RepeatCrush(gen, rep)` applies
# test i of the battery rep[i] times and skips the rest, so resolving three
# suspects costs minutes rather than the hours the whole battery would. The
# number printed beside each suspect in a summary report is that same test
# number -- not the index of the statistic -- so it can be handed straight to
# rep. That is worth stating because a battery reports more statistics than it
# has tests (144 against 96 for Crush): several statistics can carry the same
# number, and five lines reading "10 RandomWalk1 ..." are one test, not five.

include("validate.jl")

# Tests per battery, for sizing rep[]. TestU01's restriction is that rep must
# have "one more element than the number of tests in the battery": rep is
# indexed by the battery's own test numbers, which start at 1, so element 0 is
# never read. The layer keeps the table; here it is reached by the battery names
# the command line uses.
ntests(battery::AbstractString) = TU01.NTESTS[Symbol(battery)]
known_battery(battery::AbstractString) = haskey(TU01.NTESTS, Symbol(battery))

# Past everything any suite of the campaign consumes: `single` and `bits` take
# stream 1, `interleaved` takes 1..streams (64 by default). A round number well
# clear of both is easier to reason about later than a computed minimum, and
# advancing a stream is a jump-ahead, not a draw -- it costs nothing.
const DEFAULT_SKIP = 1024

"""
    suspects(log) -> Vector{Tuple{Int,String,String}}

The test number, name and printed p-value of every line TestU01 singled out in
a battery log. Reads the summary block at the end -- the same block a human
reads -- rather than re-deriving anything.
"""
function suspects(log::AbstractString)
    found = Tuple{Int,String,String}[]
    inblock = false
    for line in eachline(log)
        if occursin("The following tests gave p-values outside", line)
            inblock = true
            continue
        end
        inblock || continue
        # The block is bounded by rules of dashes and ends at the first one
        # that follows an entry; anything else before the first entry (the
        # eps legends, the column header) simply does not match.
        if occursin(r"^\s*-{10,}\s*$", line)
            isempty(found) || break
            continue
        end
        m = match(r"^\s*(\d+)\s{1,}(\S.*?)\s{2,}(\S.*?)\s*$", line)
        m === nothing && continue
        push!(found, (parse(Int, m[1]), strip(m[2]), strip(m[3])))
    end
    return found
end

"""
    finished(log) -> Bool

Whether a battery log carries a complete summary. TestU01 closes every one with
"All tests were passed", or "All other tests were passed" when it singled some
out. A log without that line is an interrupted run -- the campaign leaves one
behind for every restart, and three of them sit in some of its run directories
-- and the difference matters: such a log has no suspects to *read*, which is
not the same as having none to find.
"""
finished(log::AbstractString) =
    any(occursin(r"All (other )?tests were passed", l) for l in eachline(log))

"""
    newest_finished(dir) -> String

The most recent complete battery log in a run directory. Pointing `--log` at a
directory is the useful thing to do: a run that was restarted leaves several
logs there and only the last one ran to the end.
"""
function newest_finished(dir::AbstractString)
    logs = filter(f -> endswith(f, ".log") && !endswith(f, "driver.log"),
                  joinpath.(dir, readdir(dir)))
    done = filter(finished, logs)
    isempty(done) && error("no complete battery log in $dir " *
                           "(found $(length(logs)); none carries a summary, so every " *
                           "run there was interrupted)")
    return done[argmax(mtime.(done))]
end

"""
    log_bits(log) -> Float64

The `# bits = N` line validate.jl writes for a bit-level battery, or 0 when the
log carries none. A replay of Alphabit or Rabbit has to consume the same number
of bits as the run it replicates, and that number is not otherwise recoverable
from the log.
"""
function log_bits(log::AbstractString)
    for line in eachline(log)
        startswith(line, "#") || break
        m = match(r"^#\s*bits\s*=\s*([0-9.eE+]+)\s*$", line)
        m === nothing || return parse(Float64, m[1])
    end
    return 0.0
end

"""
    header(log) -> (battery, suite, generator)

validate.jl writes `# <battery> / <suite> / <generator>` as the first line of
every battery log, so a replay can be asked for by naming the log alone.
"""
function header(log::AbstractString)
    line = open(readline, log)
    m = match(r"^#\s*(\w+)\s*/\s*(\w+)\s*/\s*(\S+)\s*$", line)
    m === nothing && error("$log does not start with the '# battery / suite / generator' " *
                           "line validate.jl writes; pass --battery, --suite and " *
                           "--generator instead")
    return String(m[1]), String(m[2]), String(m[3])
end

"""
    build_disjoint(suite, name, streams, values, skip)

The generator the original run used, advanced past `skip` streams first, so
that nothing it produces was seen by the run being replayed.
"""
function build_disjoint(suite::String, name::String, streams::Int, values::Int, skip::Int)
    g = stream_gen(name)
    for _ in 1:skip
        next_stream!(g)
    end
    if suite == "single"
        return TU01.Gen(_single_stream(next_stream!(g)), name)
    elseif suite == "bits"
        return TU01.bitgen(_bit_stream(next_stream!(g)), name)
    elseif suite == "interleaved"
        return TU01.Gen(_interleaved([next_stream!(g) for _ in 1:streams], values), name)
    end
    error("unknown suite: $suite")
end

function parse_args(args)
    opts = (log = "", battery = "", suite = "", generator = "", tests = Int[],
            reps = 3, skip = DEFAULT_SKIP, streams = 64, values = 1, bits = 0.0,
            out = "", list = false)
    for a in args
        if a == "--list"
            opts = merge(opts, (list = true,))
        elseif startswith(a, "--")
            body = a[3:end]
            i = findfirst('=', body)
            i === nothing && error("option needs a value: $a")
            k, v = body[1:i-1], body[i+1:end]
            k == "log"       ? (opts = merge(opts, (log = v,))) :
            k == "battery"   ? (opts = merge(opts, (battery = lowercase(v),))) :
            k == "suite"     ? (opts = merge(opts, (suite = lowercase(v),))) :
            k == "generator" ? (opts = merge(opts, (generator = v,))) :
            k == "tests"     ? (opts = merge(opts, (tests = parse.(Int, split(v, ',')),))) :
            k == "reps"      ? (opts = merge(opts, (reps = parse(Int, v),))) :
            k == "skip"      ? (opts = merge(opts, (skip = parse(Int, v),))) :
            k == "streams"   ? (opts = merge(opts, (streams = parse(Int, v),))) :
            k == "values"    ? (opts = merge(opts, (values = parse(Int, v),))) :
            k == "bits"      ? (opts = merge(opts, (bits = parse(Float64, v),))) :
            k == "out"       ? (opts = merge(opts, (out = v,))) :
            error("unknown option: $a")
        else
            error("unexpected argument: $a")
        end
    end

    if !isempty(opts.log)
        if isdir(opts.log)
            opts = merge(opts, (log = newest_finished(opts.log),))
        end
        isfile(opts.log) || error("no such log or directory: $(opts.log)")
        finished(opts.log) ||
            error("$(opts.log) has no summary, so that run was interrupted and its " *
                  "suspects were never printed. Point --log at the run's directory " *
                  "to take the last complete log, or name one yourself.")
        b, s, g = header(opts.log)
        isempty(opts.battery)   && (opts = merge(opts, (battery = b,)))
        isempty(opts.suite)     && (opts = merge(opts, (suite = s,)))
        isempty(opts.generator) && (opts = merge(opts, (generator = g,)))
        isempty(opts.tests)     && (opts = merge(opts, (tests = first.(suspects(opts.log)),)))
        isempty(opts.out)       && (opts = merge(opts, (out = dirname(opts.log),)))
        opts.bits == 0          && (opts = merge(opts, (bits = log_bits(opts.log),)))
    end

    isempty(opts.battery) && error("--battery is required (or --log, which carries it)")
    known_battery(opts.battery) || error("unknown battery: $(opts.battery)")
    isempty(opts.suite) && error("--suite is required (or --log)")
    isempty(opts.generator) && error("--generator is required (or --log)")
    isempty(opts.out) && (opts = merge(opts, (out = "testu01-results",)))
    opts.reps >= 1 || error("--reps must be at least 1")
    opts.skip >= 1 || error("--skip must be at least 1: the point is disjoint output")
    if opts.battery in BIT_BATTERIES && opts.bits <= 0
        error("$(opts.battery) consumes a fixed number of bits, and a replay has to " *
              "consume the same number as the run it replicates. Pass --bits, or " *
              "--log pointing at a run whose header records it.")
    end

    n = ntests(opts.battery)
    for t in opts.tests
        1 <= t <= n || error("test $t is not in $(opts.battery), which has $n tests")
    end
    return opts
end

function main(args)
    opts = parse_args(args)

    println("TestU01 replay -- ", opts.battery, " / ", opts.suite, " / ", opts.generator)
    println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
    println(provenance_line(PROV))
    if !isempty(opts.log)
        println("source log: ", opts.log)
        for (t, name, p) in suspects(opts.log)
            @printf("  test %3d  %-32s p = %s\n", t, name, p)
        end
    end

    if isempty(opts.tests)
        println("\nNothing to replay: that run singled out no p-value.")
        return
    end

    println()
    println("replaying tests ", join(opts.tests, ", "), ", ", opts.reps, " time(s) each")
    opts.battery in BIT_BATTERIES && @printf("%.0f bits per replication, as the original run\n", opts.bits)
    println("on streams ", opts.skip + 1, " and up, which the original run never drew from")
    if opts.list
        println("\n--list given, nothing run.")
        return
    end

    rep = zeros(Cint, ntests(opts.battery) + 1)
    for t in opts.tests
        rep[t + 1] = opts.reps       # rep[i] in C is rep[i+1] here
    end

    gen = build_disjoint(opts.suite, opts.generator, opts.streams, opts.values, opts.skip)

    mkpath(opts.out)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    tag = replace(opts.generator, r"[^\w]" => "_")
    log = joinpath(opts.out, "$(opts.battery)-$(opts.suite)-$tag-replay-$stamp.log")
    println("\n--> ", log)
    flush(stdout)

    stats = TU01.Statistic[]
    elapsed = TU01.withgen(gen) do g
        open(log, "w") do io
            println(io, "# ", opts.battery, " / ", opts.suite, " / ", opts.generator)
            println(io, "# Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Sys.MACHINE)
            println(io, "# ", provenance_line(PROV))
            println(io, "# started ", now())
            println(io, "# REPLAY of tests ", join(opts.tests, ", "), ", ", opts.reps, " time(s) each")
            println(io, "# streams skipped before building the generator: ", opts.skip)
            opts.battery in BIT_BATTERIES && @printf(io, "# bits = %.0f\n", opts.bits)
            isempty(opts.log) || println(io, "# source log: ", opts.log)
            opts.suite == "interleaved" &&
                println(io, "# ", opts.streams, " streams, ", opts.values, " value(s) each, round-robin")
            flush(io)
            @elapsed redirect_stdout(io) do
                try
                    TU01.line_buffer_stdout()
                    TU01.verbose!(true)
                    stats = TU01.repeat_battery(Symbol(opts.battery), g, rep; nb = opts.bits)
                finally
                    TU01.flush_c_stdout()
                end
            end
        end
    end

    index = joinpath(opts.out, "replays.tsv")
    isfile(index) || open(io -> println(io,
        "timestamp\tbattery\tsuite\tgenerator\ttests\treps\tskip\tseconds\tlog\t" *
        "julia\tcpu\tcommit\tdirty\tpkgversion\tsource\tstatistics\tsuspect\tfailed"), index, "w")
    open(index, "a") do io
        println(io, join((now(), opts.battery, opts.suite, opts.generator,
                          join(opts.tests, ","), opts.reps, opts.skip,
                          round(elapsed, digits = 1), log, VERSION, Sys.CPU_NAME,
                          PROV.commit, PROV.dirty, PROV.version, opts.log,
                          length(stats), length(TU01.suspects(stats)),
                          length(TU01.failures(stats))), '\t'))
    end

    # Every statistic the replay produced, not only the ones a battery would
    # single out: what settles a suspect is the spread of the replications, and
    # the guide is explicit that reporting the p-values "provides more
    # information" than a verdict against a threshold.
    pvals = joinpath(opts.out, "replay-pvalues.tsv")
    isfile(pvals) || open(io -> println(io,
        "timestamp\tbattery\tsuite\tgenerator\treps\tindex\tname\tp\tclass\tlog"), pvals, "w")
    open(pvals, "a") do io
        for st in stats
            println(io, join((now(), opts.battery, opts.suite, opts.generator, opts.reps,
                              st.index, st.name, st.p, TU01.classify(st.p),
                              basename(log)), '\t'))
        end
    end

    @printf("    done in %.1f s\n", elapsed)
    println()
    for st in stats
        @printf("  %3d  %-34s p = %-24g %s\n", st.index, st.name, st.p,
                TU01.classify(st.p) === :pass ? "" : string("<-- ", TU01.classify(st.p)))
    end
    println()
    println("Read those against the originals. A p-value that stays extreme across")
    println("independent output is the generator failing the test; one that scatters")
    println("was the fluke the replay existed to rule out. Report the p-values, not")
    println("a verdict: the TestU01 guide is explicit that this is more informative")
    println("than reject/do-not-reject against a fixed threshold.")
    println("\nIndex: ", index)
    println("p-values: ", pvals)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
