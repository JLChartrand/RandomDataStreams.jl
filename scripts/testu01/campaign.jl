#!/usr/bin/env julia

# Campaign driver for the TestU01 batteries.
#
# `validate.jl` runs one battery, in this process, serially. That is the right
# shape for one generator and the wrong shape for a BigCrush sweep, which takes
# days to weeks: a single process means one crash loses everything in flight, no
# parallelism, and no way to stop and resume. This driver runs the matrix as
# independent OS processes, records what finished, and skips it on the next run.
#
#     julia scripts/testu01/campaign.jl --battery=bigcrush --calibrate
#     julia scripts/testu01/campaign.jl --battery=bigcrush --jobs=6
#     julia scripts/testu01/campaign.jl --status
#
# Options
#   --battery=<name>                      REQUIRED, no default; see BATTERIES
#   --bits-battery=<name>[,<name>]        what the bits suite runs
#                                         (default alphabit,rabbit)
#   --bits=<n>                            size for alphabit/rabbit
#   --suite=single|interleaved|bits|all   default all
#   --generator=<name>|all                default all
#   --jobs=<n>                            concurrent processes (default: cores/4)
#   --out=<dir>                           default testu01-results
#   --calibrate                           run ONE job, report its cost, exit
#   --status                              report progress and exit
#   --list                                print what remains and exit
#
# Resuming. `summary.tsv` in the output directory is the record of finished
# work: one line per (battery, suite, generator) that completed. Re-running the
# same command skips those and does the rest. Interrupting loses only the jobs
# actually in flight.
#
# Stopping cleanly. `touch <out>/STOP` makes the driver stop launching new jobs
# and wait for the running ones -- the way to end a campaign without throwing
# away a battery that is twenty hours in. Ctrl-C does the same.
#
# TestU01 keeps global state in C and is not thread-safe, which is the other
# reason each job is its own process rather than a task.

using Printf, Dates

# Resolve the script environment once, here in the driver, before any job runs.
# The jobs do the same check and find it already done; without this first pass
# a cold start would have every one of them writing the same Manifest at once.
include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))
const PROV = provenance()

const HERE      = @__DIR__
const VALIDATE  = joinpath(HERE, "validate.jl")
const SUITES    = ["single", "interleaved", "bits"]
const BATTERIES = ["smallcrush", "crush", "bigcrush", "alphabit", "rabbit"]

"Batteries that take a size in bits, which is forwarded to the jobs."
const BIT_BATTERIES = ("alphabit", "rabbit")

# What the `bits` suite runs, which is not what the other two run.
#
# Crush and BigCrush examine the 30 most significant bits of a U(0,1) output and
# were never built for an integer stream; running one of them over the bit path
# was what we had before TestU01's own bit-level batteries were reachable. They
# are reachable now, so the bits leg of a campaign runs those instead, whatever
# `--battery` says for the other two suites. `--bits-battery` overrides it --
# passing the same name as `--battery` restores the old, uniform matrix.
const BITS_BATTERIES = ["alphabit", "rabbit"]

# The generator list is `validate.jl`'s, read from it rather than duplicated, so
# that adding a generator there adds it to the campaign.
function generator_names()
    src = read(VALIDATE, String)
    block = match(r"const GENERATOR_STREAMS = Pair\{String,Any\}\[(.*?)\n\]"s, src)
    block === nothing && error("cannot find the generator list in $VALIDATE")
    return [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"\s*=>", block.captures[1])]
end

function parse_args(args)
    # No default battery. `include`ing this file, or running it with a stray
    # argument, must not start a two-week BigCrush sweep by accident -- which is
    # exactly what a default did once.
    opts = Dict{String,Any}("battery" => "", "suite" => "all",
                            "generator" => "all", "out" => "testu01-results",
                            "jobs" => max(1, Sys.CPU_THREADS ÷ 4), "bits" => "",
                            "bits-battery" => BITS_BATTERIES,
                            "calibrate" => false, "status" => false, "list" => false)
    for a in args
        if a in ("--calibrate", "--status", "--list")
            opts[a[3:end]] = true
        elseif startswith(a, "--")
            i = findfirst('=', a)
            i === nothing && error("option needs a value: $a")
            k, v = a[3:i-1], a[i+1:end]
            haskey(opts, k) || error("unknown option: $a")
            opts[k] = k == "jobs" ? parse(Int, v) :
                      k == "bits-battery" ? String.(split(v, ',')) : v
        else
            error("unexpected argument: $a")
        end
    end
    isempty(opts["battery"]) &&
        error("--battery is required: one of " * join(BATTERIES, ", "))
    opts["battery"] in BATTERIES || error("unknown battery: $(opts["battery"])")
    for b in opts["bits-battery"]
        b in BATTERIES || error("unknown battery for the bits suite: $b")
    end
    return opts
end

# The matrix, and what is left of it -------------------------------------------

"""
    batteries_for(suite, opts) -> Vector{String}

The batteries a campaign runs on one suite: `--battery` for `single` and
`interleaved`, and `--bits-battery` for `bits` -- two batteries by default,
which is why the matrix is no longer one job per (suite, generator).
"""
batteries_for(suite, opts) =
    suite == "bits" ? opts["bits-battery"] : [opts["battery"]]

function matrix(opts)
    names  = opts["generator"] == "all" ? generator_names() : [opts["generator"]]
    suites = opts["suite"] == "all" ? SUITES : [opts["suite"]]
    return [(b, s, g) for s in suites for g in names for b in batteries_for(s, opts)]
end

"Finished jobs, from the consolidated index and from each job's own directory."
function finished(out)
    done = Set{Tuple{String,String,String}}()
    for f in [joinpath(out, "summary.tsv");
              filter(isfile, [joinpath(root, "summary.tsv")
                              for (root, _, _) in walkdir(joinpath(out, "runs"), onerror = _ -> nothing)])]
        isfile(f) || continue
        for (i, line) in enumerate(eachline(f))
            i == 1 && startswith(line, "timestamp") && continue
            fields = split(line, '\t')
            length(fields) >= 4 && push!(done, (fields[2], fields[3], fields[4]))
        end
    end
    return done
end

job_id(b, s, g) = "$(b)-$(s)-$(replace(g, r"[^\w]" => "_"))"

# Running one job as its own process -------------------------------------------

function launch(job, opts)
    b, s, g = job
    dir = joinpath(opts["out"], "runs", job_id(b, s, g))
    mkpath(dir)
    cmd = `$(Base.julia_cmd()) --startup-file=no $VALIDATE
           --battery=$b --suite=$s --generator=$g --out=$dir`
    # The size only means anything to the bit-level batteries, and validate.jl
    # rejects an option it does not expect, so it is passed only when it applies.
    if !isempty(opts["bits"]) && b in BIT_BATTERIES
        cmd = `$cmd --bits=$(opts["bits"])`
    end
    io = open(joinpath(dir, "driver.log"), "a")
    println(io, "\n# launched ", now(), " by campaign.jl")
    flush(io)
    return (job = job, dir = dir, started = time(),
            proc = run(pipeline(cmd, stdout = io, stderr = io), wait = false), io = io)
end

function reap!(running, out, log)
    still = similar(running, 0)
    for r in running
        if process_running(r.proc)
            push!(still, r)
            continue
        end
        close(r.io)
        b, s, g = r.job
        elapsed = time() - r.started
        okay = success(r.proc)
        note(log, @sprintf("%-9s %-12s %-16s %s in %s", b, s, g,
                           okay ? "done" : "FAILED (exit $(r.proc.exitcode))", human(elapsed)))
        okay && open(joinpath(out, "summary.tsv"), "a") do io
            println(io, join((now(), b, s, g, round(elapsed, digits = 1),
                              relpath(r.dir, out), VERSION, Sys.CPU_NAME,
                              PROV.commit, PROV.dirty, PROV.version), '\t'))
        end
    end
    return still
end

# Reporting --------------------------------------------------------------------

function human(seconds)
    seconds < 90 && return @sprintf("%.0f s", seconds)
    seconds < 5400 && return @sprintf("%.1f min", seconds / 60)
    seconds < 172800 && return @sprintf("%.1f h", seconds / 3600)
    return @sprintf("%.1f d", seconds / 86400)
end

function note(log, msg)
    line = string(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", msg)
    println(line); flush(stdout)
    open(io -> println(io, line), log, "a")
end

function status(opts)
    out = opts["out"]
    all = matrix(opts)
    done = finished(out)
    left = [j for j in all if j ∉ done]
    times = Float64[]
    idx = joinpath(out, "summary.tsv")
    if isfile(idx)
        for (i, line) in enumerate(eachline(idx))
            i == 1 && startswith(line, "timestamp") && continue
            f = split(line, '\t')
            length(f) >= 5 && push!(times, something(tryparse(Float64, f[5]), 0.0))
        end
    end
    @printf("%s: %d of %d done, %d remaining\n", out, length(all) - length(left),
            length(all), length(left))
    if !isempty(times)
        mean_t = sum(times) / length(times)
        @printf("mean completed job: %s   remaining at %d job(s) in parallel: about %s\n",
                human(mean_t), opts["jobs"], human(mean_t * length(left) / opts["jobs"]))
    end
    isempty(left) || println("\nremaining:")
    for (b, s, g) in left
        @printf("  %-9s %-12s %s\n", b, s, g)
    end
    isfile(joinpath(out, "STOP")) && println("\nSTOP file present: the driver will not start new jobs.")
end

# Main -------------------------------------------------------------------------

function main(args)
    opts = parse_args(args)
    out  = opts["out"]
    mkpath(out)
    log = joinpath(out, "campaign.log")
    index = joinpath(out, "summary.tsv")
    isfile(index) || open(io -> println(io,
        "timestamp\tbattery\tsuite\tgenerator\tseconds\tdir\tjulia\tcpu\t" *
        "commit\tdirty\tpkgversion"), index, "w")

    if opts["status"]
        status(opts); return
    end

    todo = [j for j in matrix(opts) if j ∉ finished(out)]
    if opts["list"]
        status(opts); return
    end
    if isempty(todo)
        println("Nothing to do: every job in this matrix is already in $index.")
        return
    end
    opts["calibrate"] && (todo = todo[1:1])

    warn_if_dirty(PROV)
    note(log, "campaign start: $(length(todo)) job(s), $(opts["jobs"]) at a time, " *
              "Julia $VERSION, $(Sys.CPU_NAME), $(Sys.CPU_THREADS) threads")
    note(log, provenance_line(PROV))
    opts["calibrate"] && note(log, "calibration: one job only, to measure before committing")

    running = Any[]
    queue = copy(todo)
    stopfile = joinpath(out, "STOP")
    stopping = false
    Base.exit_on_sigint(false)

    try
        while !isempty(queue) || !isempty(running)
            if !stopping && isfile(stopfile)
                note(log, "STOP file seen: no new jobs, waiting for $(length(running)) in flight")
                stopping = true
            end
            while !stopping && !isempty(queue) && length(running) < opts["jobs"]
                job = popfirst!(queue)
                note(log, @sprintf("start  %-9s %-12s %s", job...))
                push!(running, launch(job, opts))
            end
            sleep(5)
            running = reap!(running, out, log)
            stopping && isempty(running) && break
        end
    catch err
        err isa InterruptException || rethrow()
        note(log, "interrupted: waiting for $(length(running)) job(s) in flight; " *
                  "re-run the same command to resume")
        while !isempty(running)
            sleep(5)
            running = reap!(running, out, log)
        end
    end

    note(log, "campaign end")
    println()
    status(opts)
end

main(ARGS)
