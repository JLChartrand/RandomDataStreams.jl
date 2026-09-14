# What the test suite adds on top of the TestU01 layer.
#
# The batteries themselves are in `tu01.jl`, which calls the C library directly.
# Here: the platform guard, the silence a test run wants from a battery, and the
# assertion that nothing was singled out.

using Test

@isdefined(TU01) || include("tu01.jl")

"""
    run_battery(f, what) -> result or nothing

Run `f()`, which drives a TestU01 battery, and return `nothing` instead when
the platform cannot hand TestU01 its callback.

The callback is built with `@cfunction` over a closure, which needs an
executable trampoline. Apple Silicon forbids memory that is both writable and
executable, so Julia raises `cfunction: closures are not supported on this
platform` and no battery can run there, whatever the generator.

The failure is caught at the point of use rather than predicted by a probe. A
synthetic probe is not trustworthy here: a closure over a constant is compiled
to a singleton, needs no trampoline, and succeeds on exactly the platform where
the real one fails. Only the real call answers the question. Any other error is
rethrown.
"""
function run_battery(f, what)
    try
        return f()
    catch err
        if occursin("closures are not supported", sprint(showerror, err))
            @info "$what skipped: this platform cannot build a C callback " *
                  "from a closure, which TestU01 needs to call back into Julia."
            return nothing
        end
        rethrow()
    end
end

"""
    silently(f) -> f()

Run `f` with TestU01's report sent to `devnull`.

A battery writes its report from C, so redirecting Julia's stdout is not enough
on its own: C stdout is block-buffered when it is not a terminal, and without
the flush the text is still in the C buffer when the descriptor is restored and
lands on the console instead. The p-values come back through
`bbattery_pVal[]` regardless, and are more use to a failing test than the
printed forms would be.
"""
function silently(f)
    TU01.verbose!(false)
    return redirect_stdout(devnull) do
        try
            return f()
        finally
            TU01.flush_c_stdout()
        end
    end
end

"Every statistic a battery singled out, one per line, for a test that failed."
function report(stats)
    io = IOBuffer()
    for s in stats
        println(io, "    ", lpad(s.index, 3), "  ", rpad(s.name, 32),
                " p = ", s.p, "  (", TU01.classify(s.p), ")")
    end
    return String(take!(io))
end

"""
    check_smallcrush(makegen, what)

Run SmallCrush over the generator `makegen()` returns and assert that it singles
nothing out: no p-value outside [0.001, 0.999] among the 15 statistics the
battery reports.

That threshold is `gofw_Suspectp`, a *printing* threshold, and over a campaign
of batteries a few suspects are expected from chance alone -- which is why the
on-demand harness replays them rather than reporting them (see
`docs/src/validation.md`). Here, where nine generators run one battery each on
every CI run, one suspect line is rare enough to be worth a human's attention,
so the suite treats it as a failure and prints the p-values that caused it.
"""
function check_smallcrush(makegen, what::AbstractString)
    stats = run_battery(what) do
        TU01.withgen(makegen()) do g
            silently(() -> TU01.smallcrush(g))
        end
    end
    if stats === nothing
        @test_skip "TestU01 unavailable on this platform"
        return nothing
    end
    # A battery that reports nothing has not run: `bbattery_NTests` stays at
    # zero, and an empty list would otherwise pass every assertion below.
    @test length(stats) == 15
    bad = TU01.suspects(stats)
    isempty(bad) || @warn "$what singled out $(length(bad)) of $(length(stats)) " *
                          "statistics:\n" * report(bad)
    @test isempty(bad)
    return stats
end
