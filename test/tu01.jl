# The TestU01 layer.
#
# Everything in this package that talks to TestU01 goes through here: the test
# suite, `scripts/testu01/validate.jl` and `scripts/testu01/replay.jl`. It calls
# the C library directly, through the same `TestU01_jll` artifact RNGTest.jl
# uses, and exists because the wrapper stopped carrying its weight:
#
#   * The battery functions return nothing. TestU01 fills `bbattery_pVal[]`,
#     `bbattery_NTests` and `bbattery_TestNames[]` with the answer and RNGTest
#     exposes none of them, so every p-value had to be read back out of the
#     formatted report -- as text, in TestU01's printing forms ("eps",
#     "1 -  3.0e-9"), which is a poor thing to classify against a threshold of
#     1e-10. Reading the globals gives the exact doubles instead.
#
#   * The test suite could not run the real battery for that reason. It drove
#     `RNGTest.smallcrushJulia`, a Julia reimplementation that keeps a subset of
#     the statistics and distributes its ten tests with `pmap` -- which under
#     `julia -p N` would serialise a copy of the generator to each worker and
#     test one stream ten times. With `bbattery_pVal[]` the C battery returns
#     its 15 statistics and the reimplementation is not needed.
#
#   * `Unif01` does not root its callback. `@cfunction($f, ...)` returns a
#     `Base.CFunction`, documented in Base as the "garbage-collection handle"
#     that must outlive every C call through the pointer; RNGTest builds it in a
#     local and drops it when the constructor returns. The callback allocates,
#     so the GC does run during a battery. It has not been seen to crash, but
#     nothing makes it safe, and a battery that dies at hour eighteen of a
#     BigCrush costs a day. `Gen` holds the handle and the closure for as long
#     as the generator lives.
#
#   * `Unif01(f::Function, name)` draws 100 values to check the return type and
#     the range before TestU01 sees the generator, so a battery over a closure
#     did not start at the stream's origin -- and each test of `smallcrushJulia`
#     paid it again, since every test built its own. The inferred return type is
#     checked here instead, which costs nothing and catches the same mistake
#     before it can segfault.
#
#   * `bbattery_Repeat*`, `Alphabit` and `Rabbit` have no wrapper at all.
#
# The layer lives under `test/` rather than beside the scripts because
# `Pkg.test()` has to be self-contained, while a script may reach anywhere in
# the checkout it belongs to.

module TU01

using TestU01_jll: libtestu01

# Nothing is exported: every caller reaches these as `TU01.x`. `suspects` in
# particular would collide with the log parser of the same name in replay.jl.

# Generators ------------------------------------------------------------------

"""
    Gen

A live `unif01_Gen`, together with everything on the Julia side that TestU01
will call into: the `Base.CFunction` trampoline and the closure behind it. Both
are fields rather than locals because the C library keeps the raw pointer and
calls it for the length of a battery — freeing the handle frees the trampoline,
and the next call from C lands in memory that is no longer ours.

Build one with [`Gen`](@ref) or [`bitgen`](@ref) and release it with
[`free!`](@ref); [`withgen`](@ref) does both around a block.
"""
mutable struct Gen
    ptr::Ptr{Cvoid}
    cf::Base.CFunction
    f::Any
    bits::Bool
    name::String
end

# TestU01 keeps its generator in a global and crashes when a second one is
# created, which RNGTest's own source records as the reason it deletes rather
# than finalizes. An error naming the generator still alive is a better failure
# than that crash, and the check costs nothing.
const LIVE = Ref{Union{Nothing,Gen}}(nothing)

function _claim(name::AbstractString)
    live = LIVE[]
    live === nothing && return nothing
    error("""
        a TestU01 generator ($(isempty(live.name) ? "unnamed" : live.name)) is still live,
        and the library keeps exactly one: creating "$name" now would crash it.
        Free the first one -- `withgen` does it for you -- before building another.""")
end

# The callback runs from C with no Julia frame above it to catch anything, so a
# return type the compiler cannot pin down is not a type error but a segfault.
# Checking the inferred type is the whole of what RNGTest's 100 draws achieved,
# without consuming the stream to do it.
function _check_return(f, ::Type{T}) where {T}
    rt = Base.return_types(f, ())
    (length(rt) == 1 && rt[1] === T) && return nothing
    error("""
        the generator callback must be inferred as $T, not $rt.
        TestU01 calls it from C, so an abstract return type crashes the process
        rather than raising. Put the call behind a function barrier that closes
        over a concrete generator.""")
end

"""
    Gen(f, name = "") -> Gen

A TestU01 generator over `f()`, which must return a `Float64` in [0, 1) — the
`U(0,1)` output the Crush batteries were designed for.
"""
function Gen(f, name::AbstractString = "")
    _claim(name)
    _check_return(f, Float64)
    cf = @cfunction($f, Float64, ())
    ptr = ccall((:unif01_CreateExternGen01, libtestu01), Ptr{Cvoid},
                (Ptr{UInt8}, Ptr{Cvoid}), name, cf)
    g = Gen(ptr, cf, f, false, String(name))
    LIVE[] = g
    return g
end

"""
    bitgen(f, name = "") -> Gen

A TestU01 generator over `f()`, which must return a `UInt32` — the integer path,
which the Crush batteries never see through a `U(0,1)` output (they examine its
30 most significant bits).
"""
function bitgen(f, name::AbstractString = "")
    _claim(name)
    @assert Cuint === UInt32
    _check_return(f, UInt32)
    cf = @cfunction($f, UInt32, ())
    ptr = ccall((:unif01_CreateExternGenBits, libtestu01), Ptr{Cvoid},
                (Ptr{UInt8}, Ptr{Cvoid}), name, cf)
    g = Gen(ptr, cf, f, true, String(name))
    LIVE[] = g
    return g
end

"""
    free!(g)

Release the generator. Not optional: the library holds one at a time.
"""
function free!(g::Gen)
    g.ptr == C_NULL && return nothing
    if g.bits
        ccall((:unif01_DeleteExternGenBits, libtestu01), Cvoid, (Ptr{Cvoid},), g.ptr)
    else
        ccall((:unif01_DeleteExternGen01, libtestu01), Cvoid, (Ptr{Cvoid},), g.ptr)
    end
    g.ptr = C_NULL
    LIVE[] === g && (LIVE[] = nothing)
    return nothing
end

"""
    withgen(body, g) -> body(g)

Run `body(g)` and free the generator afterwards, however it ends.
"""
function withgen(body, g::Gen)
    try
        return body(g)
    finally
        free!(g)
    end
end

# What TestU01 prints ---------------------------------------------------------

"""
    verbose!(on)

Turn TestU01's per-test reports on or off by writing `swrite_Basic`.

RNGTest used to clear this flag when it loaded, which is why the scripts had to
put it back; nothing clears it now, but the batteries are verbose by default and
the test suite wants them quiet, so the control stays. With the flag off, a
battery prints its closing summary and nothing else.
"""
verbose!(on::Bool) =
    unsafe_store!(cglobal((:swrite_Basic, libtestu01), Cint), Cint(on))

"""
    suspectp() / suspectp!(α)

`gofw_Suspectp`, the threshold a battery prints against: it singles out every
p-value outside [α, 1 − α], with α = 0.001 by default. It governs *printing*
only — the p-values themselves are in [`results`](@ref) whatever it is set to.
"""
suspectp() = unsafe_load(cglobal((:gofw_Suspectp, libtestu01), Float64))
suspectp!(α::Real) =
    unsafe_store!(cglobal((:gofw_Suspectp, libtestu01), Float64), Float64(α))

"""
    line_buffer_stdout() -> Bool

Put C's stdout in line-buffered mode.

A battery runs for hours inside one C call, and C stdout is block-buffered when
it is not a terminal: without this the log stays empty until the run ends, so an
interrupted run leaves nothing behind and progress cannot be followed with
`tail -f`. Returns false if the C symbol is not reachable, in which case the
report still arrives, just all at once.
"""
function line_buffer_stdout()
    try
        cstdout = unsafe_load(cglobal(:stdout, Ptr{Cvoid}))
        ccall(:fflush, Cint, (Ptr{Cvoid},), cstdout)
        return ccall(:setvbuf, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t),
                     cstdout, C_NULL, Cint(1), Csize_t(0)) == 0     # 1 = _IOLBF
    catch
        return false
    end
end

"Flush C's stdout. Needed before a redirection is undone, or the report lands on the console."
flush_c_stdout() = ccall(:fflush, Cint, (Ptr{Cvoid},), C_NULL)

# Results ---------------------------------------------------------------------

"""
    Statistic

One line of a battery's answer: the index TestU01 gave it, the name it prints
(`"RandomWalk1 H"`, `"MaxOft AD"`), and the p-value as the double the library
computed rather than the string it formats.
"""
struct Statistic
    index::Int
    name::String
    p::Float64
end

"""
    results() -> Vector{Statistic}

The p-values of the battery that ran last, read from `bbattery_pVal[]`,
`bbattery_NTests` and `bbattery_TestNames[]`. The arrays are 0-based and hold
one entry per *statistic*, not per test: SmallCrush reports 15 from 10 tests,
Crush 144 from 96. The summary report prints the test number instead, which is
what `rep[]` wants — see `suspects` in `scripts/testu01/replay.jl`.

Valid until the next battery runs, which overwrites the globals.
"""
function results()
    n = Int(unsafe_load(cglobal((:bbattery_NTests, libtestu01), Cint)))
    pv = cglobal((:bbattery_pVal, libtestu01), Float64)
    nm = cglobal((:bbattery_TestNames, libtestu01), Ptr{UInt8})
    out = Vector{Statistic}(undef, n)
    for i in 0:(n - 1)
        s = unsafe_load(nm, i + 1)                      # Julia's load is 1-based
        out[i + 1] = Statistic(i, s == C_NULL ? "" : unsafe_string(s),
                               unsafe_load(pv, i + 1))
    end
    return out
end

# The two thresholds of the TestU01 guide, and the reason there are three
# outcomes rather than two. `SUSPECT` is `gofw_Suspectp`, a printing threshold:
# a battery of Crush's size reports 144 statistics, so suspects arrive by chance
# in any campaign and mean nothing on their own. `FAIL` is the guide's own line
# -- "if it is less than 1e-10, one can obviously conclude that the generator
# fails the test". Everything between the two has to be replayed before it can
# be called anything; docs/src/validation.md works an example through.
const SUSPECT = 0.001
const FAIL = 1e-10

"""
    classify(p) -> :fail | :suspect | :pass
"""
classify(p::Real) = (p < FAIL || p > 1 - FAIL) ? :fail :
                    (p < SUSPECT || p > 1 - SUSPECT) ? :suspect : :pass

"Statistics a battery would single out: p outside [0.001, 0.999]."
suspects(stats::AbstractVector{Statistic}) = filter(s -> classify(s.p) !== :pass, stats)

"Statistics that are failures outright: p outside [1e-10, 1 - 1e-10]."
failures(stats::AbstractVector{Statistic}) = filter(s -> classify(s.p) === :fail, stats)

# Batteries -------------------------------------------------------------------
#
# Each runs a battery over a live generator and returns its statistics. The
# generator is not freed here: a replay runs a second battery over the same one.

# `pseudoDIEHARD` and `FIPS_140_2` are deliberately not here: the package does
# not run them, and `FIPS_140_2` leaves `bbattery_NTests` at zero, so a wrapper
# would hand back an empty vector rather than an answer.
for (jl, c) in ((:smallcrush, :bbattery_SmallCrush),
                (:crush,      :bbattery_Crush),
                (:bigcrush,   :bbattery_BigCrush))
    @eval function $jl(g::Gen)
        GC.@preserve g begin
            ccall(($(QuoteNode(c)), libtestu01), Cvoid, (Ptr{Cvoid},), g.ptr)
        end
        return results()
    end
end

"""
    rabbit(g, nb) / alphabit(g, nb, r, s) / blockalphabit(g, nb, r, s)

The bit-level batteries, which take the number of bits `nb` to use rather than
running to a fixed size. `alphabit` and `blockalphabit` test bits `r+1` through
`r+s` of each output word: `(0, 32)` is the whole word of a `bitgen` generator.

These are what a `bits` suite ought to run — Crush and BigCrush examine the 30
most significant bits of a `U(0,1)` output and were never meant for an integer
stream.
"""
function rabbit(g::Gen, nb::Real)
    GC.@preserve g begin
        ccall((:bbattery_Rabbit, libtestu01), Cvoid, (Ptr{Cvoid}, Cdouble), g.ptr, nb)
    end
    return results()
end

for (jl, c) in ((:alphabit, :bbattery_Alphabit), (:blockalphabit, :bbattery_BlockAlphabit))
    @eval function $jl(g::Gen, nb::Real, r::Integer = 0, s::Integer = 32)
        GC.@preserve g begin
            ccall(($(QuoteNode(c)), libtestu01), Cvoid,
                  (Ptr{Cvoid}, Cdouble, Cint, Cint), g.ptr, nb, r, s)
        end
        return results()
    end
end

"""
    NTESTS

Tests per battery — the size `rep[]` is indexed by, and not the number of
statistics the battery reports (Crush: 96 tests, 144 statistics; Alphabit: 9
tests, 17 statistics).

The two bit-level counts were established by running each repeat battery with
`rep` set for one test at a time and seeing which numbers produced statistics.
Rabbit is 26 even though a short run leaves two of them empty: tests 22 and 23
are the larger `MatrixRank`s, which it skips when `nb` is too small to feed
them.
"""
const NTESTS = Dict(:smallcrush => 10, :crush => 96, :bigcrush => 106,
                    :alphabit => 9, :rabbit => 26)

"""
    repeat_battery(battery, g, rep; nb = 0, r = 0, s = 32) -> Vector{Statistic}

`bbattery_Repeat<Battery>`: apply test *i* of the battery `rep[i]` times and
skip the rest. Resolving three suspects of a Crush run costs minutes against the
hours the battery itself took, which is what makes the guide's "replicate until
suspicion disappears" affordable at BigCrush.

`rep` is indexed by the battery's own test numbers, which start at 1, so it
needs one more element than the battery has tests; element 0 is never read. The
bit-level batteries need the same `nb`, `r` and `s` the original run used, or
the replay is not a replication of it.
"""
function repeat_battery(battery::Symbol, g::Gen, rep::Vector{Cint};
                        nb::Real = 0, r::Integer = 0, s::Integer = 32)
    n = get(NTESTS, battery, nothing)
    n === nothing && error("no repeat battery for $battery")
    length(rep) == n + 1 ||
        error("rep has $(length(rep)) elements; $battery needs $(n + 1)")
    (battery === :alphabit || battery === :rabbit) && nb <= 0 &&
        error("$battery needs the number of bits the original run used (nb)")
    GC.@preserve g rep begin
        # ccall needs a literal symbol, so the batteries are spelled out.
        if battery === :smallcrush
            ccall((:bbattery_RepeatSmallCrush, libtestu01), Cvoid,
                  (Ptr{Cvoid}, Ptr{Cint}), g.ptr, rep)
        elseif battery === :crush
            ccall((:bbattery_RepeatCrush, libtestu01), Cvoid,
                  (Ptr{Cvoid}, Ptr{Cint}), g.ptr, rep)
        elseif battery === :bigcrush
            ccall((:bbattery_RepeatBigCrush, libtestu01), Cvoid,
                  (Ptr{Cvoid}, Ptr{Cint}), g.ptr, rep)
        elseif battery === :rabbit
            ccall((:bbattery_RepeatRabbit, libtestu01), Cvoid,
                  (Ptr{Cvoid}, Cdouble, Ptr{Cint}), g.ptr, nb, rep)
        elseif battery === :alphabit
            ccall((:bbattery_RepeatAlphabit, libtestu01), Cvoid,
                  (Ptr{Cvoid}, Cdouble, Cint, Cint, Ptr{Cint}), g.ptr, nb, r, s, rep)
        end
    end
    return results()
end

end # module
