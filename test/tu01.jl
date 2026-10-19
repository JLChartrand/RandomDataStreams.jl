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
    NativeGen

A generator TestU01 implements itself, made by one of its own constructors and
called by the batteries with no Julia in between: no callback, and no conversion
of ours between the generator and the test. Every generator of this package goes
through the same callback, so only such a generator can tell a property of a
test from a property of that path. Build one with [`aes`](@ref) or
[`ulec_mrg32k3a`](@ref); [`withgen`](@ref) and [`repeat_battery`](@ref) take it
like a [`Gen`](@ref).

TestU01's one-generator-at-a-time restriction is on its external generators,
whose state is a global; native ones keep their own, so `LIVE` is not claimed.
"""
mutable struct NativeGen
    ptr::Ptr{Cvoid}
    delete::Symbol      # the library's matching destructor
    keep::Any           # what the constructor was handed, kept alive with it
    name::String
end

"""
    aes(key; seed = zeros(UInt8, 16), mode = :ctr, r = 0, s = 16) -> NativeGen

TestU01's AES generator, `ucrypto_CreateAES`: the Rijndael cipher of Rijmen,
Bosselaers and Barreto, with a key of 16, 24 or 32 bytes. In counter mode
(`:ctr`, the default) block `j` is the encryption of `seed + j`; `:ofb` feeds
each block back, `:ktr` counts in the key. Of each 16-byte block the first `r`
bytes are dropped and the next `s` make 32-bit outputs, as the guide describes.
"""
function aes(key::AbstractVector{UInt8}; seed::AbstractVector{UInt8} = zeros(UInt8, 16),
             mode::Symbol = :ctr, r::Integer = 0, s::Integer = 16)
    length(key) in (16, 24, 32) || error("an AES key is 16, 24 or 32 bytes, not $(length(key))")
    length(seed) == 16 || error("the AES seed is one 16-byte block, not $(length(seed)) bytes")
    (0 <= r <= 15 && 1 <= s <= 16 && r + s <= 16) || error("need 0 <= r <= 15, 1 <= s <= 16, r + s <= 16")
    m = mode === :ofb ? 0 : mode === :ctr ? 1 : mode === :ktr ? 2 :
        error("mode is :ofb, :ctr or :ktr, not :$mode")
    k, sd = Vector{UInt8}(key), Vector{UInt8}(seed)
    ptr = GC.@preserve k sd ccall((:ucrypto_CreateAES, libtestu01), Ptr{Cvoid},
                                  (Ptr{UInt8}, Cint, Ptr{UInt8}, Cint, Cint, Cint),
                                  k, 8 * length(k), sd, m, r, s)
    ptr == C_NULL && error("ucrypto_CreateAES returned NULL")
    return NativeGen(ptr, :ucrypto_DeleteAES, (k, sd), "AES-$(8 * length(k)) $(uppercase(String(mode)))")
end

"""
    ulec_mrg32k3a(state = (12345, 12345, 12345, 12345, 12345, 12345)) -> NativeGen

TestU01's own MRG32k3a, `ulec_CreateMRG32k3a`, from the six-component state.
Its outputs are `z/(m1 + 1)`, the package's `MRG32k3a` construction, so from
the same state the two hand a battery the same numbers -- one through the
callback, one without it.
"""
function ulec_mrg32k3a(state::NTuple{6,<:Integer} = (12345, 12345, 12345, 12345, 12345, 12345))
    x = Float64.(state)
    ptr = ccall((:ulec_CreateMRG32k3a, libtestu01), Ptr{Cvoid},
                (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
                x[1], x[2], x[3], x[4], x[5], x[6])
    ptr == C_NULL && error("ulec_CreateMRG32k3a returned NULL")
    return NativeGen(ptr, :ulec_DeleteGen, nothing, "ulec MRG32k3a")
end

function free!(g::NativeGen)
    g.ptr == C_NULL && return nothing
    if g.delete === :ucrypto_DeleteAES
        ccall((:ucrypto_DeleteAES, libtestu01), Cvoid, (Ptr{Cvoid},), g.ptr)
    elseif g.delete === :ulec_DeleteGen
        ccall((:ulec_DeleteGen, libtestu01), Cvoid, (Ptr{Cvoid},), g.ptr)
    else
        error("no destructor for $(g.delete)")
    end
    g.ptr = C_NULL
    return nothing
end

"""
    withgen(body, g) -> body(g)

Run `body(g)` and free the generator afterwards, however it ends.
"""
function withgen(body, g::Union{Gen,NativeGen})
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

`corrected` is `nothing` except where [`correct`](@ref) has replaced TestU01's
law, and then it is the p-value the statistic is judged by; `p` stays the
library's, which is archived beside it. Read the one to judge by with
[`pvalue`](@ref).
"""
struct Statistic
    index::Int
    name::String
    p::Float64
    corrected::Union{Nothing,Float64}
end

Statistic(index::Integer, name::AbstractString, p::Real) = Statistic(index, name, p, nothing)

"The p-value a statistic is judged by: the corrected one if there is one, TestU01's otherwise."
pvalue(s::Statistic) = something(s.corrected, s.p)

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
    classify(s::Statistic)

A statistic is classified on [`pvalue`](@ref): its corrected p-value where it
has one.
"""
classify(p::Real) = (p < FAIL || p > 1 - FAIL) ? :fail :
                    (p < SUSPECT || p > 1 - SUSPECT) ? :suspect : :pass
classify(s::Statistic) = classify(pvalue(s))

"Statistics the decision rule singles out: [`pvalue`](@ref) outside [0.001, 0.999]."
suspects(stats::AbstractVector{Statistic}) = filter(s -> classify(s) !== :pass, stats)

"Statistics that are failures outright: [`pvalue`](@ref) outside [1e-10, 1 - 1e-10]."
failures(stats::AbstractVector{Statistic}) = filter(s -> classify(s) === :fail, stats)

# CollisionOver on small alphabets ---------------------------------------------
#
# BigCrush's tests 3 to 12 are CollisionOver: n overlapping t-tuples over an
# alphabet of d letters, so k = d^t cells, the collisions counted in each of N
# replications and their sum compared with a Poisson law. On a small alphabet
# that law is wrong for every generator. If tuples i and j collide, tuples i + 1
# and j + 1 collide too with probability 1/d, so collisions come in clumps, and
# the count's variance is (1 + 1/d)/(1 − 1/d) times its mean rather than equal
# to it. At d = 4 that is 5/3, which TestU01's own AES confirmed (1.76, 95%
# interval [1.50, 2.10]); a perfect generator is singled out by each of tests 11
# and 12 about 1.7% of the time rather than 0.2%. docs/src/validation.md has the
# calibration.
#
# The correction replaces the law and nothing else. It keeps TestU01's mean and
# its rule for turning a count into a p-value, and compares the count with a
# Pólya–Aeppli law of that mean: a Poisson number of clumps, each of 1 + G
# collisions, P(G = g) = (1 − 1/d) d^(−g). It was fixed before any corrected
# p-value was computed (wsc/correction-collisionover/PROTOCOL.md in the papers
# repository), and the decision rule now classifies these ten statistics on it.
#
# The count is not among the globals -- `bbattery_pVal[]` holds the p-value
# alone -- so it is read from the battery's log, where TestU01 prints it as an
# integer. The p-value is then recomputed from it under TestU01's law and
# compared with the double the library returned, which is what says that the
# block read is the statistic's own.

"""
    COLLISIONOVER

The batteries whose CollisionOver statistics are corrected, with what their
tests share -- N replications of n tuples, in k = d^t cells -- and each test by
its number, with the bits it drops (r) and its tuple size (t). BigCrush alone:
the calibration measured its tests, and Crush's, which have other parameters,
were not part of it.
"""
const COLLISIONOVER = Dict(
    :bigcrush => (N = 30, n = 20_000_000, k = 2^42,
                  tests = [3 => (r = 0, t = 2),   4 => (r = 9, t = 2),
                           5 => (r = 0, t = 3),   6 => (r = 16, t = 3),
                           7 => (r = 0, t = 7),   8 => (r = 24, t = 7),
                           9 => (r = 0, t = 14), 10 => (r = 27, t = 14),
                           11 => (r = 0, t = 21), 12 => (r = 28, t = 21)]))

"""
    collisionover_mean(N, n, k, t) -> Float64

The mean TestU01 gives the sum of N counts: m²/(2k) − m³/(6k²) per replication,
for m = n − t + 1 overlapping tuples in k cells. The log prints it rounded
(1364.24 for BigCrush); this form was recovered from the archived campaigns, and
reproduces every CollisionOver p-value they hold to within 3 × 10⁻¹¹.
"""
function collisionover_mean(N::Integer, n::Integer, k::Integer, t::Integer)
    m, kf = Float64(n - t + 1), Float64(k)
    return N * (m^2 / (2kf) - m^3 / (6kf^2))
end

"""
    polya_aeppli_tails(c, mean, q) -> (P(S ≤ c), P(S ≥ c))

The two tails at `c` of a Pólya–Aeppli variable S of the given mean: a Poisson
number of clumps, of mean `mean (1 − q)`, each of 1 + G with
P(G = g) = (1 − q) q^g. At `q = 0` every clump is one collision, and S is
Poisson.
"""
function polya_aeppli_tails(c::Integer, mean::Real, q::Real)
    θ = mean * (1 - q)
    top = ceil(Int, mean + 60 * sqrt(mean * (1 + q) / (1 - q)))  # 60 sd: past the last double
    c < 0 && return (0.0, 1.0)
    c > top && return (1.0, 0.0)
    # Panjer's recursion for a compound Poisson law, P(n) = (θ/n) Σⱼ j fⱼ P(n − j),
    # with clump sizes fⱼ = (1 − q) q^(j−1). Its two sums over j follow from one
    # n to the next, so the whole law up to `top` costs a pass, and every term
    # is positive: nothing cancels. P(0) = e^(−θ) underflows, so `w` holds the
    # law up to a factor, rescaled when it grows; the tails are ratios of sums.
    w = Vector{Float64}(undef, top + 1)     # w[n + 1] ∝ P(S = n)
    w[1] = 1.0
    a = b = 1.0                             # Σⱼ q^(j−1) w(n − j) and Σⱼ j q^(j−1) w(n − j)
    for n in 1:top
        w[n + 1] = θ * (1 - q) / n * b
        a, b = w[n + 1] + q * a, w[n + 1] + q * (a + b)
        if w[n + 1] > 1e250
            w[1:n + 1] .*= 1e-250
            a *= 1e-250
            b *= 1e-250
        end
    end
    total = sum(w)
    return sum(@view w[1:c + 1]) / total, sum(@view w[c + 1:end]) / total
end

"""
    pdisc(left, right) -> Float64

`gofw_pDisc`, TestU01's p-value for a discrete statistic from P(Y ≤ y) and
P(Y ≥ y): the right tail above the median, the complement of the left tail
below it, and 0.5 at it.
"""
pdisc(left::Real, right::Real) = right < left ? right : left > 0.5 ? 0.5 : 1 - left

"""
    collisionover_p(count, N, n, d, t; q = 1/d) -> Float64

The p-value of `count` collisions, summed over N CollisionOver replications of
n tuples on an alphabet of `d` letters: corrected at the default, and TestU01's
own at `q = 0`.
"""
function collisionover_p(count::Integer, N::Integer, n::Integer, d::Integer, t::Integer;
                         q::Real = 1 / d)
    mean = collisionover_mean(N, n, big(d)^t, t)
    return pdisc(polya_aeppli_tails(count, mean, q)...)
end

const _PARAMS = r"N =\s*(\d+),\s*n =\s*(\d+),\s*r =\s*(\d+),\s*d =\s*(\d+),\s*t =\s*(\d+),"
const _OBSERVED = r"Observed number of collisions\s*:\s*(\d+)"

"""
    collision_counts(log) -> Vector{NamedTuple}

Every CollisionOver of a TestU01 log, in the order the battery ran them, as
`(N, n, r, d, t, count)`: the parameters from the line that opens the test, the
count from "Observed number of collisions" under "Results of CollisionOver
test". Other tests print collision counts of their own, and that heading keeps
them out. A battery run with [`verbose!`](@ref)`(false)` prints neither, and
gives an empty list.
"""
function collision_counts(log::AbstractString)
    out = NamedTuple{(:N, :n, :r, :d, :t, :count),NTuple{6,Int}}[]
    params, inresult = nothing, false
    for line in eachline(log)
        m = match(_PARAMS, line)
        if m !== nothing
            params, inresult = parse.(Int, m.captures), false
        elseif startswith(line, "Results of CollisionOver test")
            inresult = true
        elseif inresult && params !== nothing && (m = match(_OBSERVED, line)) !== nothing
            N, n, r, d, t = params
            push!(out, (N = N, n = n, r = r, d = d, t = t, count = parse(Int, m[1])))
            inresult = false
        end
    end
    return out
end

"""
    correct(battery, stats, log) -> Vector{Statistic}

`stats` with the CollisionOver statistics of `battery` corrected, from the
counts in `log`, the battery's own report. A battery not in
[`COLLISIONOVER`](@ref), or a run without CollisionOver statistics, comes back
unchanged. Works on a repeat battery as on the full one: the statistics and the
blocks of the log are paired in the order they ran.

Throws if the log does not account for the statistics -- a block missing, a
tuple size or a cell count that is not the battery's, or a count from which
TestU01's law does not give back TestU01's p-value. A caller that cannot afford
to lose the run should keep `stats` as they are and say so.
"""
function correct(battery::Symbol, stats::AbstractVector{Statistic}, log::AbstractString)
    shared = get(COLLISIONOVER, battery, nothing)
    shared === nothing && return collect(stats)
    at = findall(s -> occursin(r"^CollisionOver, t = \d+$", s.name), stats)
    isempty(at) && return collect(stats)
    blocks = collision_counts(log)
    length(blocks) == length(at) ||
        error("$log has $(length(blocks)) CollisionOver results for $(length(at)) statistics" *
              (isempty(blocks) ? "; a quiet battery does not print the counts" : ""))
    out = collect(stats)
    for (i, b) in zip(at, blocks)
        s = stats[i]
        s.name == "CollisionOver, t = $(b.t)" && (b.N, b.n) == (shared.N, shared.n) &&
            big(b.d)^b.t == shared.k ||
            error("statistic $(s.index) ($(s.name)) is paired with a block of $log " *
                  "with N = $(b.N), n = $(b.n), d = $(b.d), t = $(b.t)")
        testu01 = collisionover_p(b.count, b.N, b.n, b.d, b.t; q = 0)
        abs(testu01 - s.p) <= 1e-9 ||
            error("statistic $(s.index) ($(s.name)): $(b.count) collisions give " *
                  "p = $testu01 under TestU01's law, and TestU01 returned $(s.p)")
        out[i] = Statistic(s.index, s.name, s.p, collisionover_p(b.count, b.N, b.n, b.d, b.t))
    end
    return out
end

"""
    corrected_tests(battery, log) -> Dict{Int,Float64}

The corrected p-value of each CollisionOver test in a battery's log, by the
test number its closing summary prints -- which is what a replay asks for, and
which [`correct`](@ref) does not see, since a statistic carries an index
instead. A test is recognised by its r and t rather than by its place in the
log. Empty for a battery not in [`COLLISIONOVER`](@ref), and for a quiet log.
"""
function corrected_tests(battery::Symbol, log::AbstractString)
    out = Dict{Int,Float64}()
    shared = get(COLLISIONOVER, battery, nothing)
    shared === nothing && return out
    blocks = collision_counts(log)
    for (test, at) in shared.tests
        i = findfirst(b -> (b.N, b.n, b.r, b.t) == (shared.N, shared.n, at.r, at.t), blocks)
        i === nothing && continue
        b = blocks[i]
        out[test] = collisionover_p(b.count, b.N, b.n, b.d, b.t)
    end
    return out
end

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
    @eval function $jl(g::Union{Gen,NativeGen})
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
function repeat_battery(battery::Symbol, g::Union{Gen,NativeGen}, rep::Vector{Cint};
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
