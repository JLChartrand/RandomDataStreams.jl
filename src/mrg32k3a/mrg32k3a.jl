

const DEFAULT_SEED = PMF.DEFAULT_SEED
const checkseed = PMF.checkseed

"""
`MRG32k3a()` will generate an instance of `MRG32k3a` with initial seeds `DEFAULT_SEED = [ 12345, 12345, 12345, 12345, 12345, 12345 ]`

`MRG32k3a(x::Vector{Int})` will generate an instance of `MRG32k3a` with initial seeds `x`

`MRG32k3a(x::Vector{Int}, y::Vector{Int}, z::Vector{Int})` will generate an instance of `MRG32k3a` with `Cg = x`, `Bg = y` and `Ig = z`

# Examples

```jldoctest
julia> rng = MRG32k3a();

julia> rand(rng)
0.12701112204657714
```
"""
mutable struct MRG32k3a <: AbstractStreamableRNG
    Cg::Vector{Int64}  # the current state of the RNG
    Bg::Vector{Int64}  # the start point of the current substream
    Ig::Vector{Int64}  # the start point of the current stream
    function MRG32k3a(x::Vector{Int} = PMF.DEFAULT_SEED)
        @assert(PMF.checkseed(x))
        new(copy(x),copy(x),copy(x))
    end
    function MRG32k3a(x::Vector{Int}, y::Vector{Int}, z::Vector{Int})
        @assert(PMF.checkseed(x))
        @assert(PMF.checkseed(y))
        @assert(PMF.checkseed(z))
        return new(copy(x),copy(y),copy(z))
    end
end
#copy imported in RandomDataStreams.jl in src
function copy(m::MRG32k3a)
    MRG32k3a(copy(m.Cg), copy(m.Bg), copy(m.Ig))
end

# Uniform constructor: a plain integer seed, folded into the valid MRG32k3a
# seed space, so `MRG32k3a(12345)` means what `Xoshiro256pp(12345)` and
# `PCG64(12345)` mean.
MRG32k3a(seed::Integer) = MRG32k3a(_mrg_seed_words(UInt64(seed)))


"""
Advances the state of `rng` by one step and returns the raw pair `(p1, p2)`.
Shared by all `rand` methods to avoid code duplication.
"""
@inline function next_pair!(rng::MRG32k3a)
    Cg = rng.Cg
    @inbounds begin
        # The correction terms keep the argument of `%` non-negative, so the
        # residual needs no fix-up afterwards (Vigna's testless formulation).
        p1 = (PMF.a12 * Cg[2] - PMF.a13n * Cg[1] + PMF.corr1) % PMF.m1

        Cg[1] = Cg[2]
        Cg[2] = Cg[3]
        Cg[3] = p1

        p2 = (PMF.a21 * Cg[6] - PMF.a23n * Cg[4] + PMF.corr2) % PMF.m2

        Cg[4] = Cg[5]
        Cg[5] = Cg[6]
        Cg[6] = p2
    end
    return p1, p2
end

"""
Combines the two component outputs into `z` in `[1, m1]`.

Branchless: the arithmetic shift yields -1 when `p1 <= p2` and 0 otherwise, so
the modulus is added exactly when the difference is not already positive. Same
value as `p1 > p2 ? p1 - p2 : p1 + m1 - p2`, without the test.
"""
@inline function combine(p1::Int64, p2::Int64)
    r = p1 - p2
    return r - PMF.m1 * ((r - 1) >> 63)
end

"""
Resets a given random number generator to the beginning of the current stream.
"""
function reset_stream!(rng::MRG32k3a)::MRG32k3a
    rng.Cg[:] = rng.Ig
    rng.Bg[:] = rng.Ig
    return rng
end

"""
Resets a given random number generator to the beginning of the current substream.
"""
function reset_substream!(rng::MRG32k3a)::MRG32k3a
    rng.Cg[:] = rng.Bg
    return rng
end
"""
Takes a random number generator and shifts seed to next substream.
"""
function next_substream!(rng::MRG32k3a)::MRG32k3a
    rng.Bg[1:3] = PMF.MatVecModM(PMF.A1p76, view(rng.Bg, 1:3), PMF.m1)
    rng.Bg[4:6] = PMF.MatVecModM(PMF.A2p76, view(rng.Bg, 4:6), PMF.m2)
    for i = 1:6
        @inbounds rng.Cg[i] = rng.Bg[i]
    end
    return rng
end
#show imported in RandomDataStreams.jl in src
# Two-argument `show` is what interpolation, `@show` and container display call,
# so it stays on one line; the full three-anchor dump belongs to the REPL's
# `text/plain` rendering.
show(io::IO, rng::MRG32k3a) = print(io, "MRG32k3a(Cg = ", rng.Cg, ")")

function show(io::IO, ::MIME"text/plain", rng::MRG32k3a)
    print(io,"Full state of MRG32k3a generator:\nCg = $(rng.Cg)\nBg = $(rng.Bg)\nIg = $(rng.Ig)")
end

"""
    get_state(rng::MRG32k3a) -> Vector{Int}

Copy of the current state (`Cg`); restore it with [`set_state!`](@ref).
"""
function get_state(rng::MRG32k3a)::Array{Int, 1}
    return copy(rng.Cg)
end

"""
    set_state!(rng::MRG32k3a, state) -> rng

Restores the current position from a `get_state(rng)` value. Only the current
state moves; the stream and substream boundaries (`Bg`, `Ig`) are untouched.
"""
function set_state!(rng::MRG32k3a, state)
    s = Int.(collect(state))
    PMF.checkseed(s) || throw(ArgumentError("invalid MRG32k3a state"))
    rng.Cg[:] = s
    return rng
end

"""
    srand!(rng::MRG32k3a, seed) -> rng

Reseeds the generator, resetting the stream and substream boundaries to `seed`.
"""
srand!(rng::MRG32k3a, seed) = Random.seed!(rng, seed)


"""
Given a random number generator, jumps n steps forward if n > 0
(or -n steps backwards if n < 0), where

if e > 0, let n = 2^e + c;
if e < 0, let n = -2^(-e) + c;
if e = 0, let n = c.

# Examples

```jldoctest
julia> rng = MRG32k3a();

julia> advance_state!(rng, Int64(2), Int64(-1));   # skip n = 2^2 - 1 = 3 values

julia> rand(rng)
0.8258468629271136
```
"""
function advance_state!(rng::MRG32k3a, e::Int64, c::Int64)
    if c >= 0
        C1 = PMF.MatPowModM(PMF.A1p0, c, PMF.m1)
        C2 = PMF.MatPowModM(PMF.A2p0, c, PMF.m2)
    else
        C1 = PMF.MatPowModM(PMF.InvA1,-c, PMF.m1)
        C2 = PMF.MatPowModM(PMF.InvA2,-c, PMF.m2)
    end

    if e > 0
        B1 = PMF.MatTwoPowModM(PMF.A1p0, e, PMF.m1)
        B2 = PMF.MatTwoPowModM(PMF.A2p0, e, PMF.m2)
    elseif e < 0
        B1 = PMF.MatTwoPowModM(PMF.InvA1,-e, PMF.m1)
        B2 = PMF.MatTwoPowModM(PMF.InvA2,-e, PMF.m2)
    end

    if ~(e == 0)
        C1 = PMF.MatMatModM(B1, C1, PMF.m1)
        C2 = PMF.MatMatModM(B2, C2, PMF.m2)
    end

    rng.Cg[1:3] = PMF.MatVecModM(C1, view(rng.Cg, 1:3), PMF.m1)
    rng.Cg[4:6] = PMF.MatVecModM(C2, view(rng.Cg, 4:6), PMF.m2)
    return rng          # every family's navigation returns the generator
end
