const DEFAULT_SEED63 = PMF63.DEFAULT_SEED

"""
    checkseed63(x::Vector{Int}) -> Bool

True when `x` is a valid MRG63k3a seed: six non-negative integers, the first
three below `m1 = 2^63 - 6645` and not all zero, the last three below
`m2 = 2^63 - 21129` and not all zero.
"""
const checkseed63 = PMF63.checkseed

# The state vector runs one step ahead of the position, which is the third of
# Vigna's optimizations (https://github.com/vigna/MRG32k3a): the output of a
# draw is the pair already sitting in `Cg[3]`, `Cg[6]`, so the processor
# computes it in parallel with the next state instead of waiting for it. Worth
# 7% here, and nothing at all for MRG32k3a, whose step is short enough that the
# combination is already hidden -- both measured, see the implementation notes.
#
# The shift is invisible from outside: these two convert at the boundary, and
# every other operation is unaffected. Jumps in particular need no conversion,
# because the jump matrices commute with the one-step matrix, so jumping a
# shifted state gives the shifted jumped state. Both are cold paths.
"""
Advances a state vector by one step, seed representation -> internal.
"""
function _step63(v::AbstractVector{<:Integer})
    w = Vector{Int}(undef, 6)
    x = Int64.(collect(v))
    w[1:3] = PMF63.MatVecModM(PMF63.A1p0, view(x, 1:3), PMF63.m1)
    w[4:6] = PMF63.MatVecModM(PMF63.A2p0, view(x, 4:6), PMF63.m2)
    return w
end

"""
Steps a state vector back by one, internal -> seed representation.
"""
function _unstep63(v::AbstractVector{<:Integer})
    w = Vector{Int}(undef, 6)
    x = Int64.(collect(v))
    w[1:3] = PMF63.MatVecModM(PMF63.InvA1, view(x, 1:3), PMF63.m1)
    w[4:6] = PMF63.MatVecModM(PMF63.InvA2, view(x, 4:6), PMF63.m2)
    return w
end

"""
`MRG63k3a()` will generate an instance of `MRG63k3a` with initial seeds `DEFAULT_SEED63 = [ 12345, 12345, 12345, 12345, 12345, 12345 ]`

`MRG63k3a(x::Vector{Int})` will generate an instance of `MRG63k3a` with initial seeds `x`

`MRG63k3a(x::Vector{Int}, y::Vector{Int}, z::Vector{Int})` will generate an instance of `MRG63k3a` positioned at `x`, with the substream starting at `y` and the stream at `z`. All three are seeds, in the representation `get_state` returns; the state actually stored runs one step ahead of them (see below).

MRG63k3a is the 64-bit-arithmetic member of L'Ecuyer's (1999) combined MRG
family: same structure as `MRG32k3a`, moduli just under `2^63` instead
of `2^32`, period `≈ 2^377` instead of `2^191`. Its step yields just under 63
random bits, against just under 32 for MRG32k3a, so a machine word costs two
steps here and four there.

The state is stored one step ahead of the position, so that a draw returns a
pair the previous draw already computed and the processor overlaps the output
with the next state. That is internal: seeds, `get_state`, `set_state!` and
`show` all speak the same representation as MRG32k3a.

# Examples

```jldoctest
julia> rng = MRG63k3a();

julia> rand(rng)
0.999964376179128
```
"""
mutable struct MRG63k3a <: AbstractStreamableRNG
    Cg::Vector{Int64}  # the current state of the RNG
    Bg::Vector{Int64}  # the start point of the current substream
    Ig::Vector{Int64}  # the start point of the current stream
    function MRG63k3a(x::Vector{Int} = PMF63.DEFAULT_SEED)
        @assert(PMF63.checkseed(x))
        return new(_step63(x), _step63(x), _step63(x))
    end
    function MRG63k3a(x::Vector{Int}, y::Vector{Int}, z::Vector{Int})
        @assert(PMF63.checkseed(x))
        @assert(PMF63.checkseed(y))
        @assert(PMF63.checkseed(z))
        return new(_step63(x), _step63(y), _step63(z))
    end
end
#copy imported in RandomDataStreams.jl in src
# The three vectors are already in the internal, one-step-ahead
# representation, so they are restored through the seed constructor rather than
# handed to it -- passing them directly would shift them a second time.
function copy(m::MRG63k3a)
    c = MRG63k3a()
    c.Cg[:] = m.Cg
    c.Bg[:] = m.Bg
    c.Ig[:] = m.Ig
    return c
end

# Uniform constructor: a plain integer seed, folded into the valid MRG63k3a
# seed space, so `MRG63k3a(12345)` means what `MRG32k3a(12345)` and
# `PCG64(12345)` mean.
MRG63k3a(seed::Integer) = MRG63k3a(_mrg63_seed_words(UInt64(seed)))

"""
Advances the state of `rng` by one step and returns the raw pair `(p1, p2)`.
Shared by all `rand` methods to avoid code duplication.
"""
@inline function next_pair!(rng::MRG63k3a)
    Cg = rng.Cg
    @inbounds begin
        # The pair to return is already there, computed by the previous call:
        # the state runs one step ahead precisely so that these two loads do
        # not wait on the arithmetic below.
        p1 = Cg[3]
        p2 = Cg[6]

        # `corr1` keeps the argument of the reduction non-negative, so no
        # fix-up is needed afterwards. The products need 128 bits: the moduli
        # are just under 2^63, where MRG32k3a gets away with Float64.
        q1 = PMF63.mod_m1(widemul(PMF63.a12, Cg[2]) - widemul(PMF63.a13n, Cg[1]) + PMF63.corr1)

        Cg[1] = Cg[2]
        Cg[2] = Cg[3]
        Cg[3] = q1

        q2 = PMF63.mod_m2(widemul(PMF63.a21, Cg[6]) - widemul(PMF63.a23n, Cg[4]) + PMF63.corr2)

        Cg[4] = Cg[5]
        Cg[5] = Cg[6]
        Cg[6] = q2
    end
    return p1, p2
end

"""
Combines the two component outputs into `z` in `[1, m1]`.

Branchless: the arithmetic shift yields -1 when `p1 <= p2` and 0 otherwise, so
the modulus is added exactly when the difference is not already positive. Same
value as the reference C code's `p12 > p21 ? p12 - p21 : p12 - p21 + m1`,
without the test.
"""
@inline function combine63(p1::Int64, p2::Int64)
    r = p1 - p2
    return r - PMF63.m1 * ((r - 1) >> 63)
end

"""
Resets a given random number generator to the beginning of the current stream.
"""
function reset_stream!(rng::MRG63k3a)::MRG63k3a
    rng.Cg[:] = rng.Ig
    rng.Bg[:] = rng.Ig
    return rng
end

"""
Resets a given random number generator to the beginning of the current substream.
"""
function reset_substream!(rng::MRG63k3a)::MRG63k3a
    rng.Cg[:] = rng.Bg
    return rng
end

"""
Takes a random number generator and shifts seed to next substream, `2^150`
numbers ahead.
"""
function next_substream!(rng::MRG63k3a)::MRG63k3a
    rng.Bg[1:3] = PMF63.MatVecModM(PMF63.A1p150, view(rng.Bg, 1:3), PMF63.m1)
    rng.Bg[4:6] = PMF63.MatVecModM(PMF63.A2p150, view(rng.Bg, 4:6), PMF63.m2)
    for i = 1:6
        @inbounds rng.Cg[i] = rng.Bg[i]
    end
    return rng
end
#show imported in RandomDataStreams.jl in src
# One line for interpolation and container display; the three anchors are the
# REPL's business. Both print the seed representation, the one `get_state` and
# the constructors use.
show(io::IO, rng::MRG63k3a) = print(io, "MRG63k3a(Cg = ", _unstep63(rng.Cg), ")")

function show(io::IO, ::MIME"text/plain", rng::MRG63k3a)
    print(io, "Full state of MRG63k3a generator:\nCg = $(_unstep63(rng.Cg))" *
              "\nBg = $(_unstep63(rng.Bg))\nIg = $(_unstep63(rng.Ig))")
end

"""
    get_state(rng::MRG63k3a) -> Vector{Int}

The current position, in the same representation as a seed; restore it with
[`set_state!`](@ref). The internal vector runs one step ahead of it, so this
steps back by one -- a matrix-vector product, off any hot path.
"""
function get_state(rng::MRG63k3a)::Array{Int, 1}
    return _unstep63(rng.Cg)
end

"""
    set_state!(rng::MRG63k3a, state) -> rng

Restores the current position from a `get_state(rng)` value. Only the current
state moves; the stream and substream boundaries (`Bg`, `Ig`) are untouched.
"""
function set_state!(rng::MRG63k3a, state)
    s = Int.(collect(state))
    PMF63.checkseed(s) || throw(ArgumentError("invalid MRG63k3a state"))
    rng.Cg[:] = _step63(s)
    return rng
end

"""
    srand!(rng::MRG63k3a, seed) -> rng

Reseeds the generator, resetting the stream and substream boundaries to `seed`.
"""
srand!(rng::MRG63k3a, seed) = Random.seed!(rng, seed)

"""
Given a random number generator, jumps n steps forward if n > 0
(or -n steps backwards if n < 0), where

if e > 0, let n = 2^e + c;
if e < 0, let n = -2^(-e) + c;
if e = 0, let n = c.

# Examples

```jldoctest
julia> rng = MRG63k3a();

julia> advance_state!(rng, Int64(2), Int64(-1));   # skip n = 2^2 - 1 = 3 values

julia> rand(rng)
0.8707612110911583
```
"""
function advance_state!(rng::MRG63k3a, e::Integer, c::Integer)
    e = Int64(e)
    c = Int64(c)
    if c >= 0
        C1 = PMF63.MatPowModM(PMF63.A1p0, c, PMF63.m1)
        C2 = PMF63.MatPowModM(PMF63.A2p0, c, PMF63.m2)
    else
        C1 = PMF63.MatPowModM(PMF63.InvA1, -c, PMF63.m1)
        C2 = PMF63.MatPowModM(PMF63.InvA2, -c, PMF63.m2)
    end

    if e > 0
        B1 = PMF63.MatTwoPowModM(PMF63.A1p0, e, PMF63.m1)
        B2 = PMF63.MatTwoPowModM(PMF63.A2p0, e, PMF63.m2)
    elseif e < 0
        B1 = PMF63.MatTwoPowModM(PMF63.InvA1, -e, PMF63.m1)
        B2 = PMF63.MatTwoPowModM(PMF63.InvA2, -e, PMF63.m2)
    end

    if ~(e == 0)
        C1 = PMF63.MatMatModM(B1, C1, PMF63.m1)
        C2 = PMF63.MatMatModM(B2, C2, PMF63.m2)
    end

    rng.Cg[1:3] = PMF63.MatVecModM(C1, view(rng.Cg, 1:3), PMF63.m1)
    rng.Cg[4:6] = PMF63.MatVecModM(C2, view(rng.Cg, 4:6), PMF63.m2)
    return rng          # every family's navigation returns the generator
end
