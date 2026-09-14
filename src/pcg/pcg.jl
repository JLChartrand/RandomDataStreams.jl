# ---------------------------------------------------------------------------
# PCG (O'Neill 2014): a linear congruential state transition on 128 bits,
#
#     s <- a*s + c   (mod 2^128),
#
# followed by a permutation of the state into 64 bits of output. PCG64 is the
# default bit generator of NumPy, which is the reason it is here.
#
# What this file deliberately does NOT implement is PCG's own notion of a
# "stream", the increment `c`. Every odd increment gives a full-period orbit,
# and NumPy exposes it as part of the state, but two such orbits are not known
# to be independent. Writing t_n = s_n + h and matching the two recurrences
# gives
#
#     h*(a - 1) = c1 - c2,
#
# which is solvable whenever 4 divides c1 - c2, since a = 1 (mod 4) is forced
# by the full-period condition and v2(a - 1) = 2 for the PCG multipliers. For
# half of all pairs of odd increments, then, the two "independent streams" are
# the same state sequence translated by a constant. Choosing c2 = c1 - (a - 1)
# makes that constant 1, and the two output sequences then agree to within a
# mean Hamming distance of 2 bits out of 64, against 32 for genuinely unrelated
# sequences. Most pairs are far less structured than that, but no criterion is
# published for telling the safe pairs from the bad ones — which is precisely
# the objection L'Ecuyer et al. (2021, Sec. 3) raise against heuristic stream
# assignment.
#
# So the increment is fixed to the reference value for the whole package, and
# streams come from jumping, which for an LCG is closed-form:
#
#     s_n = a^n * s_0 + c * (a^n - 1)/(a - 1)   (mod 2^128),
#
# computable in O(log n) by Brown's (1994) doubling recurrence. This is the
# cheapest arbitrary jump of any generator here: xoshiro pays O(deg^2) GF(2)
# operations for the same thing, and MRG32k3a a sequence of matrix products.
# ---------------------------------------------------------------------------

# Reference constants of the PCG C library.
const PCG_MULT_128  = (UInt128(2549297995355413924) << 64) | UInt128(4865540595714422341)
const PCG_MULT_CM   = UInt128(0xda942042e4dd58b5)          # "cheap multiplier", used by DXSM
const PCG_INCREMENT = (UInt128(6364136223846793005) << 64) | UInt128(1442695040888963407)

# Jump distances. The obvious choice, 2^64 between substreams and 2^96 between
# streams, is wrong for an LCG, and wrong in a way that the package's own
# inter-stream battery catches: by the lifting-the-exponent lemma,
#
#     v2(a^n - 1) = v2(a - 1) + v2(a + 1) + v2(n) - 1 = 2 + v2(n)   (n even),
#
# so jumping a power-of-two distance 2^m gives a jump multiplier congruent to 1
# modulo 2^(m+2). With m = 96 the 64 streams handed out by `next_stream!` have
# states agreeing in their low 98 bits up to an arithmetic progression, and
# interleaving them round-robin fails SmallCrush outright -- 14 of 15 p-values
# at 0, against a clean pass for the same generator used as a single stream.
#
# Odd distances have v2(n) = 0, so the jump multiplier is as far from the
# identity as the multiplier itself. Substreams are therefore 2^64 + 1 apart
# and streams 2^32 * (2^64 + 1) + 1, which is still 2^32 streams of 2^32
# substreams of 2^64 draws, and still non-overlapping: the last substream of a
# stream ends at 2^96 + 2^32 - 1, two below the next stream's start.
const _PCG_SHORT_JUMP = (UInt128(1) << 64) + UInt128(1)
const _PCG_LONG_JUMP  = (UInt128(1) << 32) * _PCG_SHORT_JUMP + UInt128(1)

"An integer seed becomes a 128-bit state through splitmix64, as everywhere else."
_pcg_seed_state(seed::UInt64) =
    ((w = _splitmix_words(seed, 2); (UInt128(w[1]) << 64) | UInt128(w[2])))

"Two 64-bit words are the state itself, high word first."
function _pcg_words_state(v::AbstractVector{<:Integer})
    length(v) == 2 || throw(ArgumentError("a PCG state must have 2 elements (high and low 64 bits)"))
    return (UInt128(v[1] % UInt64) << 64) | UInt128(v[2] % UInt64)
end

@inline _pcg_mult(::Val{:xsl_rr}) = PCG_MULT_128
@inline _pcg_mult(::Val{:dxsm})   = PCG_MULT_CM

@inline _pcg_step(s::UInt128, ::Val{V}) where {V} = s * _pcg_mult(Val(V)) + PCG_INCREMENT

@inline _rotr64(x::UInt64, k::Integer) = (x >> (k & 63)) | (x << ((-k) & 63))

# XSL-RR: fold the halves together, rotate by the top six bits of the state.
@inline _pcg_output(::Val{:xsl_rr}, s::UInt128) =
    _rotr64((s >> 64) % UInt64 ⊻ (s % UInt64), (s >> 122) % Int)

# DXSM: xor-shift, cheap multiply, xor-shift, then multiply by the odd low half.
@inline function _pcg_output(::Val{:dxsm}, s::UInt128)
    hi = (s >> 64) % UInt64
    lo = (s % UInt64) | one(UInt64)
    hi ⊻= hi >> 32
    hi *= PCG_MULT_CM % UInt64
    hi ⊻= hi >> 48
    return hi * lo
end

"""
    _lcg_advance(state, delta, mult) -> state

Brown's (1994) doubling recurrence: the state `delta` steps ahead, in
`O(log delta)` multiplications. `delta` is taken modulo `2^128`, so a backward
jump is the forward jump by `2^128 - n` and costs exactly the same.
"""
@inline function _lcg_advance(state::UInt128, delta::UInt128, mult::UInt128)
    acc_mult = one(UInt128)
    acc_plus = zero(UInt128)
    cur_mult = mult
    cur_plus = PCG_INCREMENT
    d = delta
    while d != zero(UInt128)
        if isodd(d)
            acc_mult *= cur_mult
            acc_plus = acc_plus * cur_mult + cur_plus
        end
        cur_plus = (cur_mult + one(UInt128)) * cur_plus
        cur_mult *= cur_mult
        d >>= 1
    end
    return acc_mult * state + acc_plus
end

# The generator ---------------------------------------------------------------

"""
    PCGRNG{V}

Permuted congruential generator on 128 bits of state, where `V` names the
output permutation. Use the aliases [`PCG64`](@ref) and [`PCG64DXSM`](@ref).

Like the other recurrence-based generators of the package it carries three
checkpoints: the current state `Cg`, the start of the current substream `Bg`
and the start of the current stream `Ig`.
"""
mutable struct PCGRNG{V} <: AbstractStreamableRNG
    Cg::UInt128   # current state
    Bg::UInt128   # start point of the current substream
    Ig::UInt128   # start point of the current stream

    PCGRNG{V}(s::UInt128) where {V} = new{V}(s, s, s)   # a UInt128 is the state itself
    PCGRNG{V}(c::UInt128, b::UInt128, i::UInt128) where {V} = new{V}(c, b, i)
end

# Seeding follows the package-wide rule: a value in the state's own
# representation -- a `UInt128`, or a two-element vector of 64-bit words -- is
# the state itself, and any other integer is a seed, expanded through
# splitmix64. Every 128-bit value is a valid LCG state, so unlike the xoshiro
# families there is no forbidden seed and nothing to check.
PCGRNG{V}() where {V} = PCGRNG{V}(12345)
PCGRNG{V}(seed::Integer) where {V} = PCGRNG{V}(_pcg_seed_state(UInt64(seed)))

PCGRNG{V}(v::AbstractVector{<:Integer}) where {V} = PCGRNG{V}(_pcg_words_state(v))

Base.copy(rng::PCGRNG{V}) where {V} = PCGRNG{V}(rng.Cg, rng.Bg, rng.Ig)

"""
Raw 64-bit output of the generator.
"""
@inline function next(rng::PCGRNG{:xsl_rr})
    rng.Cg = _pcg_step(rng.Cg, Val(:xsl_rr))       # PCG64 steps, then permutes
    return _pcg_output(Val(:xsl_rr), rng.Cg)
end

@inline function next(rng::PCGRNG{:dxsm})
    s = rng.Cg                                      # DXSM permutes the pre-step state
    rng.Cg = _pcg_step(s, Val(:dxsm))
    return _pcg_output(Val(:dxsm), s)
end

# Navigation ------------------------------------------------------------------

"""
Jumps forward by 2^64 + 1 values: the start of the next non-overlapping
substream. The distance is odd on purpose; see the implementation notes.
"""
function short_jump!(rng::PCGRNG{V}) where {V}
    rng.Cg = rng.Bg = _lcg_advance(rng.Bg, _PCG_SHORT_JUMP, _pcg_mult(Val(V)))
    return rng
end

"""
Jumps forward by 2^32 * (2^64 + 1) + 1 values: the start of the next
independent stream. The distance is odd on purpose; see the implementation
notes.
"""
function long_jump!(rng::PCGRNG{V}) where {V}
    rng.Cg = rng.Bg = rng.Ig = _lcg_advance(rng.Ig, _PCG_LONG_JUMP, _pcg_mult(Val(V)))
    return rng
end

"""
Given a random number generator, jumps n steps forward if n > 0
(or -n steps backwards if n < 0), where

if e > 0, let n = 2^e + c;
if e < 0, let n = -2^(-e) + c;
if e = 0, let n = c.

The distance is reduced modulo the period 2^128, so any magnitude is accepted
and a backward jump costs the same as a forward one. Only the current position
moves; the stream and substream boundaries are left untouched.
"""
function advance_state!(rng::PCGRNG{V}, e::Integer, c::Integer) where {V}
    n = if e == 0
        BigInt(c)
    elseif e > 0
        BigInt(2)^BigInt(e) + BigInt(c)
    else
        -BigInt(2)^BigInt(-e) + BigInt(c)
    end
    delta = UInt128(mod(n, BigInt(2)^128))
    delta == zero(UInt128) && return rng
    rng.Cg = _lcg_advance(rng.Cg, delta, _pcg_mult(Val(V)))
    return rng
end

"""
Seeds a generator, resetting the stream and substream boundaries to the seed.
"""
srand!(rng::PCGRNG, seed::UInt128) = (rng.Cg = rng.Bg = rng.Ig = seed; rng)
srand!(rng::PCGRNG, seed::Integer) = srand!(rng, _pcg_seed_state(UInt64(seed)))

srand!(rng::PCGRNG, seed::AbstractVector{<:Integer}) = srand!(rng, _pcg_words_state(seed))

"""
    reset_stream!(rng) -> rng

Rewind the generator to the very beginning of its current stream.
"""
reset_stream!(rng::PCGRNG) = (rng.Cg = rng.Bg = rng.Ig; rng)

"""
    reset_substream!(rng) -> rng

Rewind the generator to the beginning of its current substream.
"""
reset_substream!(rng::PCGRNG) = (rng.Cg = rng.Bg; rng)

"""
    next_substream!(rng) -> rng

Move the generator to the start of the next substream: an anchored jump of
2^64 + 1 values from the current substream boundary.
"""
next_substream!(rng::PCGRNG) = short_jump!(rng)

"""
    get_state(rng::PCGRNG) -> UInt128

Current state. Hand it back to [`set_state!`](@ref) to return here.
"""
get_state(rng::PCGRNG) = rng.Cg

"""
    set_state!(rng::PCGRNG, state) -> rng

Restores the current position from a `get_state(rng)` value. Only the current
position moves; the stream and substream boundaries are untouched.
"""
function set_state!(rng::PCGRNG, state)
    rng.Cg = state % UInt128
    return rng
end

# Random API ------------------------------------------------------------------

Random.rng_native_52(::PCGRNG) = UInt64

rand(rng::PCGRNG, ::Random.SamplerType{UInt64}) = next(rng)

"""
Return a random Float64 in [0, 1).
"""
rand(rng::PCGRNG) =
    reinterpret(Float64, 0x3ff0000000000000 | (next(rng) & 0x000fffffffffffff)) - 1.0

"""
Generates a `Float32` from a PCG generator.
"""
rand(rng::PCGRNG, ::Type{Float32}) = Float32(rand(rng))

"""
Generates a `Float16` from a PCG generator.
"""
rand(rng::PCGRNG, ::Type{Float16}) = Float16(rand(rng))

# Ranges are left to `Random`: the sampler it builds from the `SamplerType`
# methods below rejects instead of folding with `%`, so it is unbiased and it
# raises the same `ArgumentError` as every other generator on an empty range.

rand(rng::PCGRNG, ::Random.SamplerType{Bool}) = (next(rng) >> 63) == 1

for T in (UInt8, UInt16, UInt32)
    @eval rand(rng::PCGRNG, ::Random.SamplerType{$T}) = next(rng) % $T
end

rand(rng::PCGRNG, ::Random.SamplerType{UInt128}) =
    (UInt128(next(rng)) << 64) | UInt128(next(rng))

for (S, U) in ((Int8, UInt8), (Int16, UInt16), (Int32, UInt32), (Int64, UInt64), (Int128, UInt128))
    @eval rand(rng::PCGRNG, ::Random.SamplerType{$S}) = reinterpret($S, rand(rng, $U))
end

rand(rng::PCGRNG, ::Random.SamplerType{Char}) = Char(rand(rng, 0x0000:0xd7ff))

# Stream generator -------------------------------------------------------------

"""
    PCGGen{V} <: AbstractRNGStream

Hands out non-overlapping [`PCGRNG`](@ref) streams: each call to
`next_stream!` applies the long jump to the stored seed. Use the aliases
[`PCG64Gen`](@ref) and [`PCG64DXSMGen`](@ref).
"""
mutable struct PCGGen{V} <: AbstractRNGStream
    nextSeed::UInt128

    # explicit, so that the `Integer` method below narrows to this one instead
    # of to the catch-all default constructor and recursing forever
    PCGGen{V}(seed::UInt128) where {V} = new{V}(seed)
end

PCGGen{V}() where {V} = PCGGen{V}(12345)
PCGGen{V}(seed::Integer) where {V} = PCGGen{V}(_pcg_seed_state(UInt64(seed)))

PCGGen{V}(seed::AbstractVector{<:Integer}) where {V} = PCGGen{V}(_pcg_words_state(seed))

"""
Given an RNG generator object, returns the next RNG stream.
"""
function next_stream!(gen::PCGGen{V}) where {V}
    seed = gen.nextSeed
    gen.nextSeed = _lcg_advance(seed, _PCG_LONG_JUMP, _pcg_mult(Val(V)))
    return PCGRNG{V}(seed)
end

"""
    srand!(gen::PCGGen, seed) -> gen

Resets the seed that the next `next_stream!` call will use.
"""
srand!(gen::PCGGen, seed::UInt128) = (gen.nextSeed = seed; gen)
srand!(gen::PCGGen, seed::Integer) = srand!(gen, _pcg_seed_state(UInt64(seed)))
srand!(gen::PCGGen, seed::AbstractVector{<:Integer}) = srand!(gen, _pcg_words_state(seed))

"""
    get_state(gen::PCGGen) -> UInt128

Seed that will be used by the next `next_stream!` call.
"""
get_state(gen::PCGGen) = gen.nextSeed

"""
    set_state!(gen::PCGGen, seed) -> gen

Restores the seed of the next stream, as returned by `get_state(gen)`.
"""
set_state!(gen::PCGGen, seed) = srand!(gen, seed % UInt128)

# Names and display ------------------------------------------------------------

_variant_name(::Type{<:PCGRNG{:xsl_rr}}) = "PCG64"
_variant_name(::Type{<:PCGRNG{:dxsm}})   = "PCG64DXSM"

_hex128(x::UInt128) = "0x" * string(x, base = 16, pad = 32)

# Two-argument `show` is what interpolation, `@show` and container display call,
# so it stays on one line; the full dump belongs to `text/plain`.
show(io::IO, rng::PCGRNG) =
    print(io, _variant_name(typeof(rng)), "(Cg = ", _hex128(rng.Cg), ")")

function show(io::IO, ::MIME"text/plain", rng::PCGRNG)
    print(io, "Full state of ", _variant_name(typeof(rng)), " generator:\n",
          "Cg = $(_hex128(rng.Cg))\nBg = $(_hex128(rng.Bg))\nIg = $(_hex128(rng.Ig))")
end

show(io::IO, gen::PCGGen{V}) where {V} =
    print(io, _variant_name(PCGRNG{V}), "Gen(next = ", _hex128(gen.nextSeed), ")")

show(io::IO, ::MIME"text/plain", gen::PCGGen{V}) where {V} =
    print(io, "Seed for next ", _variant_name(PCGRNG{V}), " generator:\n",
          _hex128(gen.nextSeed))

# Concrete variants -------------------------------------------------------------

"""
    PCG64 <: AbstractStreamableRNG

PCG64 (O'Neill 2014) with the XSL-RR output permutation: a 128-bit linear
congruential state, period 2^128, 64 bits of output per draw. This is the
default bit generator of NumPy, and the outputs here match it exactly for the
same state.

Streams are 2^96 values apart and substreams 2^64, giving 2^32 streams of 2^32
substreams — the same scale as `Xoroshiro128p`, so it suits mild parallelism
rather than large simulations. Jumping to an arbitrary position costs
`O(log n)` multiplications, the cheapest of any generator in the package.

The increment is fixed: PCG's own increment-based "streams" are not used here,
because two increments can define orbits that differ by a constant offset. See
the implementation notes.
"""
const PCG64 = PCGRNG{:xsl_rr}

"""
    PCG64DXSM <: AbstractStreamableRNG

PCG64 with the DXSM output permutation and the cheap 64-bit multiplier, the
variant NumPy recommends for large-scale parallel use. Same state size, period
and stream structure as [`PCG64`](@ref), and the outputs match NumPy's
`PCG64DXSM` exactly for the same state.
"""
const PCG64DXSM = PCGRNG{:dxsm}

"""
    PCG64Gen([seed])

Hands out independent [`PCG64`](@ref) streams, 2^32 * (2^64 + 1) + 1 values apart.
"""
const PCG64Gen = PCGGen{:xsl_rr}

"""
    PCG64DXSMGen([seed])

Hands out independent [`PCG64DXSM`](@ref) streams, 2^32 * (2^64 + 1) + 1 values apart.
"""
const PCG64DXSMGen = PCGGen{:dxsm}
