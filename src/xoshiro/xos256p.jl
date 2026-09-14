
using Random

import Base.rand

# ---------------------------------------------------------------------------
# Families of xoshiro/xoroshiro generators (Blackman & Vigna).
#
# All variants of a given family share the same linear state transition;
# only the output scrambler differs (`+`, `**`, `++`). The same holds for the
# jump constants, which depend only on the transition.
# ---------------------------------------------------------------------------

"Rotate `x` left by `k` bits."
@inline rolt(x::W, k::Integer) where {W<:Unsigned} =
    (x << k) | (x >>> (8 * sizeof(W) - k))

# Linear state transitions -----------------------------------------------------

@inline function _lin_step(s::NTuple{2,UInt64})        # xoroshiro128
    t = xor(s[2], s[1])
    return (xor(xor(rolt(s[1], 24), t), t << 16), rolt(t, 37))
end

@inline function _lin_step(s::NTuple{4,UInt64})        # xoshiro256
    t = s[2] << 17
    s3 = xor(s[3], s[1])
    s4 = xor(s[4], s[2])
    s2 = xor(s[2], s3)
    s1 = xor(s[1], s4)
    s3 = xor(s3, t)
    s4 = rolt(s4, 45)
    return (s1, s2, s3, s4)
end

@inline function _lin_step(s::NTuple{8,UInt64})        # xoshiro512
    t = s[2] << 11
    o1, o2, o3, o4, o5, o6, o7, o8 = s
    return (
        xor(o1, o7),
        xor(o2, xor(o3, o1)),
        xor(o3, o1),
        xor(o4, o5),
        xor(o5, xor(o6, o2)),
        xor(o6, o2),
        xor(xor(o7, xor(o8, o4)), t),
        rolt(xor(o8, o4), 21),
    )
end

# Output scramblers --------------------------------------------------------------

@inline _scramble(::Val{:plus}, s::NTuple{2,UInt64}) = s[1] + s[2]
@inline _scramble(::Val{:plus}, s::NTuple{4,UInt64}) = s[1] + s[4]
@inline _scramble(::Val{:plus}, s::NTuple{8,UInt64}) = s[1] + s[3]

@inline function _scramble(::Val{:starstar}, s::NTuple{N,UInt64}) where {N}
    rolt(s[2] * UInt64(5), 7) * UInt64(9)
end

@inline _scramble(::Val{:plusplus}, s::NTuple{2,UInt64}) = rolt(s[1] + s[2], 17) + s[1]
@inline _scramble(::Val{:plusplus}, s::NTuple{4,UInt64}) = rolt(s[1] + s[4], 23) + s[1]
@inline _scramble(::Val{:plusplus}, s::NTuple{8,UInt64}) = rolt(s[1] + s[3], 17) + s[3]

# Jump constants (per family; valid for every scrambler of that family) -----------

const _JUMP_128 = (
    0xdf900294d8f554a5, 0x170865df4b3201fc)             # 2^64 calls
const _LONG_JUMP_128 = (
    0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1)             # 2^96 calls

const _JUMP_256 = (
    0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
    0xa9582618e03fc9aa, 0x39abdc4529b1661c)             # 2^128 calls
const _LONG_JUMP_256 = (
    0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
    0x77710069854ee241, 0x39109bb02acbe635)             # 2^192 calls

const _JUMP_512 = (
    0x33ed89b6e7a353f9, 0x760083d7955323be, 0x2837f2fbb5f22fae,
    0x4b8c5674d309511c, 0xb11ac47a7ba28c25, 0xf1be7667092bcc1c,
    0x53851efdb6df0aaf, 0x1ebbc8b23eaf25db)             # 2^256 calls
const _LONG_JUMP_512 = (
    0x11467fef8f921d28, 0xa2a819f2e79c8ea8, 0xa8299fc284b3959a,
    0xb4d347340ca63ee1, 0x1cb0940bedbff6ce, 0xd956c5c4fa1f8e17,
    0x915e38fd4eda93bc, 0x5b3ccdfa5d7daca5)             # 2^384 calls

@inline _short_jump_consts(::Val{2}) = _JUMP_128
@inline _short_jump_consts(::Val{4}) = _JUMP_256
@inline _short_jump_consts(::Val{8}) = _JUMP_512
@inline _long_jump_consts(::Val{2}) = _LONG_JUMP_128
@inline _long_jump_consts(::Val{4}) = _LONG_JUMP_256
@inline _long_jump_consts(::Val{8}) = _LONG_JUMP_512

@inline function _jump(jump::NTuple{N,UInt64}, start::NTuple{N,UInt64}) where {N}
    s = start
    acc = ntuple(_ -> UInt64(0), Val(N))
    for j in jump
        for b in 0:63
            if (j & (UInt64(1) << b)) != 0
                acc = acc .⊻ s
            end
            s = _lin_step(s)
        end
    end
    return acc
end

# Arbitrary (forward and backward) jumps via GF(2) polynomials ----------------------
#
# The state transition is multiplication by x in GF(2)[x]/p(x) where p is the
# characteristic polynomial of the family (Vigna's published constants).
# Jumping by n steps applies x^n mod p to the state; backward jumps use the
# order of x, ord = 2^deg - 1:  x^(-n) = x^(ord - n).

const _CHARPOLY_128 = (
    0x095b8f76579aa001, 0x0008828e513b43d5)
const _CHARPOLY_256 = (
    0x9d116f2bb0f0f001, 0x0280002bcefd1a5e,
    0x04b4edcf26259f85, 0x0003c03c3f3ecb19)
const _CHARPOLY_512 = (
    0xcf3cff0c00000001, 0x7fdc78d886f00c63, 0xf05e63fca6d7b781,
    0x7a67058e7bbab6f0, 0xf11eef832e32518f, 0x51ba7c47edc758ad,
    0x8f2d27268ce4b20b, 0x0000500055d8b77f)

_charpoly(::Val{2}) = _CHARPOLY_128
_charpoly(::Val{4}) = _CHARPOLY_256
_charpoly(::Val{8}) = _CHARPOLY_512

# polynomial of degree < deg as a bit vector (coefficient of x^k at index k+1)
_poly(words::NTuple{Nw,UInt64}, deg::Int) where {Nw} =
    BitVector(((words[w] >> b) & 1) == 1 for w in 1:Nw for b in 0:63)[1:deg]

"Carry-less product `a * b mod p` (both reduced, degree < deg)."
function _poly_mul_mod(a::BitVector, b::BitVector, p::BitVector, deg::Int)
    acc = falses(deg)
    t = copy(a)
    for i in 1:deg
        if b[i]
            acc .⊻= t
        end
        top = t[deg]                     # t *= x ; x^deg folds back onto p
        t2 = falses(deg)
        t2[2:deg] .= view(t, 1:deg-1)
        if top
            t2 .⊻= p
        end
        t = t2
    end
    return acc
end

"x^n mod p by binary exponentiation (`n` may be arbitrarily large)."
function _poly_pow_x(n::BigInt, p::NTuple{Nw,UInt64}, deg::Int) where {Nw}
    pv = _poly(p, deg)
    result = falses(deg); result[1] = true          # 1
    base = falses(deg); base[2] = true              # x
    e = n
    while e > 0
        if e & 1 == 1
            result = _poly_mul_mod(result, base, pv, deg)
        end
        e >>= 1
        e == 0 && break
        base = _poly_mul_mod(base, base, pv, deg)
    end
    return result
end

_poly_to_ntuple(v::BitVector, ::Val{N}) where {N} =
    NTuple{N,UInt64}(sum(UInt64(v[(w-1)*64 + b + 1]) << b for b in 0:63; init=UInt64(0)) for w in 1:N)

# Generic RNG type -----------------------------------------------------------------

"""
    LinRNG{N,S} <: AbstractStreamableRNG

Generic xoshiro/xoroshiro generator: state of `N` 64-bit words and scrambler
`S ∈ (:plus, :starstar, :plusplus)`. Holds three checkpoints — current state
(`Cg`), substream start (`Bg`) and stream start (`Ig`) — enabling
`reset_stream!`, `reset_substream!` and anchored jumps. Use the exported
aliases (`Xoroshiro128p`, ..., `Xoshiro512pp`) instead.
"""
mutable struct LinRNG{N,S} <: AbstractStreamableRNG
    Cg::NTuple{N,UInt64}   # current state
    Bg::NTuple{N,UInt64}   # start point of the current substream
    Ig::NTuple{N,UInt64}   # start point of the current stream

    function LinRNG{N,S}(x::NTuple{N,UInt64}) where {N,S}
        checkseed(x) || throw(ArgumentError(_ZERO_STATE))
        return new{N,S}(x, x, x)
    end
    function LinRNG{N,S}(x::NTuple{N,UInt64}, y::NTuple{N,UInt64}, z::NTuple{N,UInt64}) where {N,S}
        (checkseed(x) && checkseed(y) && checkseed(z)) || throw(ArgumentError(_ZERO_STATE))
        return new{N,S}(x, y, z)
    end
end

const _ZERO_STATE = "the all-zero state is a fixed point of the xoshiro " *
                    "transition: the generator would output zeros forever"

"""
    checkseed(s::NTuple{N,UInt64}) -> Bool
    checkseed(v::AbstractVector{<:Unsigned}) -> Bool

`true` unless the words are all zero. The all-zero state maps to itself under
every xoshiro/xoroshiro transition, so it is the one state the families forbid;
there is no other constraint. (The `Vector{Int}` method is MRG32k3a's, whose
seed space is constrained differently.)
"""
checkseed(s::NTuple{N,UInt64}) where {N} = !all(iszero, s)
checkseed(v::AbstractVector{<:Unsigned}) = !all(iszero, v)

LinRNG{N,S}(v::Vector{<:Unsigned}) where {N,S} = LinRNG{N,S}(NTuple{N,UInt64}(v))

# Uniform constructors: every family accepts no argument (the package default
# seed, 12345) and a plain integer, expanded through splitmix64 because a
# xoshiro state must be well mixed and must not be all zero.
LinRNG{N,S}() where {N,S} = LinRNG{N,S}(12345)
LinRNG{N,S}(seed::Integer) where {N,S} =
    LinRNG{N,S}(NTuple{N,UInt64}(_seed_words_nonzero(UInt64(seed), N)))
LinRNG{N,S}(u::Vector{<:Unsigned}, v::Vector{<:Unsigned}, w::Vector{<:Unsigned}) where {N,S} =
    LinRNG{N,S}(NTuple{N,UInt64}(u), NTuple{N,UInt64}(v), NTuple{N,UInt64}(w))

Base.copy(rng::LinRNG{N,S}) where {N,S} = LinRNG{N,S}(rng.Cg, rng.Bg, rng.Ig)

"""
Raw 64-bit output of the generator.
"""
@inline function next(rng::LinRNG{N,S}) where {N,S}
    result = _scramble(Val(S), rng.Cg)
    rng.Cg = _lin_step(rng.Cg)
    return result
end

"""
Jumps forward by 2^96 values (xoroshiro128), 2^128 (xoshiro256) or 2^256
(xoshiro512): the start of the next non-overlapping substream.
"""
function short_jump!(rng::LinRNG{N}) where {N}
    rng.Cg = rng.Bg = _jump(_short_jump_consts(Val(N)), rng.Bg)
    return rng
end

"""
Jumps forward by 2^96 (xoroshiro128), 2^192 (xoshiro256) or 2^384 (xoshiro512)
values: the start of the next independent stream.
"""
function long_jump!(rng::LinRNG{N}) where {N}
    rng.Cg = rng.Bg = rng.Ig = _jump(_long_jump_consts(Val(N)), rng.Ig)
    return rng
end

# Deprecated pre-1.0 names (they mutate their argument despite lacking `!`).

@deprecate short_jump(rng::LinRNG) short_jump!(rng)
@deprecate long_jump(rng::LinRNG) long_jump!(rng)
@deprecate srand(rng::LinRNG, seed::Vector{UInt64}) srand!(rng, seed)

"""
Given a random number generator, jumps n steps forward if n > 0
(or -n steps backwards if n < 0), where

if e > 0, let n = 2^e + c;
if e < 0, let n = -2^(-e) + c;
if e = 0, let n = c.

The distance is reduced modulo the period (2^(64N) - 1), so any magnitude is
accepted. Only the current position moves; the stream/substream boundaries
(`Bg`, `Ig`) are left untouched. Costs O((64N)^2) GF(2) operations.
"""
function advance_state!(rng::LinRNG{N}, e::Integer, c::Integer) where {N}
    deg = 64 * N
    n = if e == 0
        BigInt(c)
    elseif e > 0
        BigInt(2)^BigInt(e) + BigInt(c)
    else
        -BigInt(2)^BigInt(-e) + BigInt(c)
    end
    ord = BigInt(2)^deg - 1              # multiplicative order of x mod p
    m = n % ord                          # signed, |m| < ord
    m == 0 && return rng
    exponent = m > 0 ? m : ord + m       # backward: x^(-m) = x^(ord - |m|)
    poly = _poly_to_ntuple(_poly_pow_x(exponent, _charpoly(Val(N)), deg), Val(N))
    rng.Cg = _jump(poly, rng.Cg)
    return rng
end

"""
Seeds a generator with the first words of `seed`.
"""
function srand!(rng::LinRNG{N}, seed::Vector{UInt64}) where {N}
    s = NTuple{N,UInt64}(seed)
    checkseed(s) || throw(ArgumentError(_ZERO_STATE))
    rng.Cg = rng.Bg = rng.Ig = s
    return rng
end

"""
    reset_stream!(rng) -> rng

Rewind the generator to the very beginning of its current stream
(`Cg = Bg = Ig`).
"""
reset_stream!(rng::LinRNG) = (rng.Cg = rng.Bg = rng.Ig; rng)

"""
    reset_substream!(rng) -> rng

Rewind the generator to the beginning of its current substream (`Cg = Bg`).
"""
reset_substream!(rng::LinRNG) = (rng.Cg = rng.Bg; rng)

"""
    next_substream!(rng) -> rng

Move the generator to the start of the next substream (anchored jump from
`Bg`): 2^64 values for xoroshiro128, 2^128 for xoshiro256, 2^256 for
xoshiro512.
"""
next_substream!(rng::LinRNG) = short_jump!(rng)

"""
    get_state(rng::LinRNG) -> Vector{UInt64}

Copy of the current state (`Cg`); restore it into a fresh generator via its
three-argument constructor.
"""
get_state(rng::LinRNG) = collect(rng.Cg)

"""
    set_state!(rng::LinRNG, state) -> rng

Restores the current position from a `get_state(rng)` value. Only the current
position moves; the stream and substream boundaries (`Bg`, `Ig`) are untouched.
"""
function set_state!(rng::LinRNG{N,S}, state) where {N,S}
    st = NTuple{N,UInt64}(state)
    checkseed(st) || throw(ArgumentError(_ZERO_STATE))
    rng.Cg = st
    return rng
end

Random.rng_native_52(::LinRNG) = UInt64

rand(rng::LinRNG, ::Random.SamplerType{UInt64}) = next(rng)

"""
Return a random Float64 in [0, 1).
"""
rand(rng::LinRNG) = next(rng) / (UInt64(0) - 1)

# Generic stream generator ---------------------------------------------------------

"""
    LinGen{N,S} <: AbstractRNGStream

Stream generator producing non-overlapping streams of `LinRNG{N,S}`: each call
to `next_stream!` applies the family's long jump to the stored seed. Use the
exported aliases (`Xoroshiro128pGen`, ..., `Xoshiro512ppGen`) instead.
"""
mutable struct LinGen{N,S} <: AbstractRNGStream
    nextSeed::Vector{UInt64}

    function LinGen{N,S}(x::Vector{UInt64}) where {N,S}
        length(x) == N || throw(ArgumentError("seed must have $N UInt64 elements"))
        checkseed(x) || throw(ArgumentError(_ZERO_STATE))
        new{N,S}(copy(x))
    end
    function LinGen{N,S}(x::Vector{T}) where {N, S, T <: Integer}
        #if the user something else than UInt64, we convert it to a required type.
        #float are not implemented right now, could be done in the future.
        newx = mod.(x, UInt64)
        length(x) == N || throw(ArgumentError("seed must have $N UInt64 elements"))
        checkseed(newx) || throw(ArgumentError(_ZERO_STATE))
        new{N,S}(copy(newx))
    end
end

LinGen{N,S}() where {N,S} = LinGen{N,S}(12345)
LinGen{N,S}(seed::Integer) where {N,S} = LinGen{N,S}(_seed_words_nonzero(UInt64(seed), N))

function _variant_name(::Type{<:LinRNG{2,:plus}});      "Xoroshiro128plus";   end
function _variant_name(::Type{<:LinRNG{2,:starstar}});  "Xoroshiro128starstar"; end
function _variant_name(::Type{<:LinRNG{2,:plusplus}});  "Xoroshiro128plusplus"; end
function _variant_name(::Type{<:LinRNG{4,:plus}});      "Xoshiro256plus";     end
function _variant_name(::Type{<:LinRNG{4,:starstar}});  "Xoshiro256starstar"; end
function _variant_name(::Type{<:LinRNG{4,:plusplus}});  "Xoshiro256plusplus"; end
function _variant_name(::Type{<:LinRNG{8,:plus}});      "Xoshiro512plus";     end
function _variant_name(::Type{<:LinRNG{8,:starstar}});  "Xoshiro512starstar"; end
function _variant_name(::Type{<:LinRNG{8,:plusplus}});  "Xoshiro512plusplus"; end

# Two-argument `show` is what interpolation, `@show` and container display call,
# so it stays on one line; the full dump belongs to `text/plain`.
show(io::IO, rng::LinRNG{N,S}) where {N,S} =
    print(io, _variant_name(typeof(rng)), "(Cg = ", collect(rng.Cg), ")")

function show(io::IO, ::MIME"text/plain", rng::LinRNG{N,S}) where {N,S}
    print(io, "Full state of ", _variant_name(typeof(rng)), " generator:\n",
          "Cg = $(collect(rng.Cg))\nBg = $(collect(rng.Bg))\nIg = $(collect(rng.Ig))")
end

show(io::IO, gen::LinGen{N,S}) where {N,S} =
    print(io, _variant_name(LinRNG{N,S}), "Gen(next = ", gen.nextSeed, ")")

function show(io::IO, ::MIME"text/plain", gen::LinGen{N,S}) where {N,S}
    print(io, "Seed for next ", _variant_name(LinRNG{N,S}), " generator:\n$(gen.nextSeed)")
end

"""
    srand!(gen, seed::Vector{UInt64}) -> gen

Resets the seed that the next `next_stream!` call will use.
"""
srand!(gen::LinGen{N,S}, seed::Vector{UInt64}) where {N,S} =
    (gen.nextSeed .= seed[1:N]; gen)

"""
    get_state(gen) -> Vector{UInt64}

Seed that will be used by the next `next_stream!` call.
"""
get_state(gen::LinGen) = copy(gen.nextSeed)

"""
    set_state!(gen::LinGen, seed) -> gen

Restores the seed of the next stream, as returned by `get_state(gen)`.
"""
set_state!(gen::LinGen, seed) = srand!(gen, collect(UInt64.(seed)))

"""
Given an RNG generator object, returns the next RNG stream.
"""
function next_stream!(gen::LinGen{N,S}) where {N,S}
    seed = NTuple{N,UInt64}(gen.nextSeed)
    gen.nextSeed .= collect(_jump(_long_jump_consts(Val(N)), seed))
    return LinRNG{N,S}(seed)
end

# Concrete variants -------------------------------------------------------------------
# `S` selects the scrambler: :plus (`+`), :starstar (`**`) or :plusplus (`++`).

"""
    Xoroshiro128p <: AbstractStreamableRNG

xoroshiro128+ (Blackman & Vigna): 128-bit state, period 2^128 - 1, output
`s0 + s1`. Fastest small-state generator for floating-point use; the four
lower bits have low linear complexity and it has a very mild Hamming-weight
dependency after ~5 TB of output. Suitable only for mild parallelism
(2^32 streams x 2^64 substreams).
"""
const Xoroshiro128p   = LinRNG{2,:plus}

"""
    Xoroshiro128ss <: AbstractStreamableRNG

xoroshiro128** (Blackman & Vigna): 128-bit state, period 2^128 - 1,
all-purpose scrambler (`rotl(s1*5,7)*9`), 1-dimensionally equidistributed.
Suitable only for mild parallelism (2^32 streams x 2^64 substreams).
"""
const Xoroshiro128ss  = LinRNG{2,:starstar}

"""
    Xoroshiro128pp <: AbstractStreamableRNG

xoroshiro128++ (Blackman & Vigna): 128-bit state, period 2^128 - 1,
all-purpose scrambler (`rotl(s0+s1,17)+s0`). Same parallelism limits as the
other 128-bit variants.
"""
const Xoroshiro128pp  = LinRNG{2,:plusplus}

"""
    Xoshiro256p <: AbstractStreamableRNG

xoshiro256+ (Blackman & Vigna): 256-bit state, period 2^256 - 1, output
`s0 + s3`. Slightly faster than `**`/`++` when generating only floating-point
numbers from the upper bits; avoid for raw 64-bit consumption.
"""
const Xoshiro256p     = LinRNG{4,:plus}

"""
    Xoshiro256ss <: AbstractStreamableRNG

xoshiro256** (Blackman & Vigna): 256-bit state, period 2^256 - 1, all-purpose,
3-dimensionally equidistributed (default PRNG of Lua and .NET 6).
"""
const Xoshiro256ss    = LinRNG{4,:starstar}

"""
    Xoshiro256pp <: AbstractStreamableRNG

xoshiro256++ (Blackman & Vigna): 256-bit state, period 2^256 - 1, all-purpose,
3-dimensionally equidistributed (default PRNG of Julia and Rust's SmallRng).
The recommended general-purpose variant.

# Examples

```jldoctest
julia> rng = Xoshiro256pp(fill(UInt64(42), 4));

julia> rand(rng, UInt64)
0x000000002a00002a
```
"""
const Xoshiro256pp    = LinRNG{4,:plusplus}

"""
    Xoshiro512p <: AbstractStreamableRNG

xoshiro512+ (Blackman & Vigna): 512-bit state, period 2^512 - 1, output
`s0 + s2`. Floating-point oriented; same caveats as `Xoshiro256p`.
"""
const Xoshiro512p     = LinRNG{8,:plus}

"""
    Xoshiro512ss <: AbstractStreamableRNG

xoshiro512** (Blackman & Vigna): 512-bit state, period 2^512 - 1, all-purpose,
7-dimensionally equidistributed.
"""
const Xoshiro512ss    = LinRNG{8,:starstar}

"""
    Xoshiro512pp <: AbstractStreamableRNG

xoshiro512++ (Blackman & Vigna): 512-bit state, period 2^512 - 1, all-purpose.
Offers 2^256 substreams of 2^256 values — more than any application needs;
prefer `Xoshiro256pp` unless you have a specific reason.
"""
const Xoshiro512pp    = LinRNG{8,:plusplus}

"""
    Xoroshiro128pGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoroshiro128p` streams via a
long jump (2^96 values) per call. Seeds are 2 `UInt64` words.
"""
const Xoroshiro128pGen  = LinGen{2,:plus}

"""
    Xoroshiro128ssGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoroshiro128ss` streams via a
long jump (2^96 values) per call. Seeds are 2 `UInt64` words.
"""
const Xoroshiro128ssGen = LinGen{2,:starstar}

"""
    Xoroshiro128ppGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoroshiro128pp` streams via a
long jump (2^96 values) per call. Seeds are 2 `UInt64` words.
"""
const Xoroshiro128ppGen = LinGen{2,:plusplus}

"""
    Xoshiro256plusGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro256p` streams via a long
jump (2^192 values) per call. Seeds are 4 `UInt64` words.
"""
const Xoshiro256plusGen = LinGen{4,:plus}

"""
    Xoshiro256pGen <: AbstractRNGStream

Regular spelling of [`Xoshiro256plusGen`](@ref), matching `Xoshiro256ssGen`,
`Xoshiro512pGen` and the rest. The two names are the same type.
"""
const Xoshiro256pGen = Xoshiro256plusGen

"""
    Xoshiro256ssGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro256ss` streams via a long
jump (2^192 values) per call. Seeds are 4 `UInt64` words.
"""
const Xoshiro256ssGen   = LinGen{4,:starstar}

"""
    Xoshiro256ppGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro256pp` streams via a long
jump (2^192 values) per call. Seeds are 4 `UInt64` words.
"""
const Xoshiro256ppGen   = LinGen{4,:plusplus}

"""
    Xoshiro512pGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro512p` streams via a long
jump (2^384 values) per call. Seeds are 8 `UInt64` words.
"""
const Xoshiro512pGen    = LinGen{8,:plus}

"""
    Xoshiro512ssGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro512ss` streams via a long
jump (2^384 values) per call. Seeds are 8 `UInt64` words.
"""
const Xoshiro512ssGen   = LinGen{8,:starstar}

"""
    Xoshiro512ppGen <: AbstractRNGStream

Stream generator minting non-overlapping `Xoshiro512pp` streams via a long
jump (2^384 values) per call. Seeds are 8 `UInt64` words.
"""
const Xoshiro512ppGen   = LinGen{8,:plusplus}

@deprecate srand(gen::LinGen, seed::Vector{UInt64}) srand!(gen, seed)
