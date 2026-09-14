# Philox counter-based generators (Salmon, Moraes, Dror & Shaw, SC 2011).
#
# Philox is an S-box-free Feistel-like network: each round multiplies half the
# words by a fixed constant, keeping both halves of the product, and XORs the
# high halves into the other words together with the round key. The round key
# is a Weyl sequence, bumped by a fixed odd constant between rounds.
#
# The generic counter/key/stream machinery lives in cbrng/cbrng.jl; this file
# only supplies the bijection and the concrete aliases.

# Multipliers and Weyl steps (Random123 reference implementation).
const PHILOX_M4x32_0 = UInt32(0xD2511F53)
const PHILOX_M4x32_1 = UInt32(0xCD9E8D57)
const PHILOX_W32_0   = UInt32(0x9E3779B9)   # odd part of phi   * 2^32
const PHILOX_W32_1   = UInt32(0xBB67AE85)   # odd part of sqrt3 * 2^32

const PHILOX_M4x64_0 = UInt64(0xD2E7470EE14C6C93)
const PHILOX_M4x64_1 = UInt64(0xCA5A826395121157)
const PHILOX_W64_0   = UInt64(0x9E3779B97F4A7C15)
const PHILOX_W64_1   = UInt64(0xBB67AE8584CAA73B)

@inline _philox_mult(::Type{UInt32}) = (PHILOX_M4x32_0, PHILOX_M4x32_1)
@inline _philox_mult(::Type{UInt64}) = (PHILOX_M4x64_0, PHILOX_M4x64_1)
@inline _philox_weyl(::Type{UInt32}) = (PHILOX_W32_0, PHILOX_W32_1)
@inline _philox_weyl(::Type{UInt64}) = (PHILOX_W64_0, PHILOX_W64_1)

"""
One Philox-4x round: a bijection on `(x1, x2, x3, x4)` under the current round
key, returned together with the key bumped for the next round.
"""
@inline function philox_round(x::NTuple{4,W}, k::NTuple{2,W}) where {W<:Union{UInt32,UInt64}}
    mult0, mult1 = _philox_mult(W)
    weyl0, weyl1 = _philox_weyl(W)
    half = 8 * sizeof(W)

    m0 = widemul(x[1], mult0)
    m1 = widemul(x[3], mult1)
    hi0, lo0 = (m0 >>> half) % W, m0 % W
    hi1, lo1 = (m1 >>> half) % W, m1 % W

    y = (hi1 ⊻ x[2] ⊻ k[1], lo1, hi0 ⊻ x[4] ⊻ k[2], lo0)
    return y, (k[1] + weyl0, k[2] + weyl1)
end

"""
    philox(C, K, Val(R) = Val(10)) -> NTuple{4,W}

`R` rounds of Philox-4x over the counter block `C` under the key `K`: one
output block of `4 * sizeof(W)` bytes of random bits.
"""
function philox(C::NTuple{4,W}, K::NTuple{2,W}, ::Val{R} = Val(10)) where {W,R}
    x, k = C, K
    for _ in 1:R
        x, k = philox_round(x, k)
    end
    return x
end

bijection(::Val{:philox4x32_10}, ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}) =
    philox(ctr, key, Val(10))

bijection(::Val{:philox4x64_10}, ctr::NTuple{4,UInt64}, key::NTuple{2,UInt64}) =
    philox(ctr, key, Val(10))

_variant_name(::Type{<:CBRNG{:philox4x32_10}}) = "Philox4x32-10"
_variant_name(::Type{<:CBRNG{:philox4x64_10}}) = "Philox4x64-10"

# Concrete variants -------------------------------------------------------------

"""
    PhiloxRNG([key, [ctr]])

Philox4x32-10 counter-based stream. The key identifies the stream, the 128-bit
counter its position; substreams are `2^64` blocks (`2^66` draws) apart.
"""
const PhiloxRNG = CBRNG{:philox4x32_10,UInt32,4,2}

"""
    PhiloxGen([key])

Hands out independent [`PhiloxRNG`](@ref) streams, one distinct key each.
"""
const PhiloxGen = CBGen{:philox4x32_10,UInt32,4,2}

"""
    Philox4x64RNG([key, [ctr]])

Philox4x64-10 counter-based stream: same construction as [`PhiloxRNG`](@ref)
with 64-bit words, so a block holds four `UInt64` and `rand(rng, UInt64)` costs
no bit assembly. Only the low 128 bits of its 256-bit counter space are used.
"""
const Philox4x64RNG = CBRNG{:philox4x64_10,UInt64,4,2}

"""
    Philox4x64Gen([key])

Hands out independent [`Philox4x64RNG`](@ref) streams, one distinct key each.
"""
const Philox4x64Gen = CBGen{:philox4x64_10,UInt64,4,2}
