# Threefry counter-based generators (Salmon, Moraes, Dror & Shaw, SC 2011).
#
# Threefry is a non-cryptographic reduction of the Threefish block cipher used
# in Skein. Unlike Philox it uses no multiplication at all: each round is
# add / rotate / xor only, so it is reproducible bit-for-bit on any
# architecture, including ones with no fast integer multiply. Salmon et al.
# found it the fastest of the family on the CPUs of 2011 and recommend 20
# rounds there; 13 already pass Crush, and 20 keeps a safety margin. On a
# current x86 the picture has moved -- see scripts/benchmarks/throughput.jl,
# where Philox4x64-10 comes out ahead of Threefry4x64-20, ten multiply-rounds
# against twenty ARX rounds.
#
# The generic counter/key/stream machinery lives in cbrng/cbrng.jl; this file
# only supplies the bijection and the concrete aliases.

# Skein key-schedule parity constants.
const THREEFRY_PARITY32 = UInt32(0x1BD11BDA)
const THREEFRY_PARITY64 = UInt64(0x1BD11BDAA9FC1A22)

@inline _threefry_parity(::Type{UInt32}) = THREEFRY_PARITY32
@inline _threefry_parity(::Type{UInt64}) = THREEFRY_PARITY64

# Skein rotation constants, one pair per round, repeating with period 8.
const THREEFRY_ROT_4x32 = ((10, 26), (11, 21), (13, 27), (23, 5),
                           ( 6, 20), (17, 11), (25, 10), (18, 20))
const THREEFRY_ROT_4x64 = ((14, 16), (52, 57), (23, 40), ( 5, 37),
                           (25, 33), (46, 12), (58, 22), (32, 32))

@inline _threefry_rot(::Type{UInt32}) = THREEFRY_ROT_4x32
@inline _threefry_rot(::Type{UInt64}) = THREEFRY_ROT_4x64

"""
Threefish key schedule: the four key words plus the parity word that makes
their sum invariant, used round-robin at each key injection.
"""
@inline function _threefry_ks(key::NTuple{4,W}) where {W}
    parity = key[1] ⊻ key[2] ⊻ key[3] ⊻ key[4] ⊻ _threefry_parity(W)
    return (key[1], key[2], key[3], key[4], parity)
end

"""
    threefry(C, K, Val(R) = Val(20)) -> NTuple{4,W}

`R` rounds of Threefry-4x over the counter block `C` under the key `K`: one
output block of `4 * sizeof(W)` bytes of random bits. Rounds alternate between
mixing `(x1, x2)`/`(x3, x4)` and `(x1, x4)`/`(x3, x2)`, with a key injection
after every fourth round.

The rounds are unrolled at compile time. Each one uses a different pair of
rotation constants, so a plain loop would index the constant table with a value
LLVM cannot fold, which costs roughly a factor of nine — Threefry is supposed
to be the *fastest* member of the family on a CPU, so the unrolling is not a
micro-optimisation here.
"""
threefry(C::NTuple{4,W}, K::NTuple{4,W}) where {W} = threefry(C, K, Val(20))

@generated function threefry(C::NTuple{4,W}, K::NTuple{4,W}, ::Val{R}) where {W,R}
    rot = _threefry_rot(W)
    body = Expr(:block,
                :(ks = _threefry_ks(K)),
                :(x1 = C[1] + ks[1]), :(x2 = C[2] + ks[2]),
                :(x3 = C[3] + ks[3]), :(x4 = C[4] + ks[4]))   # injection 0

    for r in 0:(R - 1)
        r0, r1 = rot[(r % 8) + 1]
        if iseven(r)
            push!(body.args,
                  :(x1 += x2), :(x2 = rolt(x2, $r0) ⊻ x1),
                  :(x3 += x4), :(x4 = rolt(x4, $r1) ⊻ x3))
        else
            push!(body.args,
                  :(x1 += x4), :(x4 = rolt(x4, $r0) ⊻ x1),
                  :(x3 += x2), :(x2 = rolt(x2, $r1) ⊻ x3))
        end

        if r % 4 == 3                       # key injection number s = 1, 2, ...
            s = (r + 1) ÷ 4
            push!(body.args,
                  :(x1 += ks[$((s + 0) % 5 + 1)]),
                  :(x2 += ks[$((s + 1) % 5 + 1)]),
                  :(x3 += ks[$((s + 2) % 5 + 1)]),
                  :(x4 += ks[$((s + 3) % 5 + 1)] + $(W(s))))
        end
    end

    push!(body.args, :((x1, x2, x3, x4)))
    return body
end

bijection(::Val{:threefry4x32_20}, ctr::NTuple{4,UInt32}, key::NTuple{4,UInt32}) =
    threefry(ctr, key, Val(20))

bijection(::Val{:threefry4x64_20}, ctr::NTuple{4,UInt64}, key::NTuple{4,UInt64}) =
    threefry(ctr, key, Val(20))

_variant_name(::Type{<:CBRNG{:threefry4x32_20}}) = "Threefry4x32-20"
_variant_name(::Type{<:CBRNG{:threefry4x64_20}}) = "Threefry4x64-20"

# Concrete variants -------------------------------------------------------------

"""
    Threefry4x64RNG([key, [ctr]])

Threefry4x64-20 counter-based stream, the variant Salmon et al. recommend on
CPUs. Four 64-bit key words identify the stream; substreams are `2^64` blocks
apart. Only the low 128 bits of its 256-bit counter space are used, which still
gives `2^130` draws per substream.

The round count is a parameter of the bijection rather than of the type, so a
faster 13-round variant — enough to pass Crush, per Salmon et al. — is three
lines:

```julia
RandomDataStreams.bijection(::Val{:threefry4x64_13}, ctr::NTuple{4,UInt64}, key::NTuple{4,UInt64}) =
    RandomDataStreams.threefry(ctr, key, Val(13))
RandomDataStreams._variant_name(::Type{<:RandomDataStreams.CBRNG{:threefry4x64_13}}) = "Threefry4x64-13"
const Threefry4x64_13RNG = RandomDataStreams.CBRNG{:threefry4x64_13,UInt64,4,4}
```
"""
const Threefry4x64RNG = CBRNG{:threefry4x64_20,UInt64,4,4}

"""
    Threefry4x64Gen([seed])

Hands out independent [`Threefry4x64RNG`](@ref) streams, one distinct key each.
"""
const Threefry4x64Gen = CBGen{:threefry4x64_20,UInt64,4,4}

"""
    Threefry4x32RNG([key, [ctr]])

Threefry4x32-20 counter-based stream: same construction as
[`Threefry4x64RNG`](@ref) with 32-bit words, for a 128-bit counter and a
128-bit key.
"""
const Threefry4x32RNG = CBRNG{:threefry4x32_20,UInt32,4,4}

"""
    Threefry4x32Gen([seed])

Hands out independent [`Threefry4x32RNG`](@ref) streams, one distinct key each.
"""
const Threefry4x32Gen = CBGen{:threefry4x32_20,UInt32,4,4}
