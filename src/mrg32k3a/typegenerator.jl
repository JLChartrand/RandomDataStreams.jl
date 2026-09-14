# All integer types are derived from the native Float64 step: the pair (p1, p2)
# is combined exactly as for the float, but truncated to UInt16, then assembled
# into wider unsigned types by concatenating 16-bit chunks.

@inline rand(rng::MRG32k3a, ::Random.SamplerType{UInt16})::UInt16 =
    UInt64(let (p1, p2) = next_pair!(rng)
        combine(p1, p2)
    end) % UInt16

@inline rand(rng::MRG32k3a, ::Random.SamplerType{UInt8})::UInt8 = rand(rng, UInt16) % UInt8

@inline rand(rng::MRG32k3a, ::Random.SamplerType{Bool})::Bool = rand(rng, UInt16) % Bool

@inline function rand(rng::MRG32k3a, ::Type{S}) where {S <: Union{UInt32, UInt64, UInt128}}
    result = zero(S)
    for _ in 1:(sizeof(S) >> 1)
        result = (result << 16) + rand(rng, UInt16)
    end
    return result
end

# The wide unsigned types above are defined on `::Type`, which catches a direct
# `rand(rng, UInt32)` but not the `Sampler` machinery behind `rand!` and the
# array/range methods -- `rand!(rng, Vector{UInt32})` was a MethodError. Route
# those to the same implementation, so that filling an array and drawing in a
# loop give the same sequence.
#
# UInt64 used to have a second, disagreeing implementation here, assembled from
# the mantissas of two [1,2) draws. That construction assumes 52 random bits
# per double; MRG32k3a has log2(m1) = 32.0 bits of entropy per step, so the low
# mantissa bits are a near-deterministic function of the high ones. The result
# failed SmallCrush on its bit stream with six p-values at 0, and bit 12 came
# out set in two draws out of three. It is gone; `rand!` and `rand` now share
# the assembly above, at four MRG steps per UInt64 instead of two.
for S in (UInt32, UInt64, UInt128)
    @eval @inline rand(rng::MRG32k3a, ::Random.SamplerType{$S})::$S = rand(rng, $S)
end

@inline rand(rng::MRG32k3a, ::Random.SamplerType{Int8})::Int8 = reinterpret(Int8, rand(rng, UInt8))
@inline rand(rng::MRG32k3a, ::Random.SamplerType{Int16})::Int16 = reinterpret(Int16, rand(rng, UInt16))
@inline rand(rng::MRG32k3a, ::Random.SamplerType{Int32})::Int32 = reinterpret(Int32, rand(rng, UInt32))
@inline rand(rng::MRG32k3a, ::Random.SamplerType{Int64})::Int64 = reinterpret(Int64, rand(rng, UInt64))
@inline rand(rng::MRG32k3a, ::Random.SamplerType{Int128})::Int128 = reinterpret(Int128, rand(rng, UInt128))

"""
Produces a raw random number with 32 bits of precision.
"""
@inline function rand(rng::MRG32k3a)::Float64
    p1, p2 = next_pair!(rng)
    return combine(p1, p2) * PMF.norm
end

rand(rng::MRG32k3a, ::Type{Float64}) = rand(rng)

"""
Hook required by Random's generic machinery for Float64-native generators:
produces a Float64 in [1, 2). Everything else (ints, arrays, shuffle...)
derives from this without further plumbing.
"""
rand(rng::MRG32k3a, ::Random.SamplerTrivial{Random.CloseOpen12_64}) = 1.0 + rand(rng)

# No `has_fast_64`: MRG32k3a has no fast 64-bit channel. Its native step yields
# just under 32 bits, so a 64-bit word costs four steps. Advertising one only
# steered Random's machinery towards the defective construction removed above.

# MRG32k3a produces natively Float64
Random.rng_native_52(::MRG32k3a) = Float64
