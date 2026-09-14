# All integer types are derived from the native step: the pair (p1, p2) is
# combined into z exactly as for the float, then truncated to UInt32, and wider
# types are assembled by concatenating 32-bit chunks.
#
# The chunk is twice as wide as MRG32k3a's, and that is the whole point of this
# generator: z lies in [1, m1] with m1 = 2^63 - 6645, so a residue class modulo
# 2^32 holds 2^31 values give or take one, a departure from uniformity of order
# 2^-31 per chunk against 2^-16 for MRG32k3a's 16-bit chunks. A UInt64 costs
# two steps here, four there.

@inline rand(rng::MRG63k3a, ::Random.SamplerType{UInt32})::UInt32 =
    UInt64(let (p1, p2) = next_pair!(rng)
        combine63(p1, p2)
    end) % UInt32

@inline rand(rng::MRG63k3a, ::Random.SamplerType{UInt16})::UInt16 = rand(rng, UInt32) % UInt16

@inline rand(rng::MRG63k3a, ::Random.SamplerType{UInt8})::UInt8 = rand(rng, UInt32) % UInt8

@inline rand(rng::MRG63k3a, ::Random.SamplerType{Bool})::Bool = rand(rng, UInt32) % Bool

@inline function rand(rng::MRG63k3a, ::Type{S}) where {S <: Union{UInt64, UInt128}}
    result = zero(S)
    for _ in 1:(sizeof(S) >> 2)
        result = (result << 32) + rand(rng, UInt32)
    end
    return result
end

# As for MRG32k3a: the methods above are defined on `::Type`, which catches a
# direct `rand(rng, UInt64)` but not the `Sampler` machinery behind `rand!` and
# the array/range methods. Route those to the same implementation, so that
# filling an array and drawing in a loop give the same sequence.
for S in (UInt64, UInt128)
    @eval @inline rand(rng::MRG63k3a, ::Random.SamplerType{$S})::$S = rand(rng, $S)
end

@inline rand(rng::MRG63k3a, ::Random.SamplerType{Int8})::Int8 = reinterpret(Int8, rand(rng, UInt8))
@inline rand(rng::MRG63k3a, ::Random.SamplerType{Int16})::Int16 = reinterpret(Int16, rand(rng, UInt16))
@inline rand(rng::MRG63k3a, ::Random.SamplerType{Int32})::Int32 = reinterpret(Int32, rand(rng, UInt32))
@inline rand(rng::MRG63k3a, ::Random.SamplerType{Int64})::Int64 = reinterpret(Int64, rand(rng, UInt64))
@inline rand(rng::MRG63k3a, ::Random.SamplerType{Int128})::Int128 = reinterpret(Int128, rand(rng, UInt128))

"""
Produces a raw random number with 63 bits of precision, of which a Float64
keeps 53.
"""
@inline function rand(rng::MRG63k3a)::Float64
    p1, p2 = next_pair!(rng)
    return combine63(p1, p2) * PMF63.norm
end

rand(rng::MRG63k3a, ::Type{Float64}) = rand(rng)

"""
Hook required by Random's generic machinery for Float64-native generators:
produces a Float64 in [1, 2). Everything else (ints, arrays, shuffle...)
derives from this without further plumbing.
"""
rand(rng::MRG63k3a, ::Random.SamplerTrivial{Random.CloseOpen12_64}) = 1.0 + rand(rng)

# Float64, not UInt64, even though the step yields 63 bits: one step covers a
# double, and declaring UInt64 native would make every double cost the two
# steps of a word instead of one.
Random.rng_native_52(::MRG63k3a) = Float64
