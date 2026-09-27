"""
Generates a `Float32` in [0, 1) from any xoshiro/xoroshiro generator: the top
24 bits of one output, scaled by 2^-24. The top bits are the ones the `+`
scramblers leave strongest.
"""
rand(rng::LinRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float32}}) =
    _u01(Float32, next(rng))

"""
Generates a `Float16` in [0, 1) from any xoshiro/xoroshiro generator: the top
11 bits of one output, scaled by 2^-11.
"""
rand(rng::LinRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float16}}) =
    _u01(Float16, next(rng))

# Full Random-API coverage for integer and character types, mirroring the
# derivation used by the standard library's Xoshiro. Ranges are left to
# `Random`: the sampler it builds from these methods rejects instead of folding
# with `%`, so it is unbiased and it raises the same `ArgumentError` as every
# other generator on an empty range.

rand(rng::LinRNG, ::Random.SamplerType{Bool}) = (next(rng) >> 63) == 1

for T in (UInt8, UInt16, UInt32)
    @eval rand(rng::LinRNG, ::Random.SamplerType{$T}) = next(rng) % $T
end

rand(rng::LinRNG, ::Random.SamplerType{UInt128}) =
    (UInt128(next(rng)) << 64) | UInt128(next(rng))

for (S, U) in ((Int8, UInt8), (Int16, UInt16), (Int32, UInt32), (Int64, UInt64), (Int128, UInt128))
    @eval rand(rng::LinRNG, ::Random.SamplerType{$S}) = reinterpret($S, rand(rng, $U))
end

rand(rng::LinRNG, ::Random.SamplerType{Char}) = Char(rand(rng, 0x0000:0xd7ff))
