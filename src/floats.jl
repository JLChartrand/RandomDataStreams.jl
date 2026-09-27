# Float32 and Float16 draws for the families whose native output is an unsigned
# word: xoshiro/xoroshiro, PCG and the counter-based generators.
#
# The value is the top precision(T) bits of the word, read as an integer and
# scaled by 2^-precision(T). Every result is exact, the grid is uniform (2^24
# points for Float32, 2^11 for Float16), and the largest value is 1 - 2^-24
# (1 - 2^-11). Rounding a Float64 draw instead, as `Float32(rand(rng))` did,
# sends every draw above 1 - 2^-25 to 1.0f0 -- one in 2^25, and one in 2^12 for
# Float16. The construction is the standard library's for its own Xoshiro, and
# NumPy's.
#
# The methods are on the sampler rather than on `::Type{Float32}`, so that
# `rand(rng, Float32)`, `rand(rng, Float32, n)` and `rand!` share one path.

@inline _u01(::Type{Float32}, u::UInt64) = Float32(u >>> 40) * Float32(0x1p-24)
@inline _u01(::Type{Float32}, u::UInt32) = Float32(u >>> 8)  * Float32(0x1p-24)
@inline _u01(::Type{Float16}, u::UInt64) = Float16(u >>> 53) * Float16(0x1p-11)
@inline _u01(::Type{Float16}, u::UInt32) = Float16(u >>> 21) * Float16(0x1p-11)
