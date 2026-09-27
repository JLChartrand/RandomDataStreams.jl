# Floating-point draws. Every generator returns values in the open interval
# (0, 1) -- never 0 and never 1 -- in every floating-point type, so that
# inversion (a quantile function, -log(u), Phi^-1(u)) is finite for every draw.
#
# From an unsigned word, the value is an odd multiple of 2^-p, p = precision(T):
# the top p-1 bits of the word, read as an integer k, give (2k + 1) * 2^-p. Every
# result is exact, the draws are the midpoints of 2^(p-1) equal cells, the
# extremes are 2^-p and 1 - 2^-p, and 1 - u is exact and is itself a possible
# draw, which antithetic variates rely on. The open interval costs one bit of
# resolution: 52 for Float64, 23 for Float32, 10 for Float16. The more obvious
# constructions each fail at one end: dividing by typemax(UInt64) or rounding a
# Float64 to a Float32 can give 1, and filling the mantissa of 1.0 and
# subtracting 1 can give 0.
#
# It is computed by filling the mantissa of 1.0 with k and subtracting the float
# just below 1: (1 + k 2^(1-p)) - (1 - 2^-p) = (2k + 1) 2^-p. The exact result is
# representable, so IEEE 754 makes the subtraction exact on every device -- the
# same bits as Float(2k + 1) * 2^-p, without the integer-to-float conversion,
# which is the slower instruction on a GPU. A narrower float from a 64-bit word
# takes the top 32 bits of the word, then the construction for a 32-bit one.
#
# The methods are on the sampler rather than on `::Type{T}`, so that
# `rand(rng, T)`, `rand(rng, T, n)` and `rand!` share one path.

@inline _u01(::Type{Float64}, u::UInt64) = reinterpret(Float64, 0x3ff0000000000000 | (u >>> 12)) - prevfloat(1.0)
@inline _u01(::Type{Float32}, u::UInt64) = _u01(Float32, (u >>> 32) % UInt32)
@inline _u01(::Type{Float16}, u::UInt64) = _u01(Float16, (u >>> 32) % UInt32)
@inline _u01(::Type{Float32}, u::UInt32) = reinterpret(Float32, 0x3f800000 | (u >>> 9)) - prevfloat(1f0)
@inline _u01(::Type{Float16}, u::UInt32) = reinterpret(Float16, 0x3c00 | ((u >>> 22) % UInt16)) - prevfloat(Float16(1))

# The MRG families produce a Float64 in (0, 1) natively. A narrower float takes
# the top 32 bits of that fraction through the construction above, so it costs
# one step, like the Float64.
@inline _u01(::Type{T}, u::Float64) where {T<:Union{Float16,Float32}} =
    _u01(T, unsafe_trunc(UInt32, u * 0x1p32))
