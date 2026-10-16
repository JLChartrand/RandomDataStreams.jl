# Normal and exponential variates, by inversion.
#
# Every generator of the package draws a standard normal as Φ⁻¹(u) and an
# exponential as -log(1 - u), u being the next `rand(rng, Float64)`: one uniform
# per variate, through a monotone map. That is what the variance-reduction
# techniques the package exists for need, and what the standard library's
# Ziggurat does not give -- it takes the sign, the layer and the abscissa from
# different bits of one word, and rejects now and then:
#
#   * antithetic variates: 1 - u gives -z, exactly, for the word generators,
#     whose uniforms lie on a grid symmetric about 1/2 (see floats.jl);
#   * common random numbers: the k-th variate is a function of the k-th uniform
#     alone, so two systems that draw the same uniforms draw the same variates;
#   * randomized quasi-Monte Carlo: one coordinate per variate and no
#     rejection, so a point of dimension d gives d variates.
#
# docs/src/streams.md, "Normal variates and variance reduction", has the table.
#
# Both maps use only IEEE 754 operations whose results are correctly rounded --
# +, -, *, /, sqrt and fma -- and integer operations on the bits. No libm call:
# the CPU's log and a GPU's (libdevice) disagree in the last bits. And no
# product is ever an operand of a sum: the GPU compiler contracts `a * b + c`
# into an fma where the CPU's does not, which changed one result in 200 000 by
# an ulp before every such sum was written as the fma itself. The same function
# therefore gives the same bits on the CPU and on a GPU, which is what lets the
# CUDA extension's `randn!` reproduce the CPU's `randn!` exactly.

# log -------------------------------------------------------------------------
#
# The algorithm of fdlibm's e_log.c, with its constants, under this notice:
#
#   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#   Developed at SunSoft, a Sun Microsystems, Inc. business.
#   Permission to use, copy, modify, and distribute this software is freely
#   granted, provided that this notice is preserved.
#
# x = 2^k (1 + f) with sqrt(2)/2 <= 1 + f < sqrt(2); log(1 + f) = 2 atanh(s),
# s = f / (2 + f), from a polynomial in s^2; then k log(2) in two parts. Less
# than one ulp of error. The only change is that each product feeding a sum is
# an explicit fma, for the reason above -- which also rounds once fewer.

const _LN2_HI = reinterpret(Float64, 0x3fe62e42fee00000)   # 6.93147180369123816490e-01
const _LN2_LO = reinterpret(Float64, 0x3dea39ef35793c76)   # 1.90821492927058770002e-10
const _LG1 = reinterpret(Float64, 0x3fe5555555555593)      # 6.666666666666735130e-01
const _LG2 = reinterpret(Float64, 0x3fd999999997fa04)      # 3.999999999940941908e-01
const _LG3 = reinterpret(Float64, 0x3fd2492494229359)      # 2.857142874366239149e-01
const _LG4 = reinterpret(Float64, 0x3fcc71c51d8e78af)      # 2.222219843214978396e-01
const _LG5 = reinterpret(Float64, 0x3fc7466496cb03de)      # 1.818357216161805012e-01
const _LG6 = reinterpret(Float64, 0x3fc39a09d078c69f)      # 1.531383769920937332e-01
const _LG7 = reinterpret(Float64, 0x3fc2f112df3e5244)      # 1.479819860511658591e-01

"""
    _log(x::Float64) -> Float64

The natural logarithm, bit for bit the same on every IEEE 754 device; see the
comment above. Within one ulp of `log`, and not meant to replace it elsewhere.
"""
@inline function _log(x::Float64)
    x > 0 || return x == 0 ? -Inf : NaN
    x < Inf || return x
    ux = reinterpret(UInt64, x)
    k = 0
    if ux < 0x0010000000000000                 # subnormal: scale into the normal range
        x *= 0x1p54
        ux = reinterpret(UInt64, x)
        k = -54
    end
    hx = (ux >> 32) % Int
    k += (hx >> 20) - 1023
    hx &= 0x000fffff
    i = (hx + 0x95f64) & 0x100000
    # 1 + f is x or x/2, whichever lies in [sqrt(2)/2, sqrt(2))
    x = reinterpret(Float64, (((hx | xor(i, 0x3ff00000)) % UInt64) << 32) | (ux & 0x00000000ffffffff))
    k += i >> 20
    f = x - 1.0
    dk = Float64(k)
    if (0x000fffff & (2 + hx)) < 3             # |f| < 2^-20
        f == 0 && return k == 0 ? 0.0 : fma(dk, _LN2_HI, dk * _LN2_LO)
        c = fma(-0.33333333333333333, f, 0.5)                 # R = f^2 c
        k == 0 && return fma(-(f * f), c, f)
        return fma(dk, _LN2_HI, -(fma(-dk, _LN2_LO, (f * f) * c) - f))
    end
    s = f / (2.0 + f)
    z = s * s
    w = z * z
    t1 = w * fma(w, fma(w, _LG6, _LG4), _LG2)
    R = fma(z, fma(w, fma(w, fma(w, _LG7, _LG5), _LG3), _LG1), t1)   # t2 + t1
    if ((hx - 0x6147a) | (0x6b851 - hx)) > 0
        hf = 0.5 * f                                         # hfsq = hf f
        a = fma(hf, f, R)                                    # hfsq + R
        k == 0 && return f - fma(-s, a, hf * f)
        return fma(dk, _LN2_HI, -(fma(hf, f, -fma(s, a, dk * _LN2_LO)) - f))
    else
        k == 0 && return fma(-s, f - R, f)
        return fma(dk, _LN2_HI, -(fma(s, f - R, -(dk * _LN2_LO)) - f))
    end
end

# Φ⁻¹ -------------------------------------------------------------------------
#
# Wichura's algorithm AS 241, PPND16 (Applied Statistics 37, 1988, 477-484):
# rational approximations of degree 7 in three regions, about 1e-16 relative
# error. The central one, |u - 1/2| <= 0.425 -- 85% of all draws -- needs no
# logarithm; the two tails need sqrt(-log(min(u, 1 - u))).

const _AS241_A = (3.38713_28727_96366_6080e0, 1.33141_66789_17843_7745e2,
                  1.97159_09503_06551_4427e3, 1.37316_93765_50946_1125e4,
                  4.59219_53931_54987_1457e4, 6.72657_70927_00870_0853e4,
                  3.34305_75583_58812_8105e4, 2.50908_09287_30122_6727e3)
const _AS241_B = (1.0, 4.23133_30701_60091_1252e1,
                  6.87187_00749_20579_0830e2, 5.39419_60214_24751_1077e3,
                  2.12137_94301_58659_5867e4, 3.93078_95800_09271_0610e4,
                  2.87290_85735_72194_2674e4, 5.22649_52788_52854_5610e3)
const _AS241_C = (1.42343_71107_49683_57734e0, 4.63033_78461_56545_29590e0,
                  5.76949_72214_60691_40550e0, 3.64784_83247_63204_60504e0,
                  1.27045_82524_52368_38258e0, 2.41780_72517_74506_11770e-1,
                  2.27238_44989_26918_45833e-2, 7.74545_01427_83414_07640e-4)
const _AS241_D = (1.0, 2.05319_16266_37758_82187e0,
                  1.67638_48301_83803_84940e0, 6.89767_33498_51000_04550e-1,
                  1.48103_97642_74800_74590e-1, 1.51986_66563_61645_71966e-2,
                  5.47593_80849_95344_94600e-4, 1.05075_00716_44416_84324e-9)
const _AS241_E = (6.65790_46435_01103_77720e0, 5.46378_49111_64114_36990e0,
                  1.78482_65399_17291_33580e0, 2.96560_57182_85048_91230e-1,
                  2.65321_89526_57612_30930e-2, 1.24266_09473_88078_43860e-3,
                  2.71155_55687_43487_57815e-5, 2.01033_43992_92288_13265e-7)
const _AS241_F = (1.0, 5.99832_20655_58879_37690e-1,
                  1.36929_88092_27358_05310e-1, 1.48753_61290_85061_48525e-2,
                  7.86869_13114_56132_59100e-4, 1.84631_83175_10054_68180e-5,
                  1.42151_17583_16445_88870e-7, 2.04426_31033_89939_78564e-15)

# c[1] + x (c[2] + x (c[3] + ...)), each step an fma. Recursion over the
# arguments rather than a loop over the tuple: it unrolls at compile time, and
# a tuple indexed at run time spills to local memory in a GPU kernel.
@inline _horner(x, c1) = c1
@inline _horner(x, c1, cs...) = fma(_horner(x, cs...), x, c1)

"""
    norminv(u) -> z

The standard normal quantile function Φ⁻¹: the `z` with Φ(z) = `u`, for `u` in
`(0, 1)`, and `-Inf`, `Inf` at 0 and 1. It is the map every generator of the
package uses for `randn` -- `randn(rng)` is `norminv(rand(rng, Float64))` -- and
is exported so that a randomized quasi-Monte Carlo point, or the uniforms of a
kernel of one's own, can be mapped the same way.

Wichura's AS 241 (relative error about 1e-16), computed with correctly rounded
IEEE 754 operations only, so a GPU gives the same bits as the CPU. Odd about
1/2: `norminv(1 - u) == -norminv(u)` whenever `1 - u` is exact, which it is for
every uniform the word generators return.

A `Float32` or `Float16` argument is computed in `Float64` and rounded.
"""
@inline function norminv(u::Float64)
    0 < u < 1 || return u == 0 ? -Inf : u == 1 ? Inf : NaN
    q = u - 0.5
    if abs(q) <= 0.425
        r = fma(-q, q, 0.180625)
        return q * _horner(r, _AS241_A...) / _horner(r, _AS241_B...)
    end
    r = sqrt(-_log(q < 0 ? u : 1.0 - u))
    z = if r <= 5.0
        r -= 1.6
        _horner(r, _AS241_C...) / _horner(r, _AS241_D...)
    else
        r -= 5.0
        _horner(r, _AS241_E...) / _horner(r, _AS241_F...)
    end
    return q < 0 ? -z : z
end
@inline norminv(u::T) where {T<:Union{Float16,Float32}} = T(norminv(Float64(u)))

# The exponential's quantile function, increasing in u like Φ⁻¹.
@inline _expinv(u::Float64) = -_log(1.0 - u)

# The Random API ---------------------------------------------------------------
#
# `randn(rng, T)` for T = Float16, Float32, Float64, `randn!`, `randn(rng, dims)`
# and complex normals all reach `randn(rng)` through the standard library, which
# converts; the same holds for `randexp`. So the scalar method is the one that
# sets the sequence, and a Float32 normal is the Float64 one rounded, as in the
# standard library -- its tails reach the Float64 grid's +-8.2, not the +-5.3 a
# 24-bit uniform would allow.

"""
    randn(rng::AbstractStreamableRNG) -> Float64

A standard normal by inversion: `norminv(rand(rng, Float64))`, one uniform per
variate. Every generator of the package draws its normals this way, rather than
by the standard library's Ziggurat, so that antithetic variates, common random
numbers and randomized quasi-Monte Carlo carry through from the uniforms to the
normals; see "Normal variates and variance reduction" in the manual. The CUDA
extension's `randn!` computes the same values.
"""
Random.randn(rng::AbstractStreamableRNG) = norminv(rand(rng, Float64))

"""
    randexp(rng::AbstractStreamableRNG) -> Float64

An exponential variate of rate 1 by inversion, `-log(1 - rand(rng, Float64))`,
for the same reasons as [`randn`](@ref Random.randn(::AbstractStreamableRNG)).
"""
Random.randexp(rng::AbstractStreamableRNG) = _expinv(rand(rng, Float64))

# Arrays: the uniforms first, in one fill -- the counter-based generators fill
# a Float64 array block by block -- then the map, which gives the same values as
# the scalar draws because `rand!` gives the same uniforms as `rand`.
function Random.randn!(rng::AbstractStreamableRNG, A::Array{Float64})
    rand!(rng, A)
    @inbounds for i in eachindex(A)
        A[i] = norminv(A[i])
    end
    return A
end

function Random.randexp!(rng::AbstractStreamableRNG, A::Array{Float64})
    rand!(rng, A)
    @inbounds for i in eachindex(A)
        A[i] = _expinv(A[i])
    end
    return A
end
