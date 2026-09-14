#
# Implementation of MRG63k3a RNG Streams
#
# L'Ecuyer, "Good Parameters and Implementations for Combined Multiple
# Recursive Random Number Generators", Operations Research 47, 1 (1999),
# 159--164; fourth entry of Table II, period length approximately 2^377.
#
# The 32-bit sibling of this file is mrg32k3a/mrg32k3aPrivateBehavior.jl. The
# recurrences have the same shape; only the arithmetic differs, because a
# modulus just under 2^63 leaves no room for the Float64 trick that carries
# MRG32k3a.
########################################
module PrivateMRG63Funcs

# Both moduli are just below 2^63, so a product a*x needs 128 bits. That is the
# whole difference with MRG32k3a: there, a*x fits in a Float64 mantissa.
const m1 = Int64(9223372036854769163)   # 2^63 - 6645
const m2 = Int64(9223372036854754679)   # 2^63 - 21129
const c1 = Int64(6645)                  # 2^63 - m1
const c2 = Int64(21129)                 # 2^63 - m2

const a12  = Int64(1754669720)
const a13n = Int64(3182104042)          # x1n = a12*x1_{n-2} - a13n*x1_{n-3}
const a21  = Int64(31387477935)
const a23n = Int64(6199136374)          # x2n = a21*x2_{n-1} - a23n*x2_{n-3}

# Same testless formulation as MRG32k3a: the negative coefficient is carried as
# a positive number and subtracted, and a multiple of the modulus is added so
# that the argument of the reduction can never be negative.
const corr1 = Int128(m1) * a13n
const corr2 = Int128(m2) * a23n

const norm = Float64(1.0 / (1 + m1))    # 1.0842021724855052e-19, as in the C code

const DEFAULT_SEED = [ 12345, 12345, 12345, 12345, 12345, 12345 ]

"""
Reduces `t` modulo `m1`, for `0 <= t <= (a12 + a13n)*m1 < 2^96`, which is the
largest value the component-1 step can produce.

Pseudo-Mersenne reduction: `2^63 = m1 + c1`, so splitting `t` at bit 63 into
`t = hi*2^63 + lo` gives `t ≡ hi*c1 + lo (mod m1)`. Over that range the high
part is below `2^33` and `hi*c1 < 2^46`, so the folded value is below
`2^63 + 2^46`, which is under `2*m1`, and one conditional subtraction finishes
the job. No division, and no 128-bit remainder, which is a software routine.
"""
@inline function mod_m1(t::Int128)
    hi = (t >> 63) % UInt64
    lo = (t % UInt64) & 0x7fff_ffff_ffff_ffff
    v = hi * (c1 % UInt64) + lo
    return (v < (m1 % UInt64) ? v : v - (m1 % UInt64)) % Int64
end

"""
Reduces `t` modulo `m2`, for `0 <= t <= (a21 + a23n)*m2 < 2^99`, the largest
value the component-2 step can produce. Here `hi < 2^36` and `hi*c2 < 2^51`,
so the same single conditional subtraction suffices. See `mod_m1`.
"""
@inline function mod_m2(t::Int128)
    hi = (t >> 63) % UInt64
    lo = (t % UInt64) & 0x7fff_ffff_ffff_ffff
    v = hi * (c2 % UInt64) + lo
    return (v < (m2 % UInt64) ? v : v - (m2 % UInt64)) % Int64
end

"""
Ensures a given seed is valid for the MRG63k3a random number generator.
Allocation-free.
"""
function checkseed(x::Vector{Int})
    length(x) == 6 || return false
    @inbounds for i in 1:6
        x[i] >= 0 || return false
        i <= 3 ? (x[i] < m1 || return false) : (x[i] < m2 || return false)
    end
    @inbounds (x[1] != 0 || x[2] != 0 || x[3] != 0) || return false
    @inbounds (x[4] != 0 || x[5] != 0 || x[6] != 0) || return false
    return true
end

"""
Computes `(a*s + c) % m` for arbitrary `0 <= a, s, c < m < 2^63`.

Unlike its MRG32k3a namesake this one is not on any hot path -- it serves the
jump matrices only -- so it takes the straightforward route through a 128-bit
remainder instead of the approximate-factoring dance of the reference C code.
"""
function MultModM(a::Int64, s::Int64, c::Int64, m::Int64)
    v = mod(widemul(a, s) + c, Int128(m))
    return Int64(v)
end

"""
Computes A*s % m, assuming 0 <= s[i] < m.
"""
function MatVecModM(A::Array{Int64,2}, s::AbstractVector{Int64}, m::Int64)

    v = [0, 0, 0]
    for i = 1:3
        for j = 1:3
            @inbounds v[i] = MultModM(A[i,j], s[j], v[i], m)
        end
    end

    return v
end

"""
Computes matrix A*B % m, assuming 0 <= entries < m.
"""
function MatMatModM(A::Array{Int64,2}, B::Array{Int64,2}, m::Int64)

    C = zeros(Int64, 3, 3)
    for i = 1:3
        C[:,i] = MatVecModM(A, B[:,i], m)
    end

    return C
end

"""
Computes the matrix A^(2^e) % m.
"""
function MatTwoPowModM(A::Array{Int64,2}, e::Int64, m::Int64)

    B = A
    for i = 1:e
        B = MatMatModM(B, B, m)
    end

    return B
end

"""
Computes the matrix (A^n % m).
"""
function MatPowModM(A::Array{Int64,2}, n::Int64, m::Int64)
    W = A
    B = [1 0 0; 0 1 0; 0 0 1]

    while n > 0
        if ( n % 2 == 1 )
            B = MatMatModM(W, B, m)
        end
        W = MatMatModM(W, W, m)
        n ÷= 2
    end
    return B
end

# One step of each component, as a matrix acting on (x_{n-3}, x_{n-2}, x_{n-1}).
const A1p0 =  [  0          1     0   ;
                 0          0     1   ;
                 m1 - a13n  a12   0   ]

const A2p0 =  [  0          1     0   ;
                 0          0     1   ;
                 m2 - a23n  0     a21 ]

# One step backwards. Reading the recurrences the other way,
#   x1_{n-3} = a13n^{-1} (a12 x1_{n-2} - x1_n),
#   x2_{n-3} = a23n^{-1} (a21 x2_{n-1} - x2_n),
# which is the first row of each matrix; the other two rows shift the state
# back. The same formulas reproduce L'Ecuyer's published InvA1 and InvA2 for
# MRG32k3a, which is how they were checked (see the test suite).
const inv13 = Int64(invmod(a13n, m1))
const inv23 = Int64(invmod(a23n, m2))

const InvA1 = [ MultModM(a12, inv13, 0, m1)  0  m1 - inv13 ;
                1                            0  0          ;
                0                            1  0          ]

const InvA2 = [ 0  MultModM(a21, inv23, 0, m2)  m2 - inv23 ;
                1  0                            0          ;
                0  1                            0          ]

# Stream and substream spacing. L'Ecuyer published jump matrices for MRG32k3a
# (2^127 and 2^76 of a 2^191 period) but none for MRG63k3a, so these are ours,
# scaled by the ratio of the periods: a 2^377 period cut into 2^127 streams of
# 2^250 numbers, each holding 2^100 substreams of 2^150 numbers. They are
# computed here rather than tabulated -- 400 matrix squarings, once, at
# precompilation.
const STREAM_EXPONENT    = 250
const SUBSTREAM_EXPONENT = 150

const A1p150 = MatTwoPowModM(A1p0, Int64(SUBSTREAM_EXPONENT), m1)
const A2p150 = MatTwoPowModM(A2p0, Int64(SUBSTREAM_EXPONENT), m2)
const A1p250 = MatTwoPowModM(A1p0, Int64(STREAM_EXPONENT), m1)
const A2p250 = MatTwoPowModM(A2p0, Int64(STREAM_EXPONENT), m2)
end
const PMF63 = PrivateMRG63Funcs
