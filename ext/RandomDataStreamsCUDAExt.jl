# CUDA extension for RandomDataStreams.
#
# Loaded automatically whenever CUDA.jl is loaded alongside RandomDataStreams
# (Julia's package-extension mechanism, see the [weakdeps]/[extensions]
# sections of Project.toml), so a CPU-only user never pulls in a CUDA
# dependency.
#
# Two capabilities live here.
#
#   * `rand!`, `randn!` and `randexp!` into a `CuArray{Float64}`, `CuArray{Float32}`
#     or `CuArray{Float16}` for every counter-based generator -- Philox4x32,
#     Philox4x64, Threefry4x32, Threefry4x64 -- bit for bit what the same
#     generator gives on the CPU, from any position, and leaving the host object
#     exactly where the CPU would. Element k of a uniform fill is
#     `RandomDataStreams._draw_at(T, G, key, hi, lo, w0, k)`, the pure function the
#     CPU test suite checks against the stream object: the same words in the same
#     order, through the same `open01` conversion, whose one floating-point
#     operation -- a subtraction with a representable exact result -- gives the
#     same bits on any IEEE 754 device. A Float64 takes two words of a 32-bit
#     family and one of a 64-bit family; a Float32 or Float16 takes one word of
#     either. A normal or exponential fill takes the k-th Float64 uniform through
#     `norminv` or `-log(1 - u)`, which the package computes with correctly
#     rounded operations only (src/normal.jl), and rounds it to the array's type:
#     the CPU's `randn!` and `randexp!`, exactly. `randn32!` does the same in
#     single precision from the Float32 uniforms. One thread per element.
#
#   * Box-Muller normals for Philox4x32, `randn_boxmuller!`, and Marsaglia's polar
#     method, `randn_polar!`, kept for comparison with inversion. Their
#     transcendental functions come from libdevice rather than Julia's libm, so
#     they match no CPU output bit for bit, and neither carries antithetic
#     variates or randomized quasi-Monte Carlo through from the uniforms -- see
#     "Normal variates and variance reduction" in the manual. They need a
#     block-aligned generator.
#
# The per-thread primitives (`philox4x32_10`, `philox4x32_counter`, `open01`,
# `norminv`) that a kernel of your own would use are plain functions in the
# package and need no CUDA-specific wrapping at all.
module RandomDataStreamsCUDAExt

using CUDA
using Random: Random
using RandomDataStreams: RandomDataStreams, CBRNG, PhiloxRNG, philox4x32_10, open01, norminv
import RandomDataStreams: randn_inversion!, randn_polar!, randn_boxmuller!, randn32!

# The block `b` (0-based) past the stream's current counter (base_hi, base_lo).
@inline function _philox_block(key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, b)
    lo, carry = Base.add_with_overflow(base_lo, UInt64(b))
    hi = base_hi + UInt64(carry)
    ctr = (lo % UInt32, (lo >> 32) % UInt32, hi % UInt32, (hi >> 32) % UInt32)
    return philox4x32_10(ctr, key)
end

@inline _pair_lo(blk) = (UInt64(blk[2]) << 32) | UInt64(blk[1])
@inline _pair_hi(blk) = (UInt64(blk[4]) << 32) | UInt64(blk[3])

# Element k of a uniform fill: one thread per element, each computing the block
# its word(s) come from. Every draw is independent of the launch configuration.
# `straddle` is `Val(isodd(w0))`: see `_draw_at` for why it is a type parameter.
function _fill_kernel!(A, ::Type{G}, key, hi::UInt64, lo::UInt64, w0::Int, n::Int, straddle::Val) where {G}
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        @inbounds A[k + 1] = RandomDataStreams._draw_at(eltype(A), G, key, hi, lo, w0, k, straddle)
    end
    return nothing
end

# Element k of a fill by inversion: the k-th uniform of type U the stream would
# draw -- the one the CPU's `randn`, `randexp` or `randn32` would take -- through
# `f`, and rounded to the array's type, as the CPU rounds a Float32 normal.
function _invert_kernel!(A, f::F, ::Type{U}, ::Type{G}, key, hi::UInt64, lo::UInt64, w0::Int, n::Int,
                         straddle::Val) where {F,U,G}
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        u = RandomDataStreams._draw_at(U, G, key, hi, lo, w0, k, straddle)
        @inbounds A[k + 1] = convert(eltype(A), f(u))
    end
    return nothing
end

# One thread per Philox block, writing up to two normals: Box-Muller applied
# to the block's two uniforms. `open01` never returns 0, so `log(u1)` is finite.
function _philox_randn_kernel!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based block index
    if 2b < n
        blk = _philox_block(key, base_hi, base_lo, b)
        u1 = open01(_pair_lo(blk))
        u2 = open01(_pair_hi(blk))
        r     = sqrt(-2.0 * log(u1))
        theta = 2.0 * Float64(pi) * u2
        @inbounds A[2b + 1] = r * cos(theta)
        2b + 2 <= n && (@inbounds A[2b + 2] = r * sin(theta))
    end
    return nothing
end

# One thread per Philox block, writing up to four normals: Box-Muller applied
# to each of the block's two word-pairs (words 1,2 and words 3,4).
function _philox_randn_kernel_f32!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based block index
    if 4b < n
        blk = _philox_block(key, base_hi, base_lo, b)

        u1, u2 = open01(blk[1]), open01(blk[2])
        r1     = sqrt(-2f0 * log(u1))
        theta1 = 2f0 * Float32(pi) * u2
        @inbounds A[4b + 1] = r1 * cos(theta1)
        4b + 2 <= n && (@inbounds A[4b + 2] = r1 * sin(theta1))

        if 4b + 3 <= n
            u3, u4 = open01(blk[3]), open01(blk[4])
            r2     = sqrt(-2f0 * log(u3))
            theta2 = 2f0 * Float32(pi) * u4
            @inbounds A[4b + 3] = r2 * cos(theta2)
            4b + 4 <= n && (@inbounds A[4b + 4] = r2 * sin(theta2))
        end
    end
    return nothing
end

# Worst-case Philox blocks reserved per output pair for `randn_polar!`.
# Marsaglia's polar method rejects a candidate `(x, y)` with probability
# `1 - pi/4 ~ 0.215`; the average number of attempts is `4/pi ~ 1.27`, and
# `0.215^32 ~ 5e-23` -- reserving this many blocks and always advancing the
# host counter by the full reservation keeps the counter-advance contract
# static (computable from `n` alone, before the kernel runs at all), at the
# cost of routinely reserving ~25x more counter space than the average draw
# actually uses.
const _POLAR_BUDGET = 32

# One thread per output pair-slot: repeatedly draws a Philox block, maps its
# two word-pairs to (x, y) uniform on (-1, 1)^2, and accepts the first one
# landing inside the unit disc (excluding the center, where the log below
# would blow up). Different threads take different numbers of attempts --
# exactly the warp-divergence cost this kernel exists to demonstrate. The
# `s = clamp(...)` after the loop is a fallback for the budget being
# exhausted, which is not expected to ever trigger at this budget; see
# `randn_polar!`'s docstring.
function _philox_randn_kernel_polar!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based pair-slot index
    if 2p < n
        x = 0.0
        y = 0.0
        s = 1.0
        for attempt in 0:(_POLAR_BUDGET - 1)
            blk = _philox_block(key, base_hi, base_lo, p * _POLAR_BUDGET + attempt)
            x = 2.0 * open01(_pair_lo(blk)) - 1.0
            y = 2.0 * open01(_pair_hi(blk)) - 1.0
            s = x * x + y * y
            (s > 0.0 && s < 1.0) && break
        end
        s = clamp(s, 1.0e-300, 0.9999999999999998)     # largest Float64 below 1.0
        f = sqrt(-2.0 * log(s) / s)
        @inbounds A[2p + 1] = x * f
        2p + 2 <= n && (@inbounds A[2p + 2] = y * f)
    end
    return nothing
end

# Checks alignment, advances `rng.ctr` by `nblocks`, and returns
# (base_hi, base_lo) for the kernel to start from -- the bookkeeping the
# normal-variate paths share. `rand!` needs none of it: it starts anywhere.
function _philox_gpu_prep!(fname, rng::PhiloxRNG, nblocks::Integer)
    rng.idx == 5 || throw(ArgumentError(
        "$fname(::PhiloxRNG, ::CuArray) requires a block-aligned generator; " *
        "call reset_substream!(rng) (or start from a fresh stream) first"))
    base = rng.ctr
    rng.ctr = (base + UInt128(nblocks)) & RandomDataStreams._ctr_mask(UInt32, Val(4))
    return (base >> 64) % UInt64, base % UInt64
end

"""
    Random.rand!(rng::CBRNG, A::CuArray{T}) -> A    (T = Float64, Float32, Float16)

Fill `A` on the device with uniforms in `(0, 1)`, and leave `rng` on the host
exactly where the same fill on the CPU would: `Array(rand!(rng, A))` equals
`rand!(copy(rng), Array(A))` bit for bit, for every counter-based generator and
from any position, mid-block included. A later draw, on either device,
continues from there.

A `Float64` takes two words of a 32-bit family and one word of a 64-bit family,
a `Float32` or `Float16` one word of either, as on the CPU. Each element is
computed by its own thread from the key and the counter alone, so the result
does not depend on the launch configuration.
"""
function Random.rand!(rng::CBRNG, A::CuArray{T}) where {T<:Union{Float16,Float32,Float64}}
    n = length(A)
    n == 0 && return A
    hi, lo, w0 = RandomDataStreams._draw_position(rng)

    threads = 256
    blocks = cld(n, threads)
    if isodd(w0)
        @cuda threads = threads blocks = blocks _fill_kernel!(A, typeof(rng), rng.key, hi, lo, w0, n, Val(true))
    else
        @cuda threads = threads blocks = blocks _fill_kernel!(A, typeof(rng), rng.key, hi, lo, w0, n, Val(false))
    end
    RandomDataStreams._skip_draws!(rng, T, n)
    return A
end

function _invert_fill!(f, ::Type{U}, rng::CBRNG, A::CuArray) where {U}
    n = length(A)
    n == 0 && return A
    hi, lo, w0 = RandomDataStreams._draw_position(rng)

    threads = 256
    blocks = cld(n, threads)
    if isodd(w0)
        @cuda threads = threads blocks = blocks _invert_kernel!(A, f, U, typeof(rng), rng.key, hi, lo, w0, n, Val(true))
    else
        @cuda threads = threads blocks = blocks _invert_kernel!(A, f, U, typeof(rng), rng.key, hi, lo, w0, n, Val(false))
    end
    RandomDataStreams._skip_draws!(rng, U, n)
    return A
end

"""
    Random.randn!(rng::CBRNG, A::CuArray{T}) -> A    (T = Float64, Float32, Float16)

Fill `A` on the device with standard normals by inversion -- [`norminv`](@ref)
of the `Float64` uniforms the stream would draw next -- and leave `rng` where
the same fill on the CPU would: `Array(randn!(rng, A))` equals
`randn!(copy(rng), Array(A))` bit for bit, for every counter-based generator
and from any position, mid-block included.

A `Float32` or `Float16` normal is the `Float64` one rounded, as on the CPU, so
each takes a 64-bit uniform: two words of a 32-bit family. Inversion is what
keeps antithetic variates, common random numbers and randomized quasi-Monte
Carlo intact; [`randn_boxmuller!`](@ref) is the faster alternative that does
not.
"""
Random.randn!(rng::CBRNG, A::CuArray{T}) where {T<:Union{Float16,Float32,Float64}} =
    _invert_fill!(norminv, Float64, rng, A)

"""
    Random.randexp!(rng::CBRNG, A::CuArray{T}) -> A    (T = Float64, Float32, Float16)

Fill `A` on the device with exponential variates of rate 1 by inversion,
`-log(1 - u)` of the `Float64` uniforms the stream would draw next: the CPU's
`randexp!` bit for bit, with the same contract as `randn!`.
"""
Random.randexp!(rng::CBRNG, A::CuArray{T}) where {T<:Union{Float16,Float32,Float64}} =
    _invert_fill!(RandomDataStreams._expinv, Float64, rng, A)

"""
    randn32!(rng::CBRNG, A::CuArray{Float32}) -> A

Fill `A` on the device with single-precision normals by inversion, the
`Float32` [`norminv`](@ref) of the `Float32` uniforms the stream would draw
next -- one word each -- and leave `rng` where the CPU's `randn32!` would: the
same values, bit for bit, in `Float32` arithmetic throughout. The fast choice
where `Float64` is slow; its tails end at ±5.29, where `randn!`'s reach ±8.21.
"""
randn32!(rng::CBRNG, A::CuArray{Float32}) = _invert_fill!(norminv, Float32, rng, A)

"""
    randn_boxmuller!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with standard normal draws via Box-Muller, one
independent Philox block feeding one pair of normals, and advance `rng`'s
counter on the host by the number of blocks consumed, so GPU normal and uniform
fills on the same stream stay non-overlapping with each other and with the CPU
path.

The alternative to `randn!`, which draws by inversion: Box-Muller takes exactly
two uniforms and produces exactly two normals every time, with no tail branch,
but it mixes the two uniforms nonlinearly -- one sets a radius, the other an
angle -- so reflecting a uniform does not negate a normal, and a quasi-Monte
Carlo coordinate does not map to one variate. Its `log`, `cos` and `sin` come
from libdevice, so its output matches no CPU output bit for bit.

`rng` must be block-aligned: fresh from [`next_stream!`](@ref), or just after
[`reset_substream!`](@ref), [`reset_stream!`](@ref) or
[`next_substream!`](@ref), or after a fill that ended on a block boundary. On a
generator that has made draws mid-block this throws `ArgumentError`.
"""
function randn_boxmuller!(rng::PhiloxRNG, A::CuArray{Float64})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 2)
    base_hi, base_lo = _philox_gpu_prep!("randn_boxmuller!", rng, nblocks)

    threads = 256
    blocks = cld(nblocks, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel!(A, rng.key, base_hi, base_lo, n)
    return A
end

"""
    randn_boxmuller!(rng::PhiloxRNG, A::CuArray{Float32}) -> A

Fill `A` on the device with standard normal `Float32` draws via Box-Muller,
one independent Philox4x32-10 block feeding two pairs of normals -- four per
block, twice the `Float64` method's rate, for the same reason `rand!` does:
a word is already 32 bits, so no pairing is needed to build a uniform.
Advances `rng`'s counter by the number of blocks consumed, same contract as
the `Float64` method.

Same caveats as the `Float64` method, and the same block-alignment requirement.
"""
function randn_boxmuller!(rng::PhiloxRNG, A::CuArray{Float32})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 4)
    base_hi, base_lo = _philox_gpu_prep!("randn_boxmuller!", rng, nblocks)

    threads = 256
    blocks = cld(nblocks, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel_f32!(A, rng.key, base_hi, base_lo, n)
    return A
end

# `randn_inversion!` was the libdevice inversion, Philox4x32 and block-aligned
# only; `randn!` is inversion now, for every counter-based family and on the
# CPU's sequence, so the old name forwards to it.
function randn_inversion!(rng::CBRNG, A::CuArray{T}) where {T<:Union{Float16,Float32,Float64}}
    Base.depwarn("`randn_inversion!` is deprecated: `randn!` now draws by inversion, " *
                 "for every counter-based generator and as the CPU does", :randn_inversion!)
    return Random.randn!(rng, A)
end

"""
    randn_polar!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with standard normal draws via Marsaglia's polar
method (an accept-reject algorithm): draw `(x, y)` uniform on `(-1, 1)^2`
until `0 < x^2 + y^2 < 1`, then scale by `sqrt(-2 log(s) / s)`. Included only
as the third point of comparison, beside inversion (`randn!`) and Box-Muller
(`randn_boxmuller!`) -- not intended for production use.

Two costs this demonstrates directly, beyond sharing Box-Muller's broken
antithetic/RQMC structure (see `randn_boxmuller!`'s docstring; a single
attempt here also mixes two uniforms nonlinearly through `s`, and reflecting
just one of them does not negate the eventual output either):

  * Warp divergence: the number of attempts before acceptance varies per
    thread (`P(reject) ~ 1 - pi/4 ~ 0.215` per attempt, mean ~1.27
    attempts), so threads in the same warp finish their rejection loops at
    different times and the warp runs at the pace of its slowest lane --
    with no such loop at all in `randn!` or `randn_boxmuller!`.
  * A *reserved, worst-case* counter advance rather than an exact one: this
    package's whole design commits to a counter advance the host can compute
    from `n` alone, before the kernel runs. An algorithm whose attempt count
    is unbounded in principle cannot offer an exact advance under that
    constraint, only a reservation for a worst case; see `_POLAR_BUDGET`
    above for the number chosen and why it is safe in practice. Every draw
    pays for the reservation (`_POLAR_BUDGET` Philox blocks per output pair)
    whether or not it is used -- wasteful of counter space, though
    negligible against Philox's `2^128`-block space, and a second,
    independent way this family of algorithms does not fit this library's
    block-aligned, statically-advancing counter model as cleanly as
    `randn!` or `randn_boxmuller!` do.

`Float32` is not provided: it adds nothing further to the comparison this
function exists for.
"""
function randn_polar!(rng::PhiloxRNG, A::CuArray{Float64})
    n = length(A)
    n == 0 && return A
    npairs = cld(n, 2)
    nblocks = npairs * _POLAR_BUDGET
    base_hi, base_lo = _philox_gpu_prep!("randn_polar!", rng, nblocks)

    threads = 256
    blocks = cld(npairs, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel_polar!(A, rng.key, base_hi, base_lo, n)
    return A
end

end # module
