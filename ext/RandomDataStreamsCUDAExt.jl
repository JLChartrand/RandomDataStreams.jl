# CUDA extension for RandomDataStreams.
#
# Loaded automatically whenever CUDA.jl is loaded alongside RandomDataStreams
# (Julia's package-extension mechanism, see the [weakdeps]/[extensions]
# sections of Project.toml), so a CPU-only user never pulls in a CUDA
# dependency.
#
# Two capabilities live here.
#
#   * `rand!` into a `CuArray{Float64}`, `CuArray{Float32}` or `CuArray{Float16}`
#     for every counter-based generator -- Philox4x32, Philox4x64, Threefry4x32,
#     Threefry4x64 -- bit for bit what the same generator gives on the CPU, from
#     any position, and leaving the host object exactly where the CPU would.
#     Element k is `RandomDataStreams._draw_at(T, G, key, hi, lo, w0, k)`, the
#     pure function the CPU test suite checks against the stream object: the
#     same words in the same order, through the same `open01` conversion, whose
#     arithmetic (shifts, an exact integer-to-float conversion, a multiplication
#     by a power of two) rounds identically on any IEEE 754 device. A Float64
#     takes two words of a 32-bit family and one of a 64-bit family; a Float32
#     or Float16 takes one word of either. One thread per element.
#
#   * Normal variates for Philox4x32 -- `randn!` (Box-Muller), and for
#     comparison `randn_inversion!` and `randn_polar!`. These are not the CPU's
#     algorithm (the Ziggurat, which does not fit a kernel: its rejection loop
#     diverges within a warp), and their transcendental functions come from
#     libdevice rather than Julia's libm, so they do not match any CPU output
#     bit for bit. Their uniforms are `open01` of the same words `rand!` uses.
#     They need a block-aligned generator.
#
# Box-Muller (not the Ziggurat algorithm the CPU `randn` uses) is the fit for a
# GPU kernel: it consumes exactly two uniforms and produces exactly two normals
# every time, with no rejection loop and so no warp divergence.
#
# The two comparison paths (branch philox-gpu-randn-test) are motivated by
# variance-reduction techniques (antithetic variates, RQMC) that manipulate the
# *uniform* stream and expect the uniform -> normal transform to carry that
# structure through faithfully:
#
#   * `randn_inversion!` -- Z = Phi^-1(U) via CUDA's `normcdfinv` device
#     intrinsic, one uniform to one normal, componentwise. Since Phi^-1 is
#     monotonic and odd around u = 1/2, reflecting a single input uniform
#     (u -> 1-u) exactly negates that one output and nothing else. Box-Muller
#     does not have this property (see its docstring for why), which is the
#     whole reason this file grew a second algorithm rather than just
#     reusing `randn!`.
#   * `randn_polar!` -- Marsaglia's polar method (accept-reject), added only
#     as the third point of comparison, not for production use. See its
#     docstring for the two costs it demonstrates: warp divergence from a
#     data-dependent rejection loop, and a counter-advance that must reserve
#     for a worst case instead of committing to an exact one.
#
# The per-thread primitives (`philox4x32_10`, `philox4x32_counter`, `open01`)
# that a kernel of your own would use are plain functions in the package and
# need no CUDA-specific wrapping at all.
module RandomDataStreamsCUDAExt

using CUDA
using Random: Random
using RandomDataStreams: RandomDataStreams, CBRNG, PhiloxRNG, philox4x32_10, open01
import RandomDataStreams: randn_inversion!, randn_polar!

# CUDA.jl moved its device math intrinsics into a `CUDACore` subpackage
# somewhere around v6; `normcdfinv` (the libdevice inverse-normal-CDF this
# extension's `randn_inversion!` is built on) lives at `CUDA.normcdfinv` on
# 5.x and `CUDA.CUDACore.normcdfinv` on 6.x, and neither version re-exports it
# under the other's name. Project.toml's compat allows both major versions,
# so this resolves the name once at load time instead of hardcoding one.
const _normcdfinv = isdefined(CUDA, :normcdfinv) ? CUDA.normcdfinv : CUDA.CUDACore.normcdfinv

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
function _fill_kernel!(A, ::Type{G}, key, hi::UInt64, lo::UInt64, w0::Int, n::Int) where {G}
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        @inbounds A[k + 1] = RandomDataStreams._draw_at(eltype(A), G, key, hi, lo, w0, k)
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

# One thread per Philox block, writing up to two normals via inversion:
# Phi^-1 applied to each of the block's two word-pairs independently -- no
# pairing of uniforms into a shared radius/angle the way Box-Muller needs, so
# each output depends on exactly one input uniform.
function _philox_randn_kernel_inversion!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based block index
    if 2b < n
        blk = _philox_block(key, base_hi, base_lo, b)
        u1 = open01(_pair_lo(blk))
        u2 = open01(_pair_hi(blk))
        @inbounds A[2b + 1] = _normcdfinv(u1)
        2b + 2 <= n && (@inbounds A[2b + 2] = _normcdfinv(u2))
    end
    return nothing
end

# Native word-per-output Float32 inversion: one Phi^-1 call per word, four
# independent normals per block -- same rate as the Float32 Box-Muller
# kernel, same reason (a Philox4x32 word is already 32 bits).
function _philox_randn_kernel_inversion_f32!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        blk = _philox_block(key, base_hi, base_lo, k >> 2)
        @inbounds A[k + 1] = _normcdfinv(open01(blk[(k & 3) + 1]))
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
    @cuda threads = threads blocks = blocks _fill_kernel!(A, typeof(rng), rng.key, hi, lo, w0, n)
    RandomDataStreams._skip_draws!(rng, T, n)
    return A
end

"""
    Random.randn!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with standard normal draws via Box-Muller, one
independent Philox block feeding one pair of normals, and advance `rng`'s
counter on the host by the number of blocks consumed, so GPU normal and uniform
fills on the same stream stay non-overlapping with each other and with the CPU
path.

Box-Muller, not the Ziggurat algorithm the CPU `randn` uses: it takes exactly
two uniforms and produces exactly two normals every time, so every thread
does the same fixed amount of work. Its output is **not** meant to match CPU
`randn` bit for bit -- the algorithms differ.

`rng` must be block-aligned: fresh from [`next_stream!`](@ref), or just after
[`reset_substream!`](@ref), [`reset_stream!`](@ref) or
[`next_substream!`](@ref), or after a fill that ended on a block boundary. On a
generator that has made draws mid-block this throws `ArgumentError`.
"""
function Random.randn!(rng::PhiloxRNG, A::CuArray{Float64})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 2)
    base_hi, base_lo = _philox_gpu_prep!("randn!", rng, nblocks)

    threads = 256
    blocks = cld(nblocks, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel!(A, rng.key, base_hi, base_lo, n)
    return A
end

"""
    Random.randn!(rng::PhiloxRNG, A::CuArray{Float32}) -> A

Fill `A` on the device with standard normal `Float32` draws via Box-Muller,
one independent Philox4x32-10 block feeding two pairs of normals -- four per
block, twice the `Float64` method's rate, for the same reason `rand!` does:
a word is already 32 bits, so no pairing is needed to build a uniform.
Advances `rng`'s counter by the number of blocks consumed, same contract as
the `Float64` method.

Same caveats as the `Float64` method: Box-Muller, not Ziggurat, and not meant
to match any CPU output bit for bit. Same block-alignment requirement.
"""
function Random.randn!(rng::PhiloxRNG, A::CuArray{Float32})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 4)
    base_hi, base_lo = _philox_gpu_prep!("randn!", rng, nblocks)

    threads = 256
    blocks = cld(nblocks, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel_f32!(A, rng.key, base_hi, base_lo, n)
    return A
end

"""
    randn_inversion!(rng::PhiloxRNG, A::CuArray{Float64}) -> A
    randn_inversion!(rng::PhiloxRNG, A::CuArray{Float32}) -> A

Fill `A` on the device with standard normal draws via inversion: each output
is `Phi^-1(u)` for one Philox-generated uniform `u`, computed with CUDA's
`normcdfinv`/`normcdfinvf` device intrinsic (`__nv_normcdfinv` in libdevice)
-- a single, branch-free call per output. Same counter-advance contract as
`randn!` (this is a drop-in alternative to it: same block count for the same
`n`, same element type methods), so the two can be swapped without touching
any surrounding stream bookkeeping.

Why this exists alongside `randn!`'s Box-Muller: `Phi^-1` is monotonic and
odd around `u = 1/2`, so `Phi^-1(1 - u) == -Phi^-1(u)` exactly, and -- unlike
Box-Muller -- each output here depends on exactly *one* input uniform, not a
pair. That combination is exactly what antithetic variates and RQMC need:
those techniques manipulate the uniform stream componentwise (reflecting a
single coordinate `u -> 1 - u`, or replacing it with a low-discrepancy point)
and rely on the uniform -> normal transform to carry that structure through
unchanged in every other coordinate. Box-Muller does not have this property:
it mixes two uniforms nonlinearly (one sets a radius, the other an angle), so
reflecting just one of them changes an output that is not even the one "at"
that uniform, in a way unrelated to negation. `randn_polar!`, included in
this file for the same three-way comparison, has the same failure for the
same reason, on top of its own (see its docstring).

Its uniforms come from [`open01`](@ref), which never returns 0 or 1, so
`normcdfinv` is finite for every draw.

Not bit-compatible with any CPU path, and not intended to be; not exported by
`Random`, since it is not a Base standard-normal algorithm, just this
package's own comparison function. Same block-alignment requirement as
`randn!`; see its docstring.
"""
function randn_inversion!(rng::PhiloxRNG, A::CuArray{Float64})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 2)
    base_hi, base_lo = _philox_gpu_prep!("randn_inversion!", rng, nblocks)

    threads = 256
    blocks = cld(nblocks, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel_inversion!(A, rng.key, base_hi, base_lo, n)
    return A
end

function randn_inversion!(rng::PhiloxRNG, A::CuArray{Float32})
    n = length(A)
    n == 0 && return A
    nblocks = cld(n, 4)
    base_hi, base_lo = _philox_gpu_prep!("randn_inversion!", rng, nblocks)

    # One thread per *output element* (this kernel, like the Float32 `rand!`
    # kernel, indexes `k` up to `n`), not per block: the grid is sized from `n`
    # even though the counter advance above is sized from `nblocks`. Sizing it
    # from `nblocks` wrote only the first quarter of `A`.
    threads = 256
    blocks = cld(n, threads)
    @cuda threads = threads blocks = blocks _philox_randn_kernel_inversion_f32!(A, rng.key, base_hi, base_lo, n)
    return A
end

"""
    randn_polar!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with standard normal draws via Marsaglia's polar
method (an accept-reject algorithm): draw `(x, y)` uniform on `(-1, 1)^2`
until `0 < x^2 + y^2 < 1`, then scale by `sqrt(-2 log(s) / s)`. Included only
as the third point of comparison for `randn!` (Box-Muller) and
`randn_inversion!` in this branch -- not intended for production use.

Two costs this demonstrates directly, beyond sharing Box-Muller's broken
antithetic/RQMC structure (see `randn_inversion!`'s docstring; a single
attempt here also mixes two uniforms nonlinearly through `s`, and reflecting
just one of them does not negate the eventual output either):

  * Warp divergence: the number of attempts before acceptance varies per
    thread (`P(reject) ~ 1 - pi/4 ~ 0.215` per attempt, mean ~1.27
    attempts), so threads in the same warp finish their rejection loops at
    different times and the warp runs at the pace of its slowest lane --
    with no such loop at all in `randn!` or `randn_inversion!`.
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
    `randn!` or `randn_inversion!` do.

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
