# CUDA extension for RandomDataStreams.
#
# Loaded automatically whenever CUDA.jl is loaded alongside RandomDataStreams
# (Julia's package-extension mechanism, see the [weakdeps]/[extensions]
# sections of Project.toml), so a CPU-only user never pulls in a CUDA
# dependency.
#
# Two capabilities live here, both keyed on the same idea: one Philox4x32
# block (four 32-bit words) is independent per-block work with no shared
# state, so a kernel gives each block to one thread and writes straight into
# the output array, then the counter is advanced on the host by the number of
# blocks consumed -- exactly as the CPU `_fill_u64!` path does.
#
#   * `rand!` -- a block's two word-pairs are two independent Float64
#     uniforms, matching what two consecutive scalar `rand(rng)` calls would
#     produce.
#   * `randn!` -- the same two uniforms, fed through Box-Muller instead of
#     kept separate, giving two independent standard normals. Box-Muller
#     (not the Ziggurat algorithm the CPU `randn` uses) is the fit for a GPU
#     kernel: it consumes exactly two uniforms and produces exactly two
#     normals every time, with no rejection loop and so no warp divergence.
#     Its output does not, and is not meant to, match CPU `randn` bit for
#     bit -- the algorithms differ.
#
# The per-thread primitives (`philox4x32_10`, `philox4x32_counter`) that a
# kernel would use directly are plain functions in src/philox/philox.jl and
# need no CUDA-specific wrapping at all.
module RandomDataStreamsCUDAExt

using CUDA
using Random: Random
using RandomDataStreams: RandomDataStreams, PhiloxRNG, philox4x32_10, close_open01

# The block `b` (0-based) past the stream's current counter (base_hi, base_lo).
@inline function _philox_block(key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, b)
    lo, carry = Base.add_with_overflow(base_lo, UInt64(b))
    hi = base_hi + UInt64(carry)
    ctr = (lo % UInt32, (lo >> 32) % UInt32, hi % UInt32, (hi >> 32) % UInt32)
    return philox4x32_10(ctr, key)
end

@inline _pair_lo(blk) = (UInt64(blk[2]) << 32) | UInt64(blk[1])
@inline _pair_hi(blk) = (UInt64(blk[4]) << 32) | UInt64(blk[3])

# Mirrors `_fill_u64!` for a `UInt32`-word family (cbrng/cbrng.jl): output
# element `k` (0-based) belongs to Philox block `k >> 1` past the stream's
# current counter, using the low pair of words when `k` is even and the high
# pair when `k` is odd -- the same pairing a scalar `rand(rng)` would produce
# if called `n` times from the same starting position.
function _philox_fill_kernel!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        blk = _philox_block(key, base_hi, base_lo, k >> 1)
        word = isodd(k) ? _pair_hi(blk) : _pair_lo(blk)
        @inbounds A[k + 1] = close_open01(word)
    end
    return nothing
end

# One thread per Philox block, writing up to two normals: Box-Muller applied
# to the block's two uniforms. `1.0 - u1` (rather than `u1`) keeps the `log`
# argument in `(0, 1]`, since `close_open01` can return exactly `0.0`.
function _philox_randn_kernel!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based block index
    if 2b < n
        blk = _philox_block(key, base_hi, base_lo, b)
        u1 = close_open01(_pair_lo(blk))
        u2 = close_open01(_pair_hi(blk))
        r     = sqrt(-2.0 * log(1.0 - u1))
        theta = 2.0 * Float64(pi) * u2
        @inbounds A[2b + 1] = r * cos(theta)
        2b + 2 <= n && (@inbounds A[2b + 2] = r * sin(theta))
    end
    return nothing
end

# Checks alignment, advances `rng.ctr` by `nblocks`, and returns
# (base_hi, base_lo) for the kernel to start from -- the bookkeeping `rand!`
# and `randn!` share.
function _philox_gpu_prep!(fname, rng::PhiloxRNG, nblocks::Integer)
    rng.idx == 5 || throw(ArgumentError(
        "$fname(::PhiloxRNG, ::CuArray) requires a block-aligned generator; " *
        "call reset_substream!(rng) (or start from a fresh stream) first"))
    base = rng.ctr
    rng.ctr = (base + UInt128(nblocks)) & RandomDataStreams._ctr_mask(UInt32, Val(4))
    return (base >> 64) % UInt64, base % UInt64
end

"""
    Random.rand!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with uniforms in `[0, 1)`, one independent Philox block
per pair of output elements, and advance `rng`'s counter on the host by the
number of blocks consumed -- so a later `rand(rng, ...)` or `rand!` call, on
either the CPU or the GPU, continues from a fresh, non-overlapping position.

`rng` must be block-aligned (fresh from [`next_stream!`](@ref), or just after
[`reset_substream!`](@ref)/[`reset_stream!`](@ref)/[`next_substream!`](@ref));
this avoids replicating the CPU path's mid-block tail-buffering on the device.
Calling it on a generator that has already made scalar draws mid-block throws
`ArgumentError` -- call `reset_substream!(rng)` first if that state doesn't
matter to you.
"""
function Random.rand!(rng::PhiloxRNG, A::CuArray{Float64})
    n = length(A)
    n == 0 && return A
    base_hi, base_lo = _philox_gpu_prep!("rand!", rng, cld(n, 2))

    threads = 256
    blocks = cld(n, threads)
    @cuda threads = threads blocks = blocks _philox_fill_kernel!(A, rng.key, base_hi, base_lo, n)
    return A
end

"""
    Random.randn!(rng::PhiloxRNG, A::CuArray{Float64}) -> A

Fill `A` on the device with standard normal draws via Box-Muller, one
independent Philox block feeding one pair of normals, and advance `rng`'s
counter on the host by the number of blocks consumed -- the same bookkeeping
[`rand!`](@ref) uses, so GPU normal and uniform fills on the same stream stay
non-overlapping with each other and with the CPU path.

Box-Muller, not the Ziggurat algorithm the CPU `randn` uses: it takes exactly
two uniforms and produces exactly two normals every time, so every thread
does the same fixed amount of work. Its output is **not** meant to match CPU
`randn` bit for bit -- the algorithms differ.

Same block-alignment requirement as `rand!`; see its docstring.
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

end # module
