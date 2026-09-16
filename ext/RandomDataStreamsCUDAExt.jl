# CUDA extension for RandomDataStreams.
#
# Loaded automatically whenever CUDA.jl is loaded alongside RandomDataStreams
# (Julia's package-extension mechanism, see the [weakdeps]/[extensions]
# sections of Project.toml), so a CPU-only user never pulls in a CUDA
# dependency.
#
# One capability lives here: `rand!` on a `PhiloxRNG` into a `CuArray{Float64}`,
# filling the array with a kernel that gives each output element its own
# independent Philox block -- no per-thread state, no synchronization -- and
# then advancing the stream's counter on the host by the number of blocks
# consumed, exactly as the CPU `_fill_u64!` path does. The per-thread
# primitives (`philox4x32_10`, `philox4x32_counter`) that a kernel would use
# directly are plain functions in src/philox/philox.jl and need no CUDA-specific
# wrapping at all.
module RandomDataStreamsCUDAExt

using CUDA
using Random: Random
using RandomDataStreams: RandomDataStreams, PhiloxRNG, philox4x32_10, close_open01

# Mirrors `_fill_u64!` for a `UInt32`-word family (cbrng/cbrng.jl): output
# element `k` (0-based) belongs to Philox block `k >> 1` past the stream's
# current counter, using the low pair of words when `k` is even and the high
# pair when `k` is odd -- the same pairing a scalar `rand(rng)` would produce
# if called `n` times from the same starting position.
function _philox_fill_kernel!(A, key::NTuple{2,UInt32}, base_hi::UInt64, base_lo::UInt64, n::Int)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1     # 0-based output index
    if 0 <= k < n
        b = k >> 1
        lo, carry = Base.add_with_overflow(base_lo, UInt64(b))
        hi = base_hi + UInt64(carry)
        ctr = (lo % UInt32, (lo >> 32) % UInt32, hi % UInt32, (hi >> 32) % UInt32)
        blk = philox4x32_10(ctr, key)
        word = isodd(k) ? (UInt64(blk[4]) << 32) | UInt64(blk[3]) :
                           (UInt64(blk[2]) << 32) | UInt64(blk[1])
        @inbounds A[k + 1] = close_open01(word)
    end
    return nothing
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
    rng.idx == 5 || throw(ArgumentError(
        "rand!(::PhiloxRNG, ::CuArray) requires a block-aligned generator; " *
        "call reset_substream!(rng) (or start from a fresh stream) first"))

    base = rng.ctr
    base_hi = (base >> 64) % UInt64
    base_lo = base % UInt64

    threads = 256
    blocks = cld(n, threads)
    @cuda threads = threads blocks = blocks _philox_fill_kernel!(A, rng.key, base_hi, base_lo, n)

    nblocks = cld(n, 2)
    rng.ctr = (base + UInt128(nblocks)) & RandomDataStreams._ctr_mask(UInt32, Val(4))
    return A
end

end # module
