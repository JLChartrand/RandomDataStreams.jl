"""
Streamable Random Number Generator, see `AbstractStreamableRNG`
"""

module RandomDataStreams

using Random

import Base: rand, show, copy

export AbstractRNGStream, AbstractStreamableRNG
export checkseed, checkseed63, MRG32k3a, rand, srand!, reset_stream!, reset_substream!, next_substream!, MRG32k3aGen, show, get_state, set_state!, next_stream!
export MRG63k3a, MRG63k3aGen
export Xoshiro256p, Xoshiro256plusGen, short_jump!, long_jump!, advance_state!
# deprecated names kept for compatibility (warn on use)
export srand, short_jump, long_jump, next_stream



###mrg32k3a
include("abstract.jl")
include("floats.jl")
include("mrg32k3a/main.jl")

###mrg63k3a, the 64-bit-arithmetic member of the same family (L'Ecuyer 1999)
include("mrg63k3a/main.jl")

###xoshiro / xoroshiro families (Blackman & Vigna)
include("xoshiro/xos256p.jl")
include("xoshiro/xoshiro256plus_types.jl")

###PCG (O'Neill), the default bit generator of NumPy
include("pcg/pcg.jl")

###counter-based generators (Salmon, Moraes, Dror & Shaw)
include("cbrng/cbrng.jl")
include("philox/philox.jl")
include("threefry/threefry.jl")

###standard Random.seed! interface for every generator
include("seeding.jl")

###normal and exponential variates by inversion, the same bits on CPU and GPU
include("normal.jl")

export Xoroshiro128p, Xoroshiro128ss, Xoroshiro128pp,
       Xoshiro256p, Xoshiro256ss, Xoshiro256pp,
       Xoshiro512p, Xoshiro512ss, Xoshiro512pp,
       Xoroshiro128pGen, Xoroshiro128ssGen, Xoroshiro128ppGen,
       Xoshiro256plusGen, Xoshiro256pGen, Xoshiro256ssGen, Xoshiro256ppGen,
       Xoshiro512pGen, Xoshiro512ssGen, Xoshiro512ppGen



export PCGRNG, PCGGen
export PCG64, PCG64DXSM, PCG64Gen, PCG64DXSMGen

export CBRNG, CBGen
export PhiloxRNG, PhiloxGen
export Philox4x64RNG, Philox4x64Gen
export philox4x32_10, philox4x32_counter, open01, norminv
export Threefry4x64RNG, Threefry4x64Gen, Threefry4x32RNG, Threefry4x32Gen

# GPU-only normal-variate algorithms, kept beside inversion -- which `randn!`
# uses on the CPU and on the GPU alike -- for comparison. Declared here, as
# empty generic functions, so the package always exports a name to dispatch on;
# the CUDA extension (ext/RandomDataStreamsCUDAExt.jl) adds the actual methods
# when CUDA.jl is loaded alongside this package, and its docstrings are the
# ones that apply.
"""
    randn_boxmuller!(rng, A) -> A

Standard-normal fill via Box-Muller on the GPU, two uniforms of one Philox4x32
block to two normals. Faster than `randn!` and branch-free, but it does not
carry antithetic variates or randomized quasi-Monte Carlo through, and it
matches no CPU output. Only defined for `PhiloxRNG`/`CuArray`, by the CUDA
extension -- see its docstring.
"""
function randn_boxmuller! end

"""
    randn_inversion!(rng, A) -> A

Deprecated: `randn!` draws by inversion now, on the CPU and on the GPU, with the
same values on both. Forwards to it, on a `CuArray`, when the CUDA extension is
loaded.
"""
function randn_inversion! end

"""
    randn_polar!(rng, A) -> A

Standard-normal fill via Marsaglia's polar method (accept-reject), kept only
for comparison against inversion (`randn!`) and [`randn_boxmuller!`](@ref).
Only defined for `PhiloxRNG`/`CuArray{Float64}`, by the CUDA extension -- see
its docstring.
"""
function randn_polar! end

export randn_boxmuller!, randn_inversion!, randn_polar!
end # module
