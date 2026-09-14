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
export Threefry4x64RNG, Threefry4x64Gen, Threefry4x32RNG, Threefry4x32Gen
end # module
