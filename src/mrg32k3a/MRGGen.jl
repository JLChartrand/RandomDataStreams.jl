
"""
An object that generates independent random number streams.
"""
mutable struct MRG32k3aGen <: AbstractRNGStream
    seed::Vector{Int64}

    function MRG32k3aGen(x::Vector{Int} = PMF.DEFAULT_SEED)
        @assert(PMF.checkseed(x))
        new(copy(x))
    end
end

# Uniform constructor: a plain integer seed, folded into the valid MRG32k3a
# seed space the same way `Random.seed!` does.
MRG32k3aGen(seed::Integer) = MRG32k3aGen(_mrg_seed_words(UInt64(seed)))

#show imported in RandomDataStreams.jl in src
show(io::IO, rng_gen::MRG32k3aGen) =
    print(io, "MRG32k3aGen(next = ", rng_gen.seed, ")")

function show(io::IO, ::MIME"text/plain", rng_gen::MRG32k3aGen)
    print(io,"Seed for next MRG32k3a generator:\n$(rng_gen.seed)")
end

"""
`get_state` return the seed used to generate non-overlaping MRGs
"""
get_state(rng_gen::MRG32k3aGen) = copy(rng_gen.seed)

"""
    srand!(gen::MRG32k3aGen, seed) -> gen

Resets the seed that the next `next_stream!` call will use.
"""
function srand!(rng_gen::MRG32k3aGen, seed::AbstractVector{<:Integer})
    v = Int.(collect(seed))
    PMF.checkseed(v) || throw(ArgumentError("invalid MRG32k3a seed"))
    rng_gen.seed[:] = v
    return rng_gen
end

srand!(rng_gen::MRG32k3aGen, seed::Integer) = srand!(rng_gen, _mrg_seed_words(UInt64(seed)))

"""
    set_state!(gen::MRG32k3aGen, seed) -> gen

Restores the seed of the next stream, as returned by `get_state(gen)`.
"""
set_state!(rng_gen::MRG32k3aGen, seed) = srand!(rng_gen, seed)

"""
Given an RNG generator object, returns the next RNG stream.
"""
function next_stream!(rng_gen::MRG32k3aGen)
    rng = MRG32k3a(copy(rng_gen.seed))

    rng_gen.seed[1:3] = PMF.MatVecModM(PMF.A1p127, view(rng_gen.seed, 1:3), PMF.m1)
    rng_gen.seed[4:6] = PMF.MatVecModM(PMF.A2p127, view(rng_gen.seed, 4:6), PMF.m2)

    return rng
end
# Deprecated pre-1.0 name (mutates its argument despite lacking `!`).
@deprecate next_stream(rng_gen::MRG32k3aGen) next_stream!(rng_gen)
