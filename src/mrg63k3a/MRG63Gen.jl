"""
An object that generates independent MRG63k3a streams, `2^250` numbers apart.
"""
mutable struct MRG63k3aGen <: AbstractRNGStream
    seed::Vector{Int64}

    function MRG63k3aGen(x::Vector{Int} = PMF63.DEFAULT_SEED)
        @assert(PMF63.checkseed(x))
        new(copy(x))
    end
end

# Uniform constructor: a plain integer seed, folded into the valid MRG63k3a
# seed space the same way `Random.seed!` does.
MRG63k3aGen(seed::Integer) = MRG63k3aGen(_mrg63_seed_words(UInt64(seed)))

#show imported in RandomDataStreams.jl in src
show(io::IO, rng_gen::MRG63k3aGen) =
    print(io, "MRG63k3aGen(next = ", rng_gen.seed, ")")

function show(io::IO, ::MIME"text/plain", rng_gen::MRG63k3aGen)
    print(io, "Seed for next MRG63k3a generator:\n$(rng_gen.seed)")
end

"""
    get_state(gen::MRG63k3aGen) -> Vector{Int}

The seed the next `next_stream!` will use; restore it with `set_state!`.
"""
get_state(rng_gen::MRG63k3aGen) = copy(rng_gen.seed)

"""
    srand!(gen::MRG63k3aGen, seed) -> gen

Resets the seed that the next `next_stream!` call will use.
"""
function srand!(rng_gen::MRG63k3aGen, seed::AbstractVector{<:Integer})
    v = Int.(collect(seed))
    PMF63.checkseed(v) || throw(ArgumentError("invalid MRG63k3a seed"))
    rng_gen.seed[:] = v
    return rng_gen
end

srand!(rng_gen::MRG63k3aGen, seed::Integer) = srand!(rng_gen, _mrg63_seed_words(UInt64(seed)))

"""
    set_state!(gen::MRG63k3aGen, seed) -> gen

Restores the seed of the next stream, as returned by `get_state(gen)`.
"""
set_state!(rng_gen::MRG63k3aGen, seed) = srand!(rng_gen, seed)

"""
Given an RNG generator object, returns the next RNG stream.
"""
function next_stream!(rng_gen::MRG63k3aGen)
    rng = MRG63k3a(copy(rng_gen.seed))

    rng_gen.seed[1:3] = PMF63.MatVecModM(PMF63.A1p250, view(rng_gen.seed, 1:3), PMF63.m1)
    rng_gen.seed[4:6] = PMF63.MatVecModM(PMF63.A2p250, view(rng_gen.seed, 4:6), PMF63.m2)

    return rng
end
