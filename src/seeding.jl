# Standard `Random.seed!` interface for all generators ------------------------------

const _GOLDEN = 0x9E3779B97f4A7C15

"Expand an arbitrary integer seed into n well-distributed 64-bit words (splitmix64)."
function _splitmix_words(seed::UInt64, n::Int)
    out = Vector{UInt64}(undef, n)
    sm = seed
    for i in 1:n
        sm += _GOLDEN
        z = sm
        z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
        out[i] = z ⊻ (z >> 31)
    end
    return out
end

"Expand an integer seed into n words that are never all zero (a forbidden xoshiro state)."
function _seed_words_nonzero(seed::UInt64, n::Int)
    ws = _splitmix_words(seed, n)
    all(iszero, ws) && (ws[1] = _GOLDEN)
    return ws
end

"""
    Random.seed!(rng::LinRNG, seed::Integer) -> rng
    Random.seed!(rng::LinRNG, seed::Vector{UInt64}) -> rng

Re-seed the generator with all three checkpoints equal. Integer seeds are
expanded through splitmix64 (Vigna's recommended initialization).
"""
function Random.seed!(rng::LinRNG{N}, seed::Integer) where {N}
    rng.Cg = rng.Bg = rng.Ig = NTuple{N,UInt64}(_seed_words_nonzero(UInt64(seed), N))
    return rng
end

function Random.seed!(rng::LinRNG{N}, seed::Vector{<:Unsigned}) where {N}
    length(seed) == N || throw(ArgumentError("seed must have $N UInt64 elements"))
    checkseed(seed) || throw(ArgumentError(_ZERO_STATE))
    rng.Cg = rng.Bg = rng.Ig = NTuple{N,UInt64}(seed)
    return rng
end

"Fold an integer seed into the valid MRG32k3a seed space."
function _mrg_seed_words(seed::UInt64)
    w = _splitmix_words(seed, 6)
    v = Vector{Int}(undef, 6)
    for i in 1:3
        v[i] = Int(w[i] % PMF.m1)
    end
    for i in 4:6
        v[i] = Int(w[i] % PMF.m2)
    end
    all(iszero, view(v, 1:3)) && (v[1] = 1)
    all(iszero, view(v, 4:6)) && (v[4] = 1)
    return v
end

"""
    Random.seed!(rng::MRG32k3a, seed::Integer) -> rng
    Random.seed!(rng::MRG32k3a, seed::AbstractVector{<:Integer}) -> rng

Re-seed the generator with all three states equal. Integer seeds are expanded
through splitmix64 and folded into the valid MRG32k3a seed space; vector seeds
must satisfy `checkseed`.
"""
function Random.seed!(rng::MRG32k3a, seed::Integer)
    rng.Cg[:] = rng.Bg[:] = rng.Ig[:] = _mrg_seed_words(UInt64(seed))
    return rng
end

function Random.seed!(rng::MRG32k3a, seed::AbstractVector{<:Integer})
    s = Int.(collect(seed))
    checkseed(s) || throw(ArgumentError("invalid MRG32k3a seed"))
    rng.Cg[:] = rng.Bg[:] = rng.Ig[:] = s
    return rng
end

"Fold an integer seed into the valid MRG63k3a seed space."
function _mrg63_seed_words(seed::UInt64)
    w = _splitmix_words(seed, 6)
    v = Vector{Int}(undef, 6)
    for i in 1:3
        v[i] = Int(w[i] % PMF63.m1)
    end
    for i in 4:6
        v[i] = Int(w[i] % PMF63.m2)
    end
    all(iszero, view(v, 1:3)) && (v[1] = 1)
    all(iszero, view(v, 4:6)) && (v[4] = 1)
    return v
end

"""
    Random.seed!(rng::MRG63k3a, seed::Integer) -> rng
    Random.seed!(rng::MRG63k3a, seed::AbstractVector{<:Integer}) -> rng

Re-seed the generator with all three states equal. Integer seeds are expanded
through splitmix64 and folded into the valid MRG63k3a seed space; vector seeds
must satisfy `checkseed63`.
"""
function Random.seed!(rng::MRG63k3a, seed::Integer)
    # `_step63`: the MRG63k3a state is kept one step ahead of the position,
    # so a seed is converted on the way in. See src/mrg63k3a/mrg63k3a.jl.
    rng.Cg[:] = rng.Bg[:] = rng.Ig[:] = _step63(_mrg63_seed_words(UInt64(seed)))
    return rng
end

function Random.seed!(rng::MRG63k3a, seed::AbstractVector{<:Integer})
    s = Int.(collect(seed))
    checkseed63(s) || throw(ArgumentError("invalid MRG63k3a seed"))
    rng.Cg[:] = rng.Bg[:] = rng.Ig[:] = _step63(s)
    return rng
end

"""
    Random.seed!(rng::CBRNG, seed::Integer) -> rng
    Random.seed!(rng::CBRNG, seed::AbstractVector{<:Integer}) -> rng

Re-seed a counter-based stream: the seed selects the key and the counter is
rewound to the beginning of the stream. Integer seeds are expanded through
splitmix64; vector seeds must hold exactly the key words.
"""
function Random.seed!(rng::CBRNG{B,W,N,K}, seed::Integer) where {B,W,N,K}
    ws = _splitmix_words(UInt64(seed), K)
    return srand!(rng, NTuple{K,W}([w % W for w in ws]))
end

Random.seed!(rng::CBRNG, seed::AbstractVector{<:Integer}) = srand!(rng, seed)

"""
    Random.seed!(rng::PCGRNG, seed::Integer) -> rng
    Random.seed!(rng::PCGRNG, seed::AbstractVector{<:Integer}) -> rng

Re-seed the generator with all three checkpoints equal. Integer seeds are
expanded through splitmix64 into the 128-bit state. Every state is a valid LCG
state, so no seed is rejected.
"""
function Random.seed!(rng::PCGRNG, seed::Integer)
    ws = _splitmix_words(UInt64(seed), 2)
    return srand!(rng, (UInt128(ws[1]) << 64) | UInt128(ws[2]))
end

Random.seed!(rng::PCGRNG, seed::AbstractVector{<:Integer}) = srand!(rng, seed)
