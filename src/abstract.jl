"""
`AbstractRNGStream` is an abstraction for creating RNG objects with non-overlaping random numbers.
"""
abstract type AbstractRNGStream end
"""
`AbstractStreamableRNG` is a special kind of RNG where one can move between different stream of number that are known to be non-overlapping.

Every concrete subtype implements the same interface, so that code written
against `AbstractStreamableRNG` works with any generator of the package:

| operation | meaning |
|---|---|
| `next_substream!(rng)` | move to the start of the next substream |
| `reset_substream!(rng)` | rewind to the start of the current substream |
| `reset_stream!(rng)` | rewind to the start of the current stream |
| `advance_state!(rng, e, c)` | move by `n` draws, `n` given by `(e, c)`; `n < 0` moves backwards |
| `get_state(rng)` | current position, in a generator-specific representation |
| `set_state!(rng, state)` | restore a position obtained from `get_state` |
| `srand!(rng, seed)` | reseed, resetting the stream and substream boundaries |

The unit of `advance_state!` is one `rand(rng)` draw, for every generator, so
that a jump written against `AbstractStreamableRNG` means the same thing
everywhere.
`get_state` and `set_state!` are inverses — `set_state!(rng, get_state(rng))`
leaves `rng` unchanged — and they move only the current position, never the
stream or substream boundaries. The representation returned by `get_state`
differs between families, so generic code should treat it as opaque and only
ever hand it back to `set_state!`.

Streams themselves are produced by an [`AbstractRNGStream`](@ref) object, which
implements the same operations on its side:

| operation | meaning |
|---|---|
| `Gen()` | a generator seeded with the package default, `12345` |
| `Gen(seed)` | seeded with an integer, or with the family's own seed vector |
| `next_stream!(gen)` | the next non-overlapping stream |
| `next_stream!(gen, n)` | the next `n` of them, as a vector |
| `srand!(gen, seed)` | reset the seed the next `next_stream!` will use |
| `get_state(gen)` / `set_state!(gen, s)` | save and restore that seed |

An integer seed is expanded through splitmix64 and folded into whatever the
family accepts, so `MRG32k3aGen(12345)`, `Xoshiro256ppGen(12345)`,
`PhiloxGen(12345)` and `PCG64Gen(12345)` all mean the same kind of thing.

A generator object is a factory, and it is **not thread-safe**: it rewrites
the seed of the next stream on every call, so concurrent calls to
`next_stream!` hand out overlapping streams, silently. Take the streams first,
with `next_stream!(gen, n)`, then parallelise over them; streams themselves
share no state and need no synchronisation.

Not part of this contract: `short_jump!` and `long_jump!`, which are the
xoshiro and PCG spelling of a jump by the substream and stream distance. Use
`next_substream!` for portable code. `long_jump!` has no meaning for a
counter-based generator at all, where moving to another stream means taking
another key, an operation that belongs to the generator object.
"""
abstract type AbstractStreamableRNG <: AbstractRNG end # object of subtype of AbstractStreamableRNG are StreamableRNG
"""
    next_stream!(gen, n::Integer) -> Vector

Return `n` successive non-overlapping streams from `gen`, in order.

This is the thread-safe way to obtain streams for parallel work. A generator
object holds the seed of the next stream and rewrites it on every call, so
calling `next_stream!(gen)` from several threads at once is a data race: the
threads read the same seed and hand out streams that overlap. The failure is
silent — the result is not an error but a set of correlated streams, which is
exactly the guarantee the package exists to provide.

Call this once, from one thread, and give each worker its own element:

```julia
using Base.Threads

gen  = MRG32k3aGen()
rngs = next_stream!(gen, nthreads())     # serial, here

@threads for t in 1:nthreads()
    r = rngs[t]                          # each thread owns one stream
    ...
end
```

Streams themselves carry no shared state, so drawing from different streams on
different threads needs no synchronisation.
"""
function next_stream!(gen::AbstractRNGStream, n::Integer)
    n >= 0 || throw(ArgumentError("n must be non-negative, got $n"))
    return [next_stream!(gen) for _ in 1:n]
end
