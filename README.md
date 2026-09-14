# RandomDataStreams.jl

[![CI](https://github.com/JLChartrand/RandomDataStreams.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JLChartrand/RandomDataStreams.jl/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/JLChartrand/RandomDataStreams.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JLChartrand/RandomDataStreams.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jlchartrand.github.io/RandomDataStreams.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

**Streamable pseudo-random number generators for Julia.**

RandomDataStreams (Random Data Streams) provides random number generators (RNGs) that support
**non-overlapping streams and substreams**, in the sense of L'Ecuyer et al. (2002).
This is a key requirement for stochastic simulation, parallel Monte Carlo, and
reproducible variance-reduction techniques such as common random numbers.

Four generator families are provided, the first of them in two sizes:

| Family | Variants | Period | Native output | Streams / substreams |
|-------------|---------------|------|----------------------|----------------|
| **MRG32k3a** | `MRG32k3a` | ≈ 2^191 | `Float64` (32-bit step) | matrix jumps (L'Ecuyer) |
| **MRG63k3a** | `MRG63k3a` | ≈ 2^377 | `Float64` (63-bit step) | matrix jumps, 2^250 / 2^150 |
| **xoshiro/xoroshiro** | `Xoroshiro128p/ss/pp`, `Xoshiro256p/ss/pp`, `Xoshiro512p/ss/pp` | 2^128 – 2^512 | `UInt64` | Vigna's jump polynomials |
| **PCG** | `PCG64`, `PCG64DXSM` | 2^128 | `UInt64` | closed-form LCG jumps (O'Neill) |
| **Philox** | `PhiloxRNG` (4x32-10), `Philox4x64RNG` (4x64-10) | 2^130 | `UInt32` / `UInt64` | distinct keys / counter jumps (Salmon et al.) |
| **Threefry** | `Threefry4x64RNG` (4x64-20), `Threefry4x32RNG` (4x32-20) | 2^130 | `UInt64` / `UInt32` | distinct keys / counter jumps (Salmon et al.) |

All xoshiro/xoroshiro variants are validated byte-for-byte against the
original C implementations from [xoshiro.di.unimi.it](http://xoshiro.di.unimi.it),
Philox and Threefry against the test vectors of Salmon et al. (2011), and both
PCG variants against NumPy, whose `default_rng()` is PCG64.
See the [documentation](docs/) for a detailed comparison.

Note that PCG's own increment-based "streams" are deliberately not exposed:
they are not known to be independent, and streams here come from the
closed-form LCG jump instead. The [FAQ](docs/src/faq.md) explains why.

## Features

- **Multiple independent streams**: obtain guaranteed non-overlapping sequences
  with `next_stream!(gen)` — ideal for parallel workers or replicated experiments.
- **Ready for threads**: `next_stream!(gen, n)` hands out `n` streams at once;
  streams share no state, so one per thread needs no synchronisation.
- **Substreams** within each stream (`reset_substream!`, `next_substream!`),
  enabling common random numbers across scenarios — for *every* generator.
- **Full state control**: save/restore a generator with `get_state`,
  rewind with `reset_stream!`, and jump to any position — forward *or
  backward* — with `advance_state!(rng, e, c)` on **every** generator.
- **Standard `Random` API integration**: full drop-in substitutability with
  Julia's built-in RNGs — `rand`/`rand!` on scalars, arrays and ranges,
  `randn`, `randexp`, `shuffle`, `randperm`, `randsubseq`, `Random.seed!(rng,
  seed)` all work with every generator (details per generator in the docs).
- **Zero-allocation hot paths**: all generators produce numbers without heap
  allocation.

## Installation

```julia
using Pkg
Pkg.add("RandomDataStreams")
```

Requires Julia ≥ 1.6. The only dependency is the Julia standard library `Random`.

## Quick start

### MRG32k3a

```julia
using RandomDataStreams

gen = MRG32k3aGen()          # stream generator (manages non-overlapping seeds)
rng = next_stream!(gen)       # a fresh, independent stream

rand(rng)                    # Float64 in [0, 1)
rand(rng, UInt64)            # raw 64-bit unsigned integer
rand(rng, Int32)
rand(rng, 1:10)              # random number in 1:10
```

### MRG63k3a

Same generator family in 64-bit arithmetic: L'Ecuyer's (1999) two moduli just
under 2^63 instead of 2^32, period ≈ 2^377, and just under 63 random bits per
step instead of 32. Use it when the draws are integers — a `UInt64` costs two
steps here against four for MRG32k3a — or when a longer period matters. The
interface is the same, and so is the seed vector.

```julia
using RandomDataStreams

gen = MRG63k3aGen()          # streams 2^250 apart, substreams 2^150
rng = next_stream!(gen)

rand(rng)                    # Float64 in (0, 1)
rand(rng, UInt64)            # two steps, two 32-bit chunks
```

### Philox

```julia
using RandomDataStreams

gen = PhiloxGen()            # Philox4x32-10
rng = next_stream!(gen)

rand(rng)                    # Float64 in [0, 1)
rand(rng, UInt32)            # one 32-bit word of the current block
rand(rng, UInt64)            # raw 64-bit unsigned integer

gen64 = Philox4x64Gen()      # Philox4x64-10: 64-bit words, no bit assembly
rand(next_stream!(gen64), UInt64)
```

### Threefry

```julia
using RandomDataStreams

gen = Threefry4x64Gen()      # Threefry4x64-20, recommended on CPUs
rng = next_stream!(gen)

rand(rng)                    # Float64 in [0, 1)
rand(rng, UInt64)            # raw 64-bit unsigned integer
```

### Xoshiro256+

```julia
using RandomDataStreams

gen = Xoshiro256plusGen([0x01, 0x02, 0x03, 0x04])
rng = next_stream!(gen)

rand(rng)                    # Float64 in [0, 1)
rand(rng, 1:100)             # uniformly distributed Int64 in the range
```

### Streams and substreams

```julia
gen  = MRG32k3aGen()
rng1 = next_stream!(gen)     # stream 1
rng2 = next_stream!(gen)     # stream 2 — provably non-overlapping with stream 1

# Substreams inside rng1
u0 = rand(rng1)
next_substream!(rng1)        # move to the next substream
reset_substream!(rng1)       # back to the start of the current substream
reset_stream!(rng1)          # back to the very beginning of the stream
@assert rand(rng1) == u0
```

### One stream per thread

```julia
using RandomDataStreams, Base.Threads

gen  = MRG32k3aGen()
rngs = next_stream!(gen, nthreads())   # take the streams serially, first

totals = Vector{Float64}(undef, nthreads())
@threads for t in 1:nthreads()
    rng = rngs[t]                      # each thread owns one stream
    totals[t] = sum(rand(rng) for _ in 1:10^4)
end
```

Streams share no state, so this needs no synchronisation and gives exactly what
the same streams give drawn one after another. Do **not** call `next_stream!`
on a shared generator object inside the loop: the generator rewrites the seed
of the next stream on every call, and concurrent calls hand out overlapping
streams without reporting it.

### Drop-in use with Julia's standard RNG API

```julia
using RandomDataStreams, Random

rng = next_stream!(MRG32k3aGen())   # any RandomDataStreams generator works as an AbstractRNG

rand(rng, 5)                        # Vector{Float64}
A = rand(rng, Float64, 2, 3)        # matrix
z = randn(rng)                      # standard normal
v = shuffle(rng, collect(1:8))
p = randperm(rng, 6)
buf = zeros(3); rand!(rng, buf)

Random.seed!(rng, 42)               # standard seeding, reproducible runs
```

### Saving and restoring state

```julia
rng = next_stream!(MRG32k3aGen())
state = get_state(rng)               # copy of the current state
xs = [rand(rng) for _ in 1:5]
rng2 = MRG32k3a(state, state, state) # restore into a new generator
@assert rand(rng2) == xs[1]          # continues exactly where the snapshot was taken
```

### Jumping within a stream (MRG32k3a)

```julia
rng = MRG32k3a()
advance_state!(rng, 10, -3)  # jumps n = 2^10 - 3 = 1021 steps forward
```

## Documentation

Full documentation lives in [`docs/`](docs/) and as a PDF in
[`docs/RandomDataStreams.pdf`](docs/RandomDataStreams.pdf), regenerated with
`julia --project=docs docs/make.jl pdf`:

- [Getting started](docs/src/getting_started.md)
- [Streams & substreams](docs/src/streams.md)
- [API reference](docs/src/api.md)
- [Implementation notes](docs/src/implementation.md)
- [Validation](docs/src/validation.md)
- [Generator comparison](docs/src/comparison.md)

A runnable tour — the stream model, every generator the package ships, and a
common random numbers experiment — is in
[`notebooks/streams_tour.ipynb`](notebooks/streams_tour.ipynb).

## References

### MRG32k3a, MRG63k3a & Stream API
- L'Ecuyer, P. (1999). *Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators*. Operations Research, 47(1), 159–164. (MRG32k3a is the third entry of its Table II, MRG63k3a the fourth.)
- L'Ecuyer, P., Simard, R., Chen, E. J., & Kelton, W. D. (2002). *An Object-Oriented Random-Number Package with Many Long Streams and Substreams*. Operations Research, 50(6), 1073–1075.

### Multiple streams in parallel environments
- L'Ecuyer, P., Nadeau-Chamard, O., Chen, Y.-F., & Lebar, J. (2021). *Multiple Streams with Recurrence-Based, Counter-Based, and Splittable Random Number Generators*. Proceedings of the 2021 Winter Simulation Conference (WSC).

### xoshiro / xoroshiro
- Blackman, D., & Vigna, S. (2021). *Scrambled Linear Pseudorandom Number Generators*. ACM Transactions on Mathematical Software, 47(4), 1-32.

### Philox & Threefry
- Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). *Parallel random numbers: as easy as 1, 2, 3*. SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis.

## Contributing and support

Questions, bug reports and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
says how to report a problem, how to run the batteries and benchmarks, and what
a change has to satisfy — in particular the stream contract every generator
obeys and the external reference values a new generator needs. Participation is
governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

Changes between releases are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE.md](LICENSE.md).
