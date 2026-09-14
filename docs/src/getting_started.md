# Getting Started

RandomDataStreams.jl provides pseudo-random number generators with support for
**non-overlapping streams and substreams**, following the stream/substream model
of L'Ecuyer et al. (2002).

## Installation

```julia
using Pkg
Pkg.add("RandomDataStreams")
```

or, from a local clone:

```julia
using Pkg
Pkg.develop(path = "path/to/RandomDataStreams.jl")
```

Requirements: Julia ≥ 1.6. The only dependency is the standard library `Random`.

## Choosing a generator

RandomDataStreams.jl implements the full xoshiro/xoroshiro family of Blackman & Vigna, L'Ecuyer's MRG32k3a and MRG63k3a, O'Neill's PCG (the default bit generator of NumPy), and the Philox and Threefry CBRNGs. All variants of a xoshiro family share the same linear
transition and jump constants; only the output scrambler differs.

| Type | Family | State | Period | Scrambler |
|---|---|---|---|---|
| `PhiloxRNG` | Philox4x32-10 | 128-bit counter | 2^130 | Feistel-like network |
| `Philox4x64RNG` | Philox4x64-10 | 128-bit counter | 2^130 | Feistel-like network |
| `Threefry4x64RNG` | Threefry4x64-20 | 128-bit counter | 2^130 | Threefish rounds (add/rot/xor) |
| `Threefry4x32RNG` | Threefry4x32-20 | 128-bit counter | 2^130 | Threefish rounds (add/rot/xor) |
| `Xoroshiro128p` | xoroshiro128 | 128 bits | 2^128 − 1 | `s0 + s1` |
| `Xoroshiro128ss` | xoroshiro128 | 128 bits | 2^128 − 1 | high bits, `**` |
| `Xoroshiro128pp` | xoroshiro128 | 128 bits | 2^128 − 1 | high bits, `++` |
| `Xoshiro256p` | xoshiro256 | 256 bits | 2^256 − 1 | `s0 + s3` |
| `Xoshiro256ss` | xoshiro256 | 256 bits | 2^256 − 1 | high bits, `**` |
| `Xoshiro256pp` | xoshiro256 | 256 bits | 2^256 − 1 | high bits, `++` |
| `Xoshiro512p` | xoshiro512 | 512 bits | 2^512 − 1 | `s0 + s2` |
| `Xoshiro512ss` | xoshiro512 | 512 bits | 2^512 − 1 | high bits, `**` |
| `Xoshiro512pp` | xoshiro512 | 512 bits | 2^512 − 1 | high bits, `++` |
| `MRG32k3a` | combined MRG (L'Ecuyer) | 192 bits | ≈ 2^191 | native `Float64`, 32-bit step |
| `MRG63k3a` | combined MRG (L'Ecuyer) | 384 bits | ≈ 2^377 | native `Float64`, 63-bit step |
| `PCG64` | PCG (O'Neill) | 128 bits | 2^128 | XSL-RR permutation |
| `PCG64DXSM` | PCG (O'Neill) | 128 bits | 2^128 | DXSM permutation |

See [Generator Comparison](comparison.md) for guidance and benchmarks.

## Seeding: one rule for every generator

Whatever generator you pick, there are three ways to seed it, and they mean the
same thing in every family:

```julia
MRG32k3aGen()             # the package default seed, 12345
MRG32k3aGen(20260830)     # an integer seed
MRG32k3aGen([1,2,3,4,5,6])# the family's own seed vector

Xoshiro256ppGen(20260830) # exactly the same three forms
PCG64Gen(20260830)
PhiloxGen(20260830)
```

An **integer** is a *seed*: it is expanded through splitmix64 and folded into
whatever the family accepts, so `T(20260830)` is equivalent to
`Random.seed!(T(), 20260830)` everywhere. A value in the family's **own
representation** — its seed vector, or a `UInt128` for PCG — is taken as the
state or key itself, not hashed.

The same three forms work on the stream types directly (`MRG32k3a(20260830)`,
`Xoshiro256pp(20260830)`, ...). See
[Streams & Substreams](streams.md) for the full interface table.

## First numbers

```julia
using RandomDataStreams

rng = MRG32k3a()      # default seed [12345, ..., 12345]
rand(rng)             # 0.12701112204657714
rand(rng)             # 0.3185275653967945
```

```julia
x = Xoshiro256p(UInt64[1, 2, 3, 4])
rand(x)               # Float64 in [0, 1)
```

## Supported output types

### MRG32k3a

- `Float64` (native), plus `Float32`, `Float16` via conversion
- `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128`
- `Int8`, `Int16`, `Int32`, `Int64`, `Int128` (same bit patterns)
- `Bool`
- Ranges: `rand(rng, 1:10)` works through the standard `Random` machinery

Note: integer outputs of MRG32k3a are built from its native ~32-bit resolution,
and MRG63k3a's from its ~63-bit one; they are not uniform over their full width. Use them for indices, choices and
flags rather than cryptography.

### Xoshiro256p

- `Float64` (native path), `Float32`, `Float16`
- `UInt64`
- Ranges: `rand(rng, 1:10)` works through the standard `Random` machinery

## Reproducibility

Every generator accepts an explicit seed:

```julia
rng = MRG32k3a([42, 1, 2, 3, 4, 5])
x    = Xoshiro256p(UInt64[0xdead, 0xbeef, 0xcafe, 0xbabe])
```

Seeds are validated by `checkseed`, or `checkseed63` for MRG63k3a: a valid seed
of either has 6 non-negative values; the first three must be < m1 and not all zero, the last three < m2 and
not all zero.

The standard Julia interface also works with every generator:

```julia
Random.seed!(rng, 42)          # integer seed (splitmix64 expansion)
Random.seed!(rng, [7,7,7,8,8,8])   # explicit MRG32k3a seed (validated)
```

## Full Random API

RandomDataStreams generators are drop-in substitutes for Julia's built-in RNGs: anywhere a
function accepts an `AbstractRNG`, you can pass an RandomDataStreams generator and keep
your stream/substream control.

```julia
using RandomDataStreams, Random

rng = next_stream!(MRG32k3aGen())     # any RandomDataStreams generator works here

rand(rng, 5)                          # Vector{Float64}, 5 draws
A = rand(rng, Float64, 2, 3)          # 2x3 matrix
z = randn(rng)                        # standard normal
e = randexp(rng)                      # exponential
v = shuffle(rng, collect(1:8))        # shuffled copy
p = randperm(rng, 6)                  # random permutation
buf = Vector{Float64}(undef, 3)
rand!(rng, buf)                       # fill in place
Random.randstring(rng, 10)            # random string
```

Seeding through the standard interface makes results reproducible:

```julia
Random.seed!(rng, 42)
u1 = [rand(rng) for _ in 1:3]
Random.seed!(rng, 42)
u2 = [rand(rng) for _ in 1:3]
u1 == u2                              # true
```

Only `sample` requires StatsBase.jl and is out of scope.

## Next steps

- [Streams & substreams](streams.md)
- [API reference](api.md)
- [Implementation notes](implementation.md)
