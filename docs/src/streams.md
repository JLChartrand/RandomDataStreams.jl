# Streams & Substreams

The central abstraction of RandomDataStreams.jl is the *stream*: a long, non-overlapping
subsequence of a generator's period. Streams can themselves be split into
*substreams*. This is the architecture recommended by L'Ecuyer et al. (2002)
for parallel and replicated stochastic simulation:

- **Streams** are handed to independent replications or parallel workers;
  their outputs never overlap, so replications are statistically independent.
- **Substreams** delimit scenarios *within* one replication; rewinding a
  substream while keeping the scenario structure enables techniques such as
  common random numbers.

## Why streams? The case for Common Random Numbers (CRN)

The whole point of the stream/substream architecture is **variance reduction
when comparing systems** — arguably the most important technique in stochastic
simulation design (L'Ecuyer et al. 2002; Law, *Simulation Modeling and
Analysis*).

Suppose you compare K configurations of a stochastic system (two inventory
levels, two queue disciplines...). The naive approach runs each configuration
with *independent* random numbers. The variance of the estimated difference is
then

```
Var[D̂] = Var[Ȳ₁]/n + Var[Ȳ₂]/n
```

With **common random numbers**, every configuration of a given replication
consumes the *same* underlying uniforms: the per-replication differences are
strongly positively correlated and

```
Var[D̂_CRN] = Var[Ȳ₁]/n + Var[Ȳ₂]/n − 2·Cov[Ȳ₁, Ȳ₂]/n
```

— often orders of magnitude smaller. But CRN only works if the random number
usage can be *synchronised exactly* across configurations, replication after
replication. That is precisely what RandomDataStreams's machinery guarantees:

| Need | RandomDataStreams tool |
|---|---|
| Replications must be independent | a fresh stream per replication (`next_stream!`) |
| Configurations must see identical randomness | rewind each configuration to the same substream start (`reset_stream!` / `reset_substream!`) |
| No accidental overlap between replications | streams are provably disjoint |

### Measurable example: comparing two inventory levels

Daily cost of an inventory system over 30 days (demands N(100, 20) truncated
at 0), comparing stock level s = 50 vs s = 60:

```julia
using RandomDataStreams, Statistics

function cost(s, rng)
    total, stock = 0.0, Float64(s)
    for _ in 1:30
        stock = min(stock + 40, 200)
        d = max(0.0, 100 + 20randn(rng))
        sold = min(stock, d)
        total += 5(d - sold) + 0.1max(0, stock - sold - 10)
        stock -= sold
    end
    total
end

function replications(crn::Bool, n::Int)
    gen = MRG32k3aGen()
    diffs = Vector{Float64}(undef, n)
    for i in 1:n
        r50 = next_stream!(gen)
        l50 = cost(50, r50)
        r60 = crn ? reset_stream!(copy(r50)) : next_stream!(gen)
        l60 = cost(60, r60)
        diffs[i] = l60 - l50
    end
    return diffs
end

for crn in (true, false)
    ds = replications(crn, 4_000)
    println("CRN = \$crn: mean diff ≈ \$(round(mean(ds), digits=1)), ",
            "var(diff) ≈ \$(round(var(ds), digits=1))")
end
```

Typical output:

```
CRN = true:  mean diff ≈ -49.9, var(diff) ≈ 0.1
CRN = false: mean diff ≈ -39.2, var(diff) ≈ 540775.9
```

The variance of the comparison drops by **six orders of magnitude**: with CRN,
both policies face identical demand sequences, so the observed difference
measures the effect of the policy change rather than the noise of unrelated
demand scenarios. Without CRN you would need ~10⁶ more replications for the
same precision.

This example uses `copy(r50)` + `reset_stream!`: both policies replay the very
same stream from its beginning. `reset_substream!` generalises the pattern
when each replication itself contains several scenarios.

## The two object families

Every generator in the package comes as a pair, and the distinction is the one
the whole interface rests on.

1. **Generator objects** (`<: AbstractRNGStream`) are factories. They hold the
   seed of the *next* stream and hand out streams with `next_stream!`. They
   produce no random numbers themselves.
2. **Streams** (`<: AbstractStreamableRNG`) produce the numbers, and move along
   the stream and substream hierarchy. They are `AbstractRNG`s, so they work
   anywhere Julia's own generators do.

| Family | Generator object | Stream |
|---|---|---|
| MRG32k3a | `MRG32k3aGen` | `MRG32k3a` |
| MRG63k3a | `MRG63k3aGen` | `MRG63k3a` |
| xoroshiro128 | `Xoroshiro128pGen`, `Xoroshiro128ssGen`, `Xoroshiro128ppGen` | `Xoroshiro128p`, `Xoroshiro128ss`, `Xoroshiro128pp` |
| xoshiro256 | `Xoshiro256pGen`, `Xoshiro256ssGen`, `Xoshiro256ppGen` | `Xoshiro256p`, `Xoshiro256ss`, `Xoshiro256pp` |
| xoshiro512 | `Xoshiro512pGen`, `Xoshiro512ssGen`, `Xoshiro512ppGen` | `Xoshiro512p`, `Xoshiro512ss`, `Xoshiro512pp` |
| PCG | `PCG64Gen`, `PCG64DXSMGen` | `PCG64`, `PCG64DXSM` |
| Philox | `PhiloxGen` (4x32-10), `Philox4x64Gen` | `PhiloxRNG`, `Philox4x64RNG` |
| Threefry | `Threefry4x32Gen`, `Threefry4x64Gen` | `Threefry4x32RNG`, `Threefry4x64RNG` |

`Xoshiro256plusGen` is an alias of `Xoshiro256pGen`, kept from earlier
versions.

The interface is the same across the table: whatever the family, a generator
object answers to `next_stream!`, `srand!`, `get_state` and `set_state!`, and a
stream answers to `next_substream!`, `reset_substream!`, `reset_stream!`,
`advance_state!`, `get_state`, `set_state!` and `srand!`. Code written against
the two abstract types never names a family — see the
[API reference](api.md).

## Producing independent streams

```julia
using RandomDataStreams

gen = MRG32k3aGen()             # or Xoshiro256plusGen(UInt64[...])

worker1 = next_stream!(gen)       # stream 1
worker2 = next_stream!(gen)       # stream 2 — guaranteed disjoint from stream 1
worker3 = next_stream!(gen)       # stream 3 — ...
```

Each call to `next_stream!` advances the generator's internal seed by a huge
leap (2^127 values for MRG32k3a, 2^250 for MRG63k3a, a full `long_jump!` for
Xoshiro256+), which is what guarantees non-overlap.

## Threads

Streams share no state, so one stream per thread needs no synchronisation.
Take the streams first, then parallelise over them:

```julia
using RandomDataStreams, Base.Threads

gen  = MRG32k3aGen()                    # any generator object
rngs = next_stream!(gen, nthreads())    # n streams, taken serially

results = Vector{Float64}(undef, nthreads())
@threads for t in 1:nthreads()
    rng = rngs[t]                       # this thread owns this stream
    results[t] = sum(rand(rng) for _ in 1:10^5)
end
```

The results are those of the same streams drawn one after another, which the
test suite checks for every family.

!!! warning "Do not call `next_stream!` inside the parallel loop"

    A generator object holds the seed of the next stream and rewrites it on
    every call. Concurrent calls read the same seed, so the streams handed out
    overlap — and nothing reports it. In a measurement here, 64 threads calling
    `next_stream!` on one shared generator received between 57 and 63 distinct
    streams instead of 64, on three consecutive runs. The failure is silent and
    destroys precisely the guarantee this package exists to provide.

    `next_stream!(gen, n)` exists so that the correct pattern is also the
    shorter one.

## Distributed

Separate processes cannot share stream objects, so the seeds travel instead.
Take them from the generator before each `next_stream!`, and rebuild the stream
on the worker with `srand!`:

```julia
using Distributed
@everywhere using RandomDataStreams

gen   = MRG32k3aGen()
seeds = Vector{Any}(undef, nworkers())
for i in 1:nworkers()
    seeds[i] = get_state(gen)     # seed of the stream about to be produced
    next_stream!(gen)             # advance past it
end

@sync for (i, w) in enumerate(workers())
    seed = seeds[i]
    @spawnat w begin
        rng = MRG32k3a(1)         # any valid seed; replaced on the next line
        srand!(rng, seed)         # this worker's stream, from its beginning
        # ... draw from rng ...
    end
end
```

Use `srand!` here, not `set_state!`. They are not interchangeable: `set_state!`
moves the current position only, leaving the stream and substream boundaries
where the constructor put them, so a later `reset_stream!` on the worker would
rewind to the wrong place. `srand!` sets all three checkpoints, which is what
makes the worker's stream a stream rather than a position in someone else's.

Both are portable across families, as is `get_state` on a generator object; the
representation that travels is opaque, so hand it back unchanged rather than
interpreting it.

## Navigating substreams

Within a single RNG, three functions move along the stream/substream
hierarchy (all return the modified RNG):

| Function | Effect |
|---|---|
| `reset_stream!(rng)` | jump to the very beginning of the current stream |
| `reset_substream!(rng)` | jump to the beginning of the current substream |
| `next_substream!(rng)` | advance to the next substream |

Example — common random numbers across two scenarios:

```julia
rng = next_stream!(MRG32k3aGen())

rand(rng)                 # consume some variates of substream 1...
rand(rng)

next_substream!(rng)      # switch to substream 2 for scenario B
uB = rand(rng)

reset_stream!(rng)        # replay substream 1 from scratch for scenario A
uA = rand(rng)            # same underlying uniforms as the first draw
```

For Xoshiro256+, `next_substream!` is implemented as a `short_jump!`
(2^96-value leap) and `reset_stream!`/`next_substream!` manipulate the saved
`Bg`/`Ig` checkpoints.

## Saving, restoring, jumping

```julia
state = get_state(rng)     # a copy; safe to store
# ... consume numbers ...
```

To restore an MRG32k3a or MRG63k3a state, rebuild the generator with the triple
constructor (`Cg`, `Bg`, `Ig` all set to the snapshot):

```julia
snapshot = get_state(rng)
xs = [rand(rng) for _ in 1:5]
clone = MRG32k3a(snapshot, snapshot, snapshot)
rand(clone) == xs[1]       # true — resumes exactly after the snapshot
```

Both MRG generators additionally support arbitrary jumps inside a stream:

```julia
advance_state!(rng, e, c)
```

moves the state forward by `n` steps, where `n = 2^e + c` (`e` may be negative,
and negative `c` moves backwards). This costs O(log n) matrix operations,
independent of the distance jumped.

## Scope: host-side streams, device-side bijections

The stream object of L'Ecuyer et al. (2002) is stateful by construction — a
current position, a substream anchor, a stream anchor, mutated in place. That
makes it a **host-side** abstraction. A GPU kernel wants the opposite: no state
at all, one value computed from the thread index. The package does not run on
a device, and that is a scope boundary rather than a gap, because the two
halves fit together.

**Counter-based generators: the addressing scheme is the work assignment.** The
key names the stream, the high half of the counter the substream, the low half
the position. Any draw can therefore be recomputed from `(key, substream,
index)` alone, with no object, which is exactly what a kernel needs:

```julia
using RandomDataStreams
const RDS = RandomDataStreams

function direct_word(key::NTuple{2,UInt32}, substream::Integer, i::Integer)
    block, word = divrem(i, 4)                    # four 32-bit words per block
    ctr = (UInt128(substream) << 64) | UInt128(block)
    c = (ctr % UInt32, (ctr >> 32) % UInt32, (ctr >> 64) % UInt32, (ctr >> 96) % UInt32)
    return RDS.philox(c, key)[word + 1]
end
```

This agrees with the stream object draw for draw; the test suite checks it, so
host and device address the same sequence. `philox` and `threefry` are pure,
allocation-free functions of their arguments, which is the necessary condition
for using them inside a kernel — necessary, not sufficient: the package has no
GPU dependency and runs no device tests, so that last step is the user's.

**Recurrence-based generators: the host computes the starting points.** There
is no stateless form here, and the standard pattern runs the other way: use
`next_stream!` on the host to produce as many non-overlapping starting states
as there are tasks, ship one to each, and let each iterate its own recurrence.

```julia
gen = Xoshiro256plusGen(UInt64[1, 2, 3, 4])
seeds = [get_state(next_stream!(gen)) for _ in 1:nworkers]    # non-overlapping
```

The jump machinery is what makes this cheap — matrix jumps for MRG32k3a, GF(2)
polynomial jumps for the xoshiro families — and it is the construction
L'Ecuyer et al. (2021, Sec. 2) describe for parallel environments.

What the package does not provide: filling a `CuArray`, or any device-side
`rand`. If that is what you need, take the bijections and the addressing scheme
above, and keep the stream objects on the host for what they are good at —
assigning non-overlapping work and replaying it identically.

## One interface, every generator

The whole point of the two object families is that code written against them
does not name a generator. Every family in the package answers to the same
calls, with the same meanings — the table below is asserted for all seventeen
generators in the test suite, not just documented.

On the generator object:

| call | meaning |
|---|---|
| `Gen()` | seeded with the package default, `12345` |
| `Gen(12345)` | seeded with an integer |
| `Gen(v)` | seeded with the family's own seed vector |
| `next_stream!(gen)` | the next non-overlapping stream |
| `srand!(gen, seed)` | reset the seed the next `next_stream!` will use |
| `get_state(gen)`, `set_state!(gen, s)` | save and restore it |

On the stream:

| call | meaning |
|---|---|
| `T()`, `T(12345)`, `T(v)` | the same three seeding forms |
| `next_substream!`, `reset_substream!`, `reset_stream!` | navigation; each returns the generator |
| `advance_state!(rng, e, c)` | move by `2^e + c` draws, negative allowed |
| `get_state`, `set_state!` | save and restore the position only |
| `srand!`, `Random.seed!` | reseed, resetting both boundaries |
| `copy`, `show`, the full `Random` API | as for any `AbstractRNG` |

**The seeding rule.** A value in the family's own representation — its seed
vector, or a `UInt128` for PCG — *is* the state or key. Any other integer is a
*seed*, expanded through splitmix64 and folded into whatever the family
accepts. So `MRG32k3aGen(12345)`, `Xoshiro256ppGen(12345)`, `PhiloxGen(12345)`
and `PCG64Gen(12345)` all mean the same kind of thing, and `T(12345)` is
equivalent to `Random.seed!(T(), 12345)` everywhere.

**What is not portable.** `short_jump!` and `long_jump!` are the xoshiro and
PCG spelling of a jump by exactly the substream and stream distance. Use
`next_substream!` in code meant to work with any generator. `long_jump!` has no
meaning at all for a counter-based generator, where moving to another stream
means taking another key — an operation that belongs to the generator object.

## Guarantees

- Streams produced by successive calls to `next_stream!` on the same generator
  object are **provably non-overlapping**.
- Substreams likewise partition a stream into disjoint blocks
  (length ≈ 2^76 steps for MRG32k3a, 2^150 for MRG63k3a).
