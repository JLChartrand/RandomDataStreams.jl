# Choosing a generator

RandomDataStreams.jl ships two kinds of generator. They differ in how a stream
is defined, which is what makes them suited to different jobs — throughput is
the smaller part of the choice.

**Recurrence-based** generators (MRG32k3a and MRG63k3a, the xoshiro/xoroshiro
families) hold a state and a transition function. There is one orbit, and a
stream is a *segment* of it: `next_stream!` jumps a fixed, huge distance along
the orbit, and non-overlap is a statement about those distances.

**Counter-based** generators (Philox, Threefry) hold a counter and a key, and
produce a block as a keyed bijection of the counter, `R = b_K(C)`. A stream is
a *key*: different streams use different bijections, so their outputs cannot
overlap by construction rather than by distance. Position within a stream is an
index, not a state, so any draw can be recomputed from `(key, substream, i)`
alone — see [Streams & Substreams](streams.md).

| | Recurrence-based | Counter-based |
|---|---|---|
| Stream = | a segment of one orbit | a distinct key |
| Non-overlap because | the jump distance exceeds any usable run | distinct keys are distinct bijections |
| Arbitrary jump | polynomial or matrix computation | one counter addition |
| Draw at index `i` | requires jumping there | computable directly |
| Speed | ~5× faster per draw | block of `N` words amortises the bijection |

## Recurrence-based generators

| Property | MRG32k3a | MRG63k3a | PCG64 | Xoroshiro128* | Xoshiro256* | Xoshiro512* |
|---|---|---|---|---|---|---|
| State (bits) | 192 (6 × Int64) | 384 (6 × Int64) | 128 | 128 | 256 | 512 |
| Period | ≈ 2^191 | ≈ 2^377 | 2^128 | 2^128 − 1 | 2^256 − 1 | 2^512 − 1 |
| Native output | `Float64` [0,1), ~32-bit resolution | `Float64` (0,1), ~63-bit resolution | `UInt64` | `UInt64` | `UInt64` | `UInt64` |
| `Float64` throughput | 236 M/s | 196 M/s | 504 M/s (DXSM 561) | ≈ 599–697 M/s | ≈ 688–788 M/s | ≈ 523–570 M/s |
| `UInt64` throughput | 47 M/s (4 steps) | 93 M/s (2 steps) | 599 M/s (DXSM 743) | ≈ 818–941 M/s | ≈ 1032–1242 M/s | ≈ 725–798 M/s |
| Stream mechanism | matrix power A^2^127 on the seed | matrix power A^2^250 on the seed | closed-form LCG jump (2^32(2^64+1)+1) | `long_jump!` polynomial (2^96) | `long_jump!` (2^192) | `long_jump!` (2^384) |
| Substream mechanism | matrix power A^2^76 | matrix power A^2^150 | closed-form (2^64+1) | `short_jump!` (2^64) | `short_jump!` (2^128) | `short_jump!` (2^256) |
| Backward jumps | yes (`advance_state!`) | yes (`advance_state!`) | yes, same cost as forward | yes (`advance_state!`, GF(2) polynomials) | yes (`advance_state!`) | yes (`advance_state!`) |
| Arbitrary-jump cost | O(e) matrix products | O(e) matrix products | **O(log n) multiplies** | O(deg²) GF(2) ops (~10–500 ms) | same | same |
| Equidistribution | well analysed theory | well analysed theory | none published | 1-dim (**/++) / 0-dim (+) | 3-dim (**/++) / 2-dim (+) | 7-dim (**/++) / 6-dim (+) |
| Parallelism scale advised | large | very large | mild (2^32 streams) | mild (≈ 2^32 streams) | very large | extreme |

(`*` means any of the `+`, `**`, `++` scramblers. `PCG64DXSM` has the same
structure as `PCG64`, with a different output permutation and multiplier.)

## Counter-based generators

| Property | Philox4x32-10 | Philox4x64-10 | Threefry4x32-20 | Threefry4x64-20 |
|---|---|---|---|---|
| Block | 4 × `UInt32` | 4 × `UInt64` | 4 × `UInt32` | 4 × `UInt64` |
| Key (bits) → streams | 64 → 2^64 | 128 → 2^128 | 128 → 2^128 | 256 → 2^256 |
| Counter (bits) | 128 | 128 | 128 | 128 |
| Substreams per stream | 2^64 | 2^64 | 2^64 | 2^64 |
| Words per substream | 2^66 × 32 bits | 2^66 × 64 bits | 2^66 × 32 bits | 2^66 × 64 bits |
| Arithmetic | wide multiply | wide multiply | add–rotate–xor only | add–rotate–xor only |
| `Float64` throughput | 93 M/s | 210 M/s | 84 M/s | 147 M/s |
| `Float64` via `rand!` | 157 M/s | 359 M/s | 100 M/s | 207 M/s |
| Jump to any position | O(1) counter arithmetic | O(1) | O(1) | O(1) |
| Output uniformity | exact (bijection) | exact | exact | exact |
| BigCrush | passes (Salmon et al. 2011) | passes | passes | passes |

Throughput figures come from the [Implementation Notes](implementation.md),
which state the measurement method and the machine; read them as ratios.

## When to choose MRG32k3a

- You need **provably non-overlapping streams with published, peer-reviewed
  parameters** (L'Ecuyer et al. 2002), e.g. to stay compatible with simulation
  libraries built on that model (SSJ, Simio, Arena-style stream ids...).
- Your application consumes **floats only**: MRG32k3a emits them natively. Its
  wide integers are assembled from 16-bit chunks and cost four MRG steps.
- You can afford ~3× slower generation than xoshiro.

## When to choose MRG63k3a

Same author, same construction, same stream semantics, sized for 64-bit
arithmetic (L'Ecuyer 1999, Table II, fourth entry):

- You want the **MRG stream model but your run consumes integers**. A `UInt64`
  is two steps instead of four, and the 32-bit chunks it is built from depart
  from uniformity by `2^-31` where MRG32k3a's 16-bit chunks depart by `2^-16`.
- You want the **longer period** — `2^377` against `2^191` — and the room it
  buys: `2^127` streams of `2^100` substreams of `2^150` numbers.
- You are **not** bound to the published MRG32k3a stream ids. MRG63k3a's
  jump distances are this package's own choice; nothing else implements them,
  so a run cannot be replayed stream-for-stream against another library.
- Prefer MRG32k3a if the run is float-heavy: on `Float64` the 63-bit generator
  is about 17% slower (196 against 236 M/s), because the step costs a 128-bit
  multiply. It is ahead on every integer type — 93 against 47 M `UInt64`/s,
  199 against 95 M `UInt32`/s, and 184 against 163 M `Float64`/s under `rand!`
  — and ahead per random *bit* everywhere.

## When to choose a xoshiro/xoroshiro variant

- You want **maximum throughput** or raw 64-bit integers.
- Choose your state size by parallelism needs:
  - `Xoroshiro128*`: embedded/GPU or few workers only — its 2^64 substreams
    per 2^32 streams are enough for mild parallelism. Note `Xoroshiro128p`
    has a mild Hamming-weight dependency after ~5 TB of output; prefer
    `ss`/`pp`.
  - `Xoshiro256*`: the sweet spot for general use (Julia itself uses
    xoshiro256++ as its default RNG). Use `+` if you generate *only*
    floating-point numbers from the high bits; otherwise prefer `++`/`**`,
    whose outputs are fully equidistributed.
  - `Xoshiro512*`: essentially never needed for period reasons — any period
    ≥ 2^256 is beyond every imaginable need — but handy when you want many
    more independent substreams than 2^128 without changing design.

## When to choose PCG64

- You are **porting or checking a NumPy pipeline**. `PCG64` is what
  `numpy.random.default_rng()` gives you, and the raw 64-bit outputs here match
  NumPy's exactly for the same state.
- You want the **fastest array fill** in the package: with a single 128-bit
  word of state, `rand!` reaches 632 M `Float64`/s for `PCG64DXSM`, ahead of
  every xoshiro. On scalar draws it is slower than xoshiro, at around 500 M/s.
- You need **cheap jumps to arbitrary positions**. The LCG advance is
  closed-form, so `advance_state!(rng, e, c)` costs `O(log n)` multiplications
  at any distance, forwards or backwards — against tens to hundreds of
  milliseconds for the equivalent xoshiro jump.
- Do **not** choose it for large-scale parallelism: with a 2^128 period it
  offers 2^32 streams, the same order as `Xoroshiro128*`, and there is no
  equidistribution theory for it. Note also that the stream distances here are
  odd rather than powers of two — for an LCG that is a correctness requirement,
  not a style choice; see the [Implementation Notes](implementation.md). For many streams, use MRG32k3a, a
  256-bit xoshiro, or a counter-based generator.
- Note what this package does *not* do: PCG's own increment-based "streams".
  See the [FAQ](faq.md).

## When to choose Philox or Threefry

- You need to **address a draw without owning the generator that produced it**:
  a GPU kernel, a distributed worker, or a re-run that must reproduce replicate
  `i` of stream `s` without replaying anything. This is the property no
  recurrence-based generator has.
- You want a **very large number of streams** with no jump computation at all:
  `next_stream!` on a counter-based generator is an increment.
- You are matching another implementation of the same bijection — Random123,
  cuRAND, TensorFlow, Intel MKL all ship Philox, and the outputs here agree
  bit-for-bit with the Salmon et al. test vectors.
- Prefer **Philox4x64-10** as the default counter-based choice: it is the
  fastest of the four here and the widest-deployed. Prefer **Threefry** when
  the target hardware has no fast integer multiply, or when you want an
  add–rotate–xor construction for auditability; the same code reproduces
  bit-for-bit anywhere.
- Do not choose a counter-based generator for raw single-thread speed: on
  scalar `Float64` draws they are 3–8× slower than xoshiro. Filling arrays with
  `rand!` recovers much of that (Philox4x64-10 gains a factor of 1.7) because
  a whole block lands in the array at once.

## Scrambler choice within a xoshiro family

| Scrambler | Output | Best for |
|---|---|---|
| `+`  | sum of two state words | float-only workloads using the top bits; lowest 3 bits have low linear complexity |
| `**` | `rotl(s2·5, 7)·9` | all-purpose; maximal equidistribution (e.g. .NET/Lua default) |
| `++` | `rotl(s1+s4, k)+s1` | all-purpose; Rust's `SmallRng`, Julia's default |

All three share the same transition and therefore the same jump constants:
streams/substreams behave identically across scramblers of one family.

## Related work in the Julia ecosystem

Most Julia random-number packages live under the
[JuliaRandom](https://github.com/JuliaRandom) organisation, and they solve a
different problem from this one. They are collections of *generators*;
`RandomDataStreams.jl` is a *stream layer* that happens to need generators to
apply itself to.

| Package | What it provides | Relation to this package |
|---|---|---|
| [RandomNumbers.jl](https://github.com/JuliaRandom/RandomNumbers.jl) | PCG, the xorshift family, MT19937 behind `AbstractRNG` | overlapping generators, no stream or substream operations; jump-ahead only for PCG (`advance!`) |
| [Random123.jl](https://github.com/JuliaRandom/Random123.jl) | Philox, Threefry, ARS, AESNI | overlapping generators; counter positioning via `set_counter!`, no stream model |
| [StableRNGs.jl](https://github.com/JuliaRandom/StableRNGs.jl) | a Lehmer LCG with output stable across Julia versions | complementary; its "stable streams" mean version stability, not a partition of the period |
| [RandomExtensions.jl](https://github.com/JuliaRandom/RandomExtensions.jl) | distribution objects and `make` specifications for `rand` | orthogonal: it extends what you draw, not where you draw it from |
| [VSL.jl](https://github.com/JuliaRandom/VSL.jl) | bindings to Intel MKL's Vector Statistics Library | MKL offers `vslSkipAheadStream` and `vslLeapfrogStream`; the bindings expose `vslNewStream`/`vslNewStreamEx` only |
| [RNGTest.jl](https://github.com/JuliaRandom/RNGTest.jl) | the TestU01 interface | the same C library we validate against, wrapped; we call it directly instead, for the reasons in [Validation](validation.md) |

The practical consequence: if you need a fast generator, any of these will
serve. If you need `n` streams that are provably disjoint, each rewindable to
its own checkpoint, with the same four calls whatever the underlying family,
that is what this package adds. Should you already depend on one of them, note
that the generators here are validated against the same original references —
the Salmon et al. test vectors for Philox and Threefry, the
`xoshiro.di.unimi.it` C code for the xoshiro families, and NumPy for both PCG
variants — so from an identical state they produce identical output. Seeding
conventions differ between packages, however, so matching a run means setting
the state, not reusing the seed.

## Practical notes

- Seeding a recurrence-based generator: seed states must not be all-zero. For
  hash-like seeds, initialize with splitmix64 outputs (see Vigna's
  recommendation). A counter-based generator has no forbidden key in this
  package — Philox and Threefry are safe for consecutive small keys — but a
  family whose bijection is weak for structured keys must supply a hashing
  `stream_key`; see [Implementation Notes](implementation.md).
- The package's jumps are **anchored at stream boundaries** (`Cg ← Bg ← Ig`),
  matching the L'Ecuyer stream model: `next_substream!` always lands at the
  same place regardless of how far you consumed in the current substream,
  which makes scenario replay reproducible. Counter-based generators obey the
  same contract, with the counter's high bits playing the role of `Bg`.
- Every generator here supports `advance_state!(rng, e, c)` with negative
  distances, so backward jumping is not a differentiator — only its cost is.
- Statistical testing of every generator is described in
  [Validation](validation.md).
