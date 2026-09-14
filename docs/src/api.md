# API Reference

All names below are exported by the `RandomDataStreams` module unless marked *(internal)*.

## Types

**`AbstractRNGStream`** — supertype of stream *generators*: objects that mint
new independent streams. `MRG32k3aGen`, `MRG63k3aGen`, the nine `Xoshiro*Gen`/`Xoroshiro*Gen`,
`PCG64Gen`, `PCG64DXSMGen`, `PhiloxGen`, `Philox4x64Gen`, `Threefry4x32Gen`,
`Threefry4x64Gen`.

**`AbstractStreamableRNG <: Random.AbstractRNG`** — supertype of usable RNGs
that navigate streams and substreams. `MRG32k3a`, `MRG63k3a`, the nine xoshiro/xoroshiro
variants, `PCG64`, `PCG64DXSM`, `PhiloxRNG`, `Philox4x64RNG`,
`Threefry4x32RNG`, `Threefry4x64RNG`.

---

## The common interface

Everything in this section works on **every** generator the package ships, with
the same meaning. It is asserted for all seventeen in the test suite, so code
written against `AbstractStreamableRNG` never has to name a family. The
per-family sections below add only what is specific.

### Seeding

| Form | Meaning |
|---|---|
| `T()`, `Gen()` | the package default seed, `12345` |
| `T(12345)`, `Gen(12345)` | an **integer seed**, expanded through splitmix64 |
| `T(v)`, `Gen(v)` | the family's **own representation** — its seed vector, or a `UInt128` for PCG — taken as the state or key itself |
| `Random.seed!(rng, 12345)` | same as `T(12345)`, in place |
| `srand!(rng, seed)`, `srand!(gen, seed)` | reseed, resetting both boundaries |

The distinction is the whole rule: an integer is a *seed* and gets hashed; a
value shaped like the state *is* the state.

### On the stream

| Call | Meaning |
|---|---|
| `next_substream!(rng) -> rng` | move to the start of the next substream |
| `reset_substream!(rng) -> rng` | rewind to the start of the current substream |
| `reset_stream!(rng) -> rng` | rewind to the start of the current stream |
| `advance_state!(rng, e, c) -> rng` | move by `2^e + c` draws; negative moves backwards |
| `get_state(rng)` / `set_state!(rng, s) -> rng` | save and restore the position only |
| `copy(rng)`, `show(io, rng)` | independent copy; one-line summary, or the three anchors when displayed at the REPL |
| the full `Random` API | `rand`, `rand!`, `randn`, `shuffle`, ranges, every integer and float type |

`get_state` returns a family-specific representation. Treat it as opaque and
only ever hand it back to `set_state!`.

### On the generator object

| Call | Meaning |
|---|---|
| `next_stream!(gen) -> rng` | the next non-overlapping stream |
| `next_stream!(gen, n) -> Vector` | the next `n` of them; the thread-safe way to obtain streams for parallel work |
| `srand!(gen, seed) -> gen` | reset the seed the next `next_stream!` will use |
| `get_state(gen)` / `set_state!(gen, s) -> gen` | save and restore that seed |
| `show(io, gen)` | display it, on one line or in full as above |

### What is *not* common

| Name | Where | Why not everywhere |
|---|---|---|
| `short_jump!(rng)` | xoshiro/xoroshiro, PCG | the family's spelling of a jump by exactly the substream distance; use `next_substream!` in portable code |
| `long_jump!(rng)` | xoshiro/xoroshiro, PCG | jumps by the stream distance in place. Meaningless for a counter-based generator, where changing stream means changing *key*, which belongs to the generator object |
| `checkseed`, `checkseed63` | MRG32k3a and MRG63k3a, xoshiro/xoroshiro | those two families have invalid seeds — the MRG moduli with no all-zero component, and the xoshiro all-zero state, which is a fixed point. PCG and the counter-based families accept every value |
| `stream_key` | counter-based | the key schedule hook; there is nothing to schedule in a recurrence-based generator |
| `next(rng)` *(internal)* | all | raw word output, before any conversion |

---

## MRG32k3a

### Constructors

| Signature | Description |
|---|---|
| `MRG32k3a()` | default seed `[12345, 12345, 12345, 12345, 12345, 12345]` |
| `MRG32k3a(seed::Integer)` | integer seed, folded into the valid seed space |
| `MRG32k3a(x::Vector{Int})` | seed all three states (`Cg = Bg = Ig = x`) |
| `MRG32k3a(x, y, z::Vector{Int})` | explicit current/substream/stream starts |

Seeds are validated with `checkseed`; invalid seeds throw an `AssertionError`.

### Functions

**`rand(rng::MRG32k3a) -> Float64`**
Uniform in [0, 1) with ~32 bits of resolution. This is the native output.

**`rand(rng::MRG32k3a, T)`**
Supported `T`: `Float64`, `Float32`, `Float16`, `UInt8…UInt128`,
`Int8…Int128`, `Bool`, and any `UnitRange` (e.g. `rand(rng, 1:10)`).

**`reset_stream!(rng) -> rng`**
Set the current state (`Cg`) *and* the substream start (`Bg`) back to the
stream start (`Ig`).

**`reset_substream!(rng) -> rng`**
Set `Cg` back to `Bg`.

**`next_substream!(rng) -> rng`**
Advance `Bg` to the next substream boundary (jump of ≈ 2^76 steps) and set
`Cg = Bg`.

**`advance_state!(rng, e::Integer, c::Integer) -> rng`**
Jump forward/backward inside the current substream by `n` steps where
`n = 2^e + c`. Negative `e` or `c` move backwards. Cost is O(max(|e|, log |c|))
matrix operations.

**`get_state(rng) -> Vector{Int}`**
Return a copy of `Cg`.

**`checkseed(x) -> Bool`**
`true` iff `x` has length 6, all entries ≥ 0, entries 1–3 are < m1 and not all
zero, entries 4–6 are < m2 and not all zero.

**`DEFAULT_SEED`**
The constant seed vector used by `MRG32k3a()` and `MRG32k3aGen()`.

**`copy(m::MRG32k3a) -> MRG32k3a`**
Independent deep copy of the generator (all three state vectors).

### Stream generation

| Signature | Description |
|---|---|
| `MRG32k3aGen()`, `MRG32k3aGen(12345)`, `MRG32k3aGen(v)` | create a stream generator |
| `next_stream!(gen) -> MRG32k3a` | return a new stream; internal seed leaps 2^127 values ahead |
| `srand!(gen, seed) -> gen` | reset the stored seed |
| `get_state(gen) -> Vector{Int}` / `set_state!(gen, s)` | save and restore it |

---

## MRG63k3a

The same combined MRG in 64-bit arithmetic: L'Ecuyer (1999), Table II, fourth
entry. Two moduli just under `2^63` (`m1 = 2^63 - 6645`, `m2 = 2^63 - 21129`),
period ≈ `2^377`, and just under 63 random bits per step against 32 for
MRG32k3a. Everything below mirrors the MRG32k3a section; only the numbers
differ.

### Constructors

| Signature | Description |
|---|---|
| `MRG63k3a()` | default seed `[12345, 12345, 12345, 12345, 12345, 12345]` |
| `MRG63k3a(seed::Integer)` | integer seed, folded into the valid seed space |
| `MRG63k3a(x::Vector{Int})` | seed all three states (`Cg = Bg = Ig = x`) |
| `MRG63k3a(x, y, z::Vector{Int})` | explicit current/substream/stream starts |

Seeds are validated with `checkseed63`; invalid seeds throw an `AssertionError`.

### Functions

**`rand(rng::MRG63k3a) -> Float64`**
Uniform in (0, 1), from an integer with ~63 bits of resolution, of which a
`Float64` keeps 53. This is the native output, and it reproduces L'Ecuyer's C
implementation value for value.

**`rand(rng::MRG63k3a, T)`**
Supported `T`: as for MRG32k3a. Words are assembled from 32-bit chunks, so a
`UInt64` costs two steps (four for MRG32k3a) and a `UInt32` one.

**`next_substream!(rng) -> rng`**
Advance `Bg` by `2^150` steps and set `Cg = Bg`.

**`reset_stream!`**, **`reset_substream!`**, **`advance_state!`**,
**`get_state`**, **`set_state!`**, **`srand!`**, **`copy`**
Exactly as for MRG32k3a, with one internal difference that does not show:
MRG63k3a keeps its state one step ahead of the position (Vigna's third
optimization, worth 7% here), so `get_state` returns the seed representation by
stepping the stored vector back, and the constructors, `set_state!` and
`Random.seed!` step a seed forward. Reading `rng.Cg` directly is therefore not
the same thing as `get_state(rng)` for this generator. See the
[Implementation Notes](implementation.md).

**`checkseed63(x) -> Bool`**
`true` iff `x` has length 6, all entries ≥ 0, entries 1–3 are < m1 and not all
zero, entries 4–6 are < m2 and not all zero.

**`DEFAULT_SEED63`**
The constant seed vector used by `MRG63k3a()` and `MRG63k3aGen()`.

### Stream generation

| Signature | Description |
|---|---|
| `MRG63k3aGen()`, `MRG63k3aGen(12345)`, `MRG63k3aGen(v)` | create a stream generator |
| `next_stream!(gen) -> MRG63k3a` | return a new stream; internal seed leaps 2^250 values ahead |
| `srand!(gen, seed) -> gen` | reset the stored seed |
| `get_state(gen) -> Vector{Int}` / `set_state!(gen, s)` | save and restore it |

The jumps are unaffected by that shift: the jump matrices commute with the
one-step matrix, so a jumped shifted state is the shifted jumped state.

The stream and substream distances are not L'Ecuyer's: he published jump
matrices for MRG32k3a only. `2^250` and `2^150` are this package's choice,
scaled from his `2^127` and `2^76` by the ratio of the two periods, and the
matrices are computed at precompilation rather than tabulated.

---

## Xoshiro256+

### Constructors

| Signature | Description |
|---|---|
| `Xoshiro256p()` | the package default seed, `12345`, through splitmix64 |
| `Xoshiro256p(seed::Integer)` | integer seed, through splitmix64 (never all-zero) |
| `Xoshiro256p(s::NTuple{4,UInt64})` / `Xoshiro256p(v::Vector{<:Unsigned})` | `Cg = Bg = Ig = s` |
| `Xoshiro256p(x, y, z)` | explicit current/substream/stream states |

### Functions

**`rand(rng::Xoshiro256p) -> Float64`**
Uniform in [0, 1), built from a full 64-bit draw.

**`rand(rng::Xoshiro256p, T)`**
Supported `T`: `Float64`, `Float32`, `Float16`, and the integer and `Bool`
types of the `Random` API. Ranges (`rand(rng, 1:10)`) go through the standard
`Random` sampler, which rejects rather than folding and is therefore unbiased.

**`srand!(rng, seed::Vector{UInt64}) -> rng`**
Re-seed an existing generator (first 4 words are used).

**`reset_stream!(rng) -> rng`**, **`reset_substream!(rng) -> rng`**,
**`next_substream!(rng) -> rng`**
Same semantics as for MRG32k3a; `next_substream!` performs a `short_jump!`
(leap of ≈ 2^96 values).

**`short_jump!(rng) -> rng`**
Jump to the beginning of the next 2^96-value block (Vigna's short jump).

**`long_jump!(rng) -> rng`**
Leap ≈ 2^192 values ahead — used to delimit streams.

**`get_state(rng) -> Vector{UInt64}`**
Copy of the current state `Cg`.

**`copy(rng) -> Xoshiro256p`**
Independent copy.

*(internal)* **`next(rng) -> UInt64`**: raw 64-bit output; also available as
`rand(rng, UInt64)`.

### Stream generation

| Signature | Description |
|---|---|
| `Xoshiro256plusGen()`, `Xoshiro256plusGen(12345)`, `Xoshiro256plusGen(v)` | create a stream generator (a seed *vector* must have exactly 4 words) |
| `next_stream!(gen) -> Xoshiro256p` | new independent stream (`long_jump!` from the stored seed) |
| `srand!(gen, seed::Vector{UInt64}) -> gen` | reset the stored seed |
| `get_state(gen) -> Vector{UInt64}` | seed that will be used by the next `next_stream!` call |

---

## Xoshiro / xoroshiro families

All variants share one implementation (`LinRNG{N,S}` internally, `N` = state
words, `S` = scrambler) and expose the same API as [Xoshiro256+](#xoshiro256)
above. Exported types:

| RNGs | Stream generators |
|---|---|
| `Xoroshiro128p`, `Xoroshiro128ss`, `Xoroshiro128pp` | `Xoroshiro128pGen`, `Xoroshiro128ssGen`, `Xoroshiro128ppGen` |
| `Xoshiro256p`, `Xoshiro256ss`, `Xoshiro256pp` | `Xoshiro256plusGen` (also `Xoshiro256pGen`), `Xoshiro256ssGen`, `Xoshiro256ppGen` |
| `Xoshiro512p`, `Xoshiro512ss`, `Xoshiro512pp` | `Xoshiro512pGen`, `Xoshiro512ssGen`, `Xoshiro512ppGen` |

Constructors accept an `NTuple{N,UInt64}` or a `Vector{<:Unsigned}` of length
`N` (1 or 3 arguments). Available functions: `rand` (Float64/Float32/Float16,
UInt64, ranges), `srand!`, `reset_stream!`, `reset_substream!`,
`next_substream!`, `short_jump!`, `long_jump!`, `get_state`, `copy`.

**`advance_state!(rng, e::Integer, c::Integer) -> rng`**
Jump forward by `n` steps (`n = 2^e + c` if `e > 0`, `n = -2^(-e) + c` if
`e < 0`, `n = c` if `e = 0`) or backward if `n < 0`. Implemented by computing
the jump polynomial `x^n mod p(x)` over GF(2) (p = the family's characteristic
polynomial) and applying it to the current state; distances are reduced
modulo the period. Only `Cg` moves — stream/substream boundaries are kept.
Cost is O((64N)²) GF(2) operations (≈ 10–500 ms depending on family); for the
standard distances prefer `short_jump!`/`long_jump!`, which use precomputed
constants and are allocation-free.

Jump semantics (all xoshiro-family generators):

- `short_jump!(rng)` — jump anchored at the current **substream start** (`Bg`);
  lands at the next substream boundary (2^64 values for xoroshiro128, 2^128
  for xoshiro256, 2^256 for xoshiro512).
- `long_jump!(rng)` — jump anchored at the **stream start** (`Ig`); delimits
  independent streams.
- `next_stream!(gen)` applies the long-jump polynomial to the stored seed.

All jump constants are the official ones from
[xoshiro.di.unimi.it](http://xoshiro.di.unimi.it) and outputs are validated
byte-for-byte against the original C implementations.

---

## Where the families actually differ

Every call in [The common interface](#The-common-interface) is available on all
seventeen generators, so there is no availability matrix to consult. What differs
is cost and representation:

| | MRG32k3a | MRG63k3a | xoshiro / xoroshiro | PCG64, PCG64DXSM | Philox, Threefry |
|---|---|---|---|---|---|
| Native output | `Float64`, ~32-bit resolution | `Float64`, ~63-bit resolution | `UInt64` | `UInt64` | a block of four words |
| `get_state` returns | `Vector{Int}` (6) | `Vector{Int}` (6) | `Vector{UInt64}` (`N`) | `UInt128` | `(ctr, key, buffer, idx)` |
| Cost of `advance_state!` | O(e) matrix products | O(e) matrix products | O((64N)²) GF(2) ops, ~10–500 ms | O(log n) multiplies | O(1), a counter addition |
| `short_jump!`, `long_jump!` | — | — | yes | yes | — |
| Rejected seeds | `checkseed`: two moduli, no all-zero component | `checkseed63`: same, wider moduli | `checkseed`: the all-zero state | none | none |
| Whole-block `rand!` | — | — | — | — | yes, ~1.7× the scalar path |
| Extra hooks | `checkseed`, `DEFAULT_SEED` | `checkseed63`, `DEFAULT_SEED63` | jump polynomials | — | `stream_key` |

For the standard distances on a xoshiro or PCG generator prefer
`short_jump!`/`long_jump!` over `advance_state!`: they use precomputed
constants (xoshiro) or a single doubling chain (PCG) and are allocation-free.

## PCG

PCG (O'Neill 2014) runs a 128-bit linear congruential recurrence and permutes
the state into 64 bits of output. `PCG64` is the default bit generator of
NumPy, and the outputs here match it exactly for the same state; `PCG64DXSM`
is the variant NumPy recommends for large-scale parallel work.

Streams come from the closed-form LCG jump, not from PCG's own increment-based
stream mechanism, which this package deliberately does not expose — see the
[Implementation Notes](implementation.md) for the reason and the
[FAQ](faq.md) for what to do instead when reproducing a NumPy pipeline.

```@docs
PCG64
PCG64DXSM
PCG64Gen
PCG64DXSMGen
PCGRNG
PCGGen
```

## Counter-based generators

A counter-based RNG produces its output block as a keyed bijection of a
counter, `R = b_K(C)`, rather than by iterating a state transition. `CBRNG`
holds the machinery every such family shares — counter, key, block buffer,
streams and substreams — so a variant only supplies its bijection.

```@docs
CBRNG
RandomDataStreams.stream_key
```

### Philox

```@docs
PhiloxGen
PhiloxRNG
Philox4x64Gen
Philox4x64RNG
```

### Threefry

Same construction as Philox, with no multiplication at all: every round is
add / rotate / xor, which makes Threefry reproducible bit-for-bit across
architectures, including ones with no fast integer multiply. Salmon et al.
(2011) found it the fastest of the family on the CPUs of the time and
recommend twenty rounds there; on a current x86 our own measurement puts
Philox4x64-10 ahead (see the performance notes).

```@docs
Threefry4x64Gen
Threefry4x64RNG
Threefry4x32Gen
Threefry4x32RNG
```
