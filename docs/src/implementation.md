# Implementation Notes

## MRG32k3a

MRG32k3a is L'Ecuyer's combined multiple recursive generator of order 3. It
maintains two 3-component integer states updated by the recurrences

```
p1 = (1403580 · x2 − 810728 · x1) mod m1,      m1 = 2^32 − 209
p2 = (527612  · y2 − 1370589 · y0) mod m2,    m2 = 2^32 − 22853
u  = (p1 − p2) / m1   (with wrap-around) ∈ [0, 1)
```

The combined generator has period ≈ 2^191.

### Modular arithmetic without overflow, and without tests

The recurrence follows Vigna's testless formulation
(<https://github.com/vigna/MRG32k3a>). The two negative coefficients are
carried as positive numbers and subtracted, and a multiple of the modulus is
added, so the argument of `%` can never be negative and the residual needs no
correction afterwards; the combination `z = p1 - p2 mod m1` is done with an
arithmetic shift rather than a comparison. Worth about 15% here — 204 million
`Float64` per second before, 230 to 235 after depending on the run — and the
stream is unchanged bit for bit.

It is worth 15% and not the factor of two Vigna reports, because most of what
he removes by hand LLVM already removes for us: the previous formulation, with
its two residual corrections and its comparison, compiled to 72 instructions
with **no branch and no division** — the corrections had become conditional
moves and the constant moduli multiply-shift sequences. What is left to gain is
the fix-up a *signed* remainder needs, which disappears once the dividend is
known non-negative: 72 instructions become 68.

Two variants measured and rejected: keeping the state in six scalar fields
instead of a `Vector{Int64}` is *slower* (226 against 235) while breaking the
constructors and `Cg`; and computing the output from the current state before
the update, which needs the state kept one step ahead to preserve the stream,
gives 231 and changes what `get_state` and the jump matrices operate on.

That second variant is Vigna's third optimization, and the conclusion does
**not** carry over to MRG63k3a, where it is worth 7% and is used — see below.
Re-measured here with the same harness, MRG32k3a gives 236 against 235: the
32-bit step is short enough that the combination is already hidden behind it,
so there is nothing to overlap.

### Modular arithmetic in the jump machinery

All state components stay below `m1 < 2^32`, so products fit in `Float64`
exactly (< 2^53). The helper `MultModM(a, s, c, m)` computes `(a·s + c) mod m`
using this fact and falls back to a split-by-2^17 computation when intermediate
values would exceed 2^53. Matrix helpers:

- `MatVecModM(A, s, m)` — matrix–vector product mod m
- `MatMatModM(A, B, m)` — matrix product mod m
- `MatTwoPowModM(A, e, m)` — A raised to the power 2^e
- `MatPowModM(A, n, m)` — A^n by binary exponentiation

### Streams, substreams, jumps

Moving a stream forward by k steps is a linear operation on its state, hence a
precomputed matrix power applied to the state vector:

| Operation | Matrix | Step |
|---|---|---|
| `next_stream!(gen)` | `A1p127`, `A2p127` | 2^127 values |
| `next_substream!(rng)` | `A1p76`, `A2p76` | 2^76 values |
| `advance_state!(rng, e, c)` | `MatTwoPowModM` / `InvA1`, `InvA2` | arbitrary (inverse matrices allow backward jumps) |

The RNG keeps three checkpoints: `Cg` (current), `Bg` (substream start),
`Ig` (stream start), which is what makes `reset_substream!` and
`reset_stream!` O(1).

## MRG63k3a

The same construction as MRG32k3a, sized for 64-bit arithmetic: the fourth
entry of Table II in L'Ecuyer (1999), which he implements in C as `MRG63k3a`.

```
p1 = (1754669720 · x2 −  3182104042 · x0) mod m1,   m1 = 2^63 − 6645
p2 = (31387477935 · y2 − 6199136374 · y0) mod m2,   m2 = 2^63 − 21129
u  = (p1 − p2) / (m1 + 1)   (with wrap-around) ∈ (0, 1)
```

Period ≈ `2^377`, against `2^191`; entropy per step `log2(m1) = 63.0` bits,
against 32.0. The package reproduces L'Ecuyer's C implementation value for
value, checked on three seeds and a million draws.

### Reduction without a 128-bit division

The products no longer fit in a `Float64` mantissa — that is the whole reason
this generator is a separate implementation and not a parameter set — so they
are computed with `widemul` into an `Int128`. What must be avoided then is the
remainder: a 128-bit `%` is a software routine, and it would dominate the step.

Both moduli are pseudo-Mersenne, `m = 2^63 − c` with `c` small, so the
reduction is two shifts, a multiply and an add. Splitting the 128-bit value at
bit 63 as `t = hi·2^63 + lo` gives

```
t ≡ hi·c + lo   (mod m),         since 2^63 ≡ c (mod m).
```

With the same testless formulation as MRG32k3a — the negative coefficient
carried positive and a multiple of the modulus added — `t` stays below `2^99`,
so `hi < 2^36` and `hi·c < 2^51`: the folded value is below `2^63 + 2^51`,
which is under `2m`, and one conditional subtraction finishes it. The
combination `z = p1 − p2 mod m1` then uses the same arithmetic shift as
MRG32k3a. The whole step compiles to 61 instructions with no division and no
branch — the two conditional subtractions become conditional moves.

The reference C code takes a different route to the same values, Bratley,
Fox & Schrage's approximate factoring (`q = m ÷ a`, `r = m mod a`), which keeps
every intermediate inside a 64-bit signed integer at the cost of two divisions
and two branches per coefficient. That is the right choice for a 1999 C
compiler and the wrong one here: the reduction above is what makes MRG63k3a
*faster* than MRG32k3a per random bit rather than slower. The alternative was
measured, not assumed — a plain 128-bit remainder instead of the fold runs at
19 million draws per second against 196, a factor of ten.

### The state runs one step ahead

Vigna's three optimizations for MRG32k3a are the testless formulation above,
the branchless combination, and computing the output from the *current* state
so that the processor overlaps it with the next state. The first two are shared
with MRG32k3a; the third is used **here only**, because it is the one place
where the two generators measure differently:

| | MRG32k3a | MRG63k3a |
|---|---|---|
| output from the new state | 236 M/s | 183 M/s |
| output from the current state | 235 M/s | **196 M/s** |

The 63-bit step is a `widemul` and a fold, long enough that the combination
sitting at the end of it is visible; the 32-bit step is a `Float64` multiply
and a compiler-optimised `%`, short enough that it is not. So `next_pair!`
returns the pair already in `Cg[3]`, `Cg[6]` and computes the next one, and the
state vector is one step ahead of the position.

That shift is invisible from outside. It costs nothing anywhere it might have:
the jump matrices commute with the one-step matrix, so `advance_state!`,
`next_substream!`, `next_stream!` and the `Bg`/`Ig` checkpoints operate on the
shifted vector unchanged, and jumping a shifted state gives the shifted jumped
state. Only the boundary converts — the constructors and `Random.seed!` step a
seed forward, `get_state` and `show` step the stored vector back — at one
matrix-vector product each, on paths that are never hot. The stream is
identical bit for bit, which is what the reference vectors from L'Ecuyer's C
code check.

### Jump matrices

`MultModM` here is the straightforward `widemul` plus 128-bit `mod`, since it
serves only the jump machinery, never the draw. The matrix helpers are the same
four as for MRG32k3a.

The jump distances are **not** L'Ecuyer's: he published `A1p76`/`A1p127` for
MRG32k3a and nothing for MRG63k3a. This package uses `2^150` between substreams
and `2^250` between streams — his `2^76` and `2^127` scaled by the ratio of the
two periods, `377/191 ≈ 1.97` — which cuts the period into `2^127` streams of
`2^100` substreams of `2^150` numbers each. The four matrices are computed by
repeated squaring at precompilation rather than tabulated, roughly 400 matrix
products, and the test suite checks the arithmetic that produces them:
`A · InvA = I` for both components, `(A^(2^150))^(2^100) = A^(2^250)`,
`next_substream!` equal to `advance_state!(rng, 150, 0)`, and `next_stream!`
equal to `advance_state!(rng, 250, 0)`. The formulas behind `InvA1` and `InvA2`
are checked by a route that does not depend on this generator at all: applied
to MRG32k3a's coefficients they must reproduce the inverse matrices L'Ecuyer
tabulates, and they do.

### Integer outputs

Same construction as MRG32k3a, one size up: `z ∈ [1, m1]` is truncated to
32-bit chunks and wider words are concatenations of consecutive chunks. A
`UInt64` is two steps, against four; a `UInt32` is one, against two. Because
`m1 = 2^63 − 6645`, a residue class modulo `2^32` holds `2^31` values give or
take one, so the departure from uniformity is of order `2^-31` per chunk
against `2^-16` for MRG32k3a. Still not exact — no exact uniform word comes out
of a non-power-of-two modulus without rejection — but two orders of magnitude
closer, at half the cost.

`Float64` remains the native output (`Random.rng_native_52`): one step covers a
double, where declaring the word native would make every double cost two.

The one-step-ahead ordering helps the paths whose critical path ends in the
combination — `Float64` and `UInt32` — and not `UInt64`, where two steps run
back to back and the recurrence, not the combination, is the chain.

## Xoshiro256+

xoshiro256+ (Blackman & Vigna, 2019) keeps four 64-bit words and produces one
word per step with only shifts, XORs and rotations:

```julia
result = s1 + s4
t = s2 << 17
s3 = xor(s3, s1);  s4 = xor(s4, s2);  s2 = xor(s2, s3);  s1 = xor(s1, s4);  s3 = xor(s3, t);  s4 = rotl(s4, 45)
```

The output word is the sum of two internal words (`+` variant): fastest of the
xoshiro family for float generation, though the low three bits of the raw
output have limited linear complexity — irrelevant once scaled to `Float64`
(53 bits used). Period: 2^256 − 1.

### Jumps

`short_jump!` and `long_jump!` use Vigna's published jump polynomials. They are
computed as XOR-linear combinations of states visited while stepping through
the 256 bits of each jump constant. RandomDataStreams.jl hoists the jump constants into
compile-time tuples and evaluates jumps allocation-free on immutable state
tuples.

RandomDataStreams semantics differ subtly from the reference C code: jumps are **anchored**
at stream boundaries. `short_jump!(rng)` first resets `Cg ← Bg`, then applies
the jump polynomial, and stores the result in both `Cg` and `Bg`;
`long_jump!(rng)` does the same with respect to `Ig`. This matches L'Ecuyer's
stream model (`next_substream!`/`reset_stream!`) and makes scenario replay
reproducible regardless of consumption inside the current substream.

### Families and scramblers

The package implements all 64-bit variants from xoshiro.di.unimi.it through a
single generic type `LinRNG{N,S}` (state words × scrambler):

- transition `_lin_step` for `NTuple{2}` (xoroshiro128: a=24, b=16, c=37),
  `NTuple{4}` (xoshiro256) and `NTuple{8}` (xoshiro512);
- scramblers `+`, `**` (`rotl(s2·5,7)·9`) and `++`
  (`rotl(s1+s4,23)+s1` for 256; family-specific constants);
- per-family jump constants shared by all scramblers.

Exported aliases: `Xoroshiro128p/ss/pp`, `Xoshiro256p/ss/pp`,
`Xoshiro512p/ss/pp` plus matching `*Gen` stream generators.

All nine variants are regression-tested against byte-exact outputs produced by
the original C implementations compiled with gcc (sequences, short jumps and
long jumps).

### Arbitrary forward/backward jumps

Because the transition is multiplication by x in GF(2)[x]/p(x) — with p the
family's characteristic polynomial, as published in Vigna's reference files —
one can jump any distance n in constant time by applying the polynomial
`x^n mod p` through the same accumulate-and-step loop as the fixed jumps.
Backward distances use the multiplicative order of x: `x^(-n) = x^(2^deg-1-n)`
(deg = 64N). RandomDataStreams.jl implements a small GF(2) polynomial engine
(`_poly_mul_mod`, `_poly_pow_x`, BigInt exponents reduced modulo the period)
and exposes it as `advance_state!(rng, e, c)` with the exact distance
convention of MRG32k3a. Correctness is property-tested: fixed jumps coincide
with `short_jump!`, round trips (+k then -k) restore the state bit-for-bit,
and backward-jumped generators re-emit previously seen values.

### State representation

The state lives in `NTuple{4,UInt64}` fields. Immutable tuples let the JIT keep
the whole working set in registers inside `next`, eliminating bounds checks and
heap traffic that a `Vector{UInt64}` representation incurs. Measured
throughput: ≈ 10⁹ draws/s on a single core.

## PCG

`PCGRNG{V}` runs a linear congruential recurrence on 128 bits,
`s <- a*s + c (mod 2^128)`, and permutes the state into 64 bits of output. Two
variants ship: `PCG64` with the XSL-RR permutation, which is what
`numpy.random.default_rng()` returns, and `PCG64DXSM` with the DXSM
permutation and the cheap 64-bit multiplier, which NumPy recommends for
large-scale parallel work. They differ in one more detail that matters for
bit-exactness: PCG64 steps the state and permutes the result, DXSM permutes the
state and then steps it. Both are pinned by known-answer vectors taken from
NumPy.

**Jumps are closed-form.** For an LCG,

```
s_n = a^n * s_0 + c * (a^n - 1)/(a - 1)   (mod 2^128),
```

which Brown's (1994) doubling recurrence evaluates in `O(log n)`
multiplications. `advance_state!` therefore costs the same at any distance, and
a backward jump of `n` is the forward jump of `2^128 - n`, so it costs the same
as a forward one. This is the cheapest arbitrary jump in the package: the
xoshiro families pay `O(deg^2)` GF(2) operations for the same operation, tens
to hundreds of milliseconds, and MRG32k3a a sequence of matrix products.
**The jump distances are odd, and that is not cosmetic.** The obvious choice —
2^64 between substreams, 2^96 between streams, mirroring the xoroshiro128
layout — is wrong for a linear congruential recurrence. By the
lifting-the-exponent lemma,

```
v2(a^n - 1) = v2(a - 1) + v2(a + 1) + v2(n) - 1 = 2 + v2(n)   (n even),
```

so a jump of `2^m` produces a jump multiplier congruent to `1` modulo
`2^(m+2)`. With `m = 96`, the states of successive streams agree in their low
98 bits up to an arithmetic progression. Each stream still passes SmallCrush on
its own; interleaving 64 of them round-robin fails it outright, with 14 of 15
p-values at 0. The package's inter-stream battery found this, which is the
argument for having one.

An odd distance has `v2(n) = 0`, leaving the jump multiplier as far from the
identity as the multiplier itself. Substreams are therefore `2^64 + 1` apart
and streams `2^32 * (2^64 + 1) + 1`, which is still 2^32 streams of 2^32
substreams of 2^64 draws and still non-overlapping — the last substream of a
stream ends two values below the next stream's start. With those distances the
interleaved battery passes. Both the parity of the distances and the 2-adic
distance of the resulting jump multipliers are asserted in the test suite.

**The increment is fixed, and that is deliberate.** PCG has its own notion of a
stream: every odd increment `c` gives a full-period orbit, and NumPy exposes it
as part of the state. Those orbits are not known to be independent. Writing
`t_n = s_n + h` and matching the two recurrences gives

```
h * (a - 1) = c1 - c2,
```

which has a solution whenever `4` divides `c1 - c2` — the full-period condition
forces `a = 1 (mod 4)`, and `v2(a - 1) = 2` for the PCG multipliers. For half
of all pairs of odd increments, then, the two "independent streams" are the
same state sequence translated by a constant. Taking `c2 = c1 - (a - 1)` makes
that constant `1`, and the two output sequences then differ by a mean Hamming
distance of 2 bits out of 64, against 32 for unrelated sequences, with 94% of
draws differing in at most 4 bits.

That is an adversarial pair; most pairs of increments are far less structured
and would show nothing. The objection is not that every pair is bad but that no
criterion is published for telling the good pairs from the bad ones, which is
exactly what L'Ecuyer et al. (2021, Sec. 3) argue against — the same objection
they raise against Squares. So the package fixes the increment to the reference
value for every PCG stream and derives streams from jumps, whose non-overlap is
a statement about distances along one orbit. The algebra above is locked in the
test suite.

## Counter-based generators

`CBRNG{B,W,N,K}` holds what every counter-based family shares — the counter,
the key, the block most recently produced and the index of the next word in it.
This is the state `(k, i, j)` of L'Ecuyer et al. (2021, Sec. 3), with
`f(k, i, j) = (k, i + I[j = d-1], (j + 1) mod d)` and `d = N`. A variant only
supplies `bijection(::Val{B}, ctr, key)`.

Two families are built on it. **Philox** keeps both halves of a fixed-constant
multiplication and xors the high halves into the other words, with a Weyl
sequence for the round key. **Threefry** is a reduction of the Threefish cipher
used in Skein and contains no multiplication at all — add, rotate, xor only —
so it is reproducible bit-for-bit on any architecture, including ones with no
fast integer multiply. Adding it required only its bijection and its aliases,
no stream machinery.

Its rounds are unrolled at compile time with a generated function. Each round
uses a different pair of Skein rotation constants, so a plain loop indexes the
constant table with a value LLVM cannot fold; unrolling is worth a factor of
5.8 on the bijection (7.7 → 44.8 Mblocks/s), which is the difference between
Threefry being unusable and being in the same class as Philox. The round count
is a parameter of the bijection, not of the type, so the faster 13-round
variant is three lines of user code.

**Streams** are keys. **Substreams** partition the counter, following the same
paper: "one can use the `c0 < c` most significant bits of the counter to
determine the substream, and the remaining `c1 = c - c0` bits for the position
within the substream". `_substream_shift` splits the counter in half, so
Philox4x32-10 has 2^64 substreams of 2^64 blocks each.

**Key schedules.** `CBGen` walks the seeds `0, 1, 2, ...` and maps each through
`stream_key`, a bijection that defaults to the identity. Consecutive small keys
are safe for Philox — the ten rounds of encoding absorb the structure, as
reported by Salmon et al. (2011) and discussed in L'Ecuyer et al. (2021,
Sec. 3) — but they are *not* safe for every counter-based construction, so a
family whose bijection is weak for structured keys must override `stream_key`
with a schedule that hashes the seed.

## Integer outputs of MRG32k3a

MRG32k3a returns `z = (x1 - x2) mod m1` with `m1 = 2^32 - 209`, prime, scaled to
`u = z / (m1 + 1)`. Its native output is a uniform integer over a range that is
**not** a power of two, carrying `log2(m1) = 32.0` bits. This is why the
reference literature — L'Ecuyer (1999), and the RNGStreams package of L'Ecuyer,
Simard, Chen & Kelton (2002) whose stream API this package follows — exposes
only `RandU01()` and `RandInt(i, j) = i + floor((j - i + 1) * RandU01())`, and
no native-word or raw-bit primitive: no exact uniform word comes out of one
draw without rejection.

Machine integers are therefore an extension beyond what that literature
specifies and validates, and they are built here by assembling 16-bit chunks of
`z`, one MRG step per chunk — four steps for a `UInt64`. Truncating `z` to 16
bits carries a relative bias of `2^-16`, since `2^16` does not divide `m1`;
taking the low bits is sound because the modulus is prime, so there is no
low-bit weakness of the kind a power-of-two LCG has.

The cheaper alternative — assembling a word from the mantissas of two `[1, 2)`
draws, two steps instead of four — was implemented here and removed. It assumes
52 random bits per double, and MRG32k3a supplies 32; the low mantissa bits are
a near-deterministic function of the high ones. The result passed every
`U(0,1)` battery and failed SmallCrush on its own bit stream with six p-values
at zero, bit 12 coming out set in two draws out of three. `test/test_bits.jl`
is the regression net.

Narrower types are truncations of one chunk. None of these integers is exactly
uniform over its full width — fine for indexing, shuffling and flags in a
simulation, not for cryptographic use. `MRG63k3a` above is the same
construction with twice the chunk width and half the steps per word; it is
the one to reach for when a run consumes integers rather than floats.

## Testing strategy

Reference sequences were captured from the reference implementations
(L'Ecuyer's C code for MRG32k3a and MRG63k3a — for the latter, ten doubles, the
draw one million steps later, and the raw combined integer on three seeds;
Vigna's constants for the xoshiro jumps) and regression-checked after optimization: identical outputs are
produced for every supported type, plus reset/jump/state round-trips.

## Performance snapshot

Reproduce with `julia -O3 scripts/benchmarks/throughput.jl`, which prints the
Julia version and CPU it ran on. On a hybrid CPU (Intel 12th generation and
later), pin the process first — `taskset -c 0 julia -O3 …` — and say which core
type was used: otherwise the scheduler moves the run between performance and
efficiency cores and the minimum reflects whichever was fastest. Figures below are one run on a Skylake x86_64,
Julia 1.12.5, single core; treat them as ratios and re-run on the machine that
matters to you.

**The measurement method is part of the result.** Naive loops disagree by
nearly a factor of two, so the script fixes one method and states it:
BenchmarkTools with the minimum sample; scalar draws accumulated with `xor` on
the raw bits, because a floating-point `+` chain has four cycles of latency —
longer than a draw from the fastest generators here, so it would measure the
adder; nothing written to memory in the scalar loop, because storing every draw
turns the benchmark into one of memory bandwidth (the same xoshiro reads
867 M/s with an accumulator and 481 M/s storing into an 8 MB vector); and
`rand!` measured into a 4096-element vector, small enough to stay in cache.

Scalar draws: millions per second, and nanoseconds per draw. Every generator
the package ships is listed, each scrambler variant separately. Run-to-run
variation is a few percent, so read differences below that as noise.

| Generator | `Float64` | ns | `UInt64` | ns | `UInt32` | ns |
|---|---|---|---|---|---|---|
| `MRG32k3a` | 236 | 4.25 | 47 | 21.06 | 95 | 10.48 |
| `MRG63k3a` | 196 | 5.10 | 93 | 10.76 | 199 | 5.04 |
| `Xoroshiro128p` | 697 | 1.43 | 941 | 1.06 | 1090 | 0.92 |
| `Xoroshiro128ss` | 655 | 1.53 | 876 | 1.14 | 848 | 1.18 |
| `Xoroshiro128pp` | 599 | 1.67 | 818 | 1.22 | 926 | 1.08 |
| `Xoshiro256p` | 788 | 1.27 | 1242 | 0.81 | 1221 | 0.82 |
| `Xoshiro256ss` | 715 | 1.40 | 1140 | 0.88 | 962 | 1.04 |
| `Xoshiro256pp` | 688 | 1.45 | 1032 | 0.97 | 1055 | 0.95 |
| `Xoshiro512p` | 570 | 1.75 | 798 | 1.25 | 742 | 1.35 |
| `Xoshiro512ss` | 523 | 1.91 | 742 | 1.35 | 614 | 1.63 |
| `Xoshiro512pp` | 532 | 1.88 | 725 | 1.38 | 665 | 1.50 |
| `PCG64` | 504 | 1.99 | 599 | 1.67 | 605 | 1.65 |
| `PCG64DXSM` | 561 | 1.78 | 743 | 1.35 | 745 | 1.34 |
| `Philox4x32-10` | 93 | 10.75 | 109 | 9.20 | 272 | 3.68 |
| `Philox4x64-10` | 203 | 4.94 | 262 | 3.82 | 259 | 3.86 |
| `Threefry4x32-20` | 84 | 11.86 | 92 | 10.89 | 186 | 5.39 |
| `Threefry4x64-20` | 147 | 6.80 | 153 | 6.52 | 157 | 6.35 |

Array fill with `rand!`, millions of elements per second (nanoseconds per
element is `1000` over the figure):

| Generator | `Float64` | `UInt64` | `UInt32` |
|---|---|---|---|
| `MRG32k3a` | 163 | 47 | 94 |
| `MRG63k3a` | 184 | 93 | 185 |
| `Xoroshiro128p` | 546 | 581 | 581 |
| `Xoroshiro128ss` | 531 | 538 | 520 |
| `Xoroshiro128pp` | 507 | 500 | 511 |
| `Xoshiro256p` | 501 | 519 | 520 |
| `Xoshiro256ss` | 492 | 520 | 472 |
| `Xoshiro256pp` | 471 | 505 | 507 |
| `Xoshiro512p` | 380 | 391 | 426 |
| `Xoshiro512ss` | 375 | 393 | 390 |
| `Xoshiro512pp` | 377 | 389 | 408 |
| `PCG64` | 575 | 643 | 628 |
| `PCG64DXSM` | 632 | 744 | 735 |
| `Philox4x32-10` | 158 | 178 | 349 |
| `Philox4x64-10` | 359 | 339 | 257 |
| `Threefry4x32-20` | 100 | 106 | 206 |
| `Threefry4x64-20` | 209 | 181 | 156 |

What the tables say. Within each xoshiro family the ordering is the same and
the spread is small: `+` is fastest, then `**`, then `++`, within about 15% of
each other — the scrambler is a couple of instructions on top of a shared
transition. Across families, state size costs: 128 and 256 bits are close,
512 bits is a third slower. The counter-based generators sit an order of
magnitude below on `Float64`, which is what buying a keyed bijection per block
costs, and the two 32-bit ciphers are fastest in `UInt32`, where a draw is one
cipher word rather than two.

The two PCG variants land between the xoshiro families and the MRG generators,
at roughly two thirds of `Xoshiro256p`: a 128-bit multiply per draw is more work
than a handful of shifts and xors. `PCG64DXSM` is the faster of the two despite
its extra output multiply, because its "cheap" 64-bit multiplier turns the
128x128 state multiplication into a 64x128 one. Both are the fastest generators
in the package under `rand!`, ahead of every xoshiro: the whole state is one
128-bit word, so the bulk loop touches far less memory per element than a
four- or eight-word state does.

The one outlier is `MRG32k3a` in `UInt64`, at 21.1 ns against 4.3 ns for its
`Float64`: a 64-bit word costs four MRG steps, because the modulus is not a
power of two and the word is assembled from 16-bit chunks. See the section on
its integer outputs.

`MRG63k3a` is where that cost goes away, and the two rows read as one trade.
Its `Float64` is 20% slower (5.10 ns against 4.25) — the step is a 128-bit
multiply where MRG32k3a's is a `Float64` one — while its `UInt64` is 2.0×
faster (10.76 ns against 21.06) and its `UInt32` 2.1× faster (5.04 against
10.48), because a word is two 32-bit chunks rather than four 16-bit ones. Per
random *bit* it is ahead everywhere: 0.081 ns against 0.133 for the double,
where 63 bits come out of the same one step.

In bulk it is the recurrence-based generator that loses least: 184 against 196
million `Float64`/s is a drop of 6%, where MRG32k3a drops 31% and the xoshiro
families about a third. Its `UInt64` is flat at 93 either way. Why the drop is
so much smaller here than for its sibling is not something these figures
settle, so it is reported and not explained.

The counter-based generators are the ones that gain from filling an array:
`rand!` produces whole blocks straight into it, so the counter, the block
buffer and the index are touched once per block instead of once per draw.
Philox4x64-10 goes from 203 to 359 million `Float64` per second, a factor of
1.77. The recurrence-based generators have no such block to exploit and are
*slower* in bulk than in the accumulator loop, by the cost of the stores.

Two combinations get no block path and fall back to the scalar loop: `UInt32`
from a 64-bit family, and any 64-bit draw taken while the generator sits
mid-block because the caller has been mixing widths.

Every generator draws with zero allocations, in both paths, as do `short_jump!`
and `long_jump!`.
