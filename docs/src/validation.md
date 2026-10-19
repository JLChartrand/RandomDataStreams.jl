# Statistical Validation (TestU01)

`RandomDataStreams.jl` integrates seamlessly with [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html), the industry-standard C library for empirical statistical testing of RNGs. It calls that library directly, through the `TestU01_jll` artifact and a layer of its own in `test/tu01.jl`; [How the batteries are driven](#How-the-batteries-are-driven) says what that layer is and why it is not `RNGTest.jl`.

The validation is deliberately split in two. The **package test suite** runs
SmallCrush only, over three suites, and answers the question "does this
installation work" in about six minutes — nine generators times three suites at
some eleven seconds a battery, plus a minute for everything else. **Crush and
BigCrush run on demand** through `scripts/testu01/validate.jl`, because they
take hours to days; see [Running Crush and BigCrush](#Running-Crush-and-BigCrush).

## In the test suite: SmallCrush

A subset of the TestU01 batteries, **SmallCrush**, is automatically executed on all supported generators (`MRG32k3a`, `MRG63k3a`, `Xoshiro256+`, `PCG64`, `PCG64DXSM`, `Philox4x32-10`, `Philox4x64-10`, `Threefry4x64-20` and `Threefry4x32-20`) during the standard test suite.

It is the C battery itself, `bbattery_SmallCrush`, and the suite asserts on all
15 statistics it reports — read from `bbattery_pVal[]`, so they are the doubles
the library computed. A p-value outside [0.001, 0.999] fails the test and is
printed with its name and its value. Over a whole campaign such a line is
expected from chance and means nothing on its own (see [Reading a summary
report](#Reading-a-summary-report)); on one CI run of nine generators it is rare
enough to be worth a human's attention.

You can trigger this locally by running:

```julia
using Pkg
Pkg.test("RandomDataStreams")
```

## Dependence between streams

A battery run on a single stream says nothing about whether the *streams* are
independent of each other. Following L'Ecuyer et al. (2021, Sec. 6) — "it is
also important to test the dependence between those streams [...] one can
construct sequences that take a few values from each stream for a certain
number of streams, in a round-robin fashion" — the test suite also runs
SmallCrush on a sequence built by interleaving 64 streams, one value at a time,
for each of the nine generators. For a counter-based generator this is what
exercises the key schedule: a schedule handing out structurally related keys
shows up here and not in a battery run on a single stream.

The heavier configurations recommended by the paper (up to 1024 streams and up
to 8 values per stream) are reached through `--streams` and `--values` of
`scripts/testu01/validate.jl`. A campaign runs the 64 × 1 shape unless
`--shapes` names others, one job per shape:

```bash
./scripts/run-campaign.sh run --battery=bigcrush --suite=interleaved \
    --shapes=16x8,256x2,1024x1 --out=$HOME/bigcrush-shapes
```

**Substreams.** The paper describes two ways of making streams, and the
interleaved suite tests one: every stream its own key, or its own jump, from
`next_stream!`. The other is the substreams of a single stream, from
`next_substream!` — for a counter-based generator one key and consecutive
values of the counter's high half. That is the construction cuRAND uses:
its Philox4_32_10 subsequences are `PhiloxRNG`'s substreams, word for word,
and its MRG32k3a subsequences are MRG32k3a's substreams, 2^76 apart. The
`substreams` suite interleaves substreams 1 to `--streams` of the first stream,
in the same shapes, and replays a suspect on substreams of that same stream
past those it used. A campaign runs it only when asked: `--suite=all` still
means the three suites every earlier campaign ran, so their results directories
resume unchanged.

```bash
./scripts/run-campaign.sh run --battery=bigcrush --suite=substreams \
    --shapes=16x8,64x1,256x2,1024x1 --generator=Philox4x32-10,MRG32k3a
```

### Platform limitation

The batteries do not run on **Apple Silicon**. TestU01 is handed its callback
through `@cfunction($f, ...)`, which needs an executable trampoline for the
closure, and macOS on aarch64 forbids memory that is both writable and
executable — Julia raises `cfunction: closures are not supported on this
platform`. This affects every generator and every battery, and is a property of
the platform rather than of any package here.

The test suite probes the capability and skips the three batteries with a
message when it is missing, so the rest of the suite still runs; the on-demand
harness refuses with the same explanation. Everything else in the suite,
including the reference vectors, runs everywhere.

## The integer paths, at the bit level

The Crush batteries examine the 30 most significant bits of the `U(0,1)`
outputs, so a battery run on the float says nothing about how a generator
builds a machine integer. Where the integer *is* the native output — Philox
4x32, Threefry 4x32 — that is the same stream. Where it is a construction, it
needs testing for itself, and `test/test_bits.jl` runs SmallCrush on the bit
stream of: `MRG32k3a` in `UInt32` and in `UInt64` (16-bit chunks of a
non-power-of-two modulus), `MRG63k3a` in the same two types (32-bit chunks of
the same construction, with a modulus just under 2^63), the 64-bit
counter-based families in `UInt32` (the low half of a cipher word), `PCG64` and `PCG64DXSM` in `UInt32` (the low bits
of a permuted output -- DXSM ends in a multiplication, whose low bits depend on
fewer input bits than its high ones), and `Xoshiro256+` in `UInt32` (the low bits of an
additive scrambler).

This battery earns its place: a `UInt64` construction for MRG32k3a that passed
every `U(0,1)` battery failed here with six p-values at zero. Note also that
passing does not clear the lowest bits of the `+` scramblers, whose linear
weakness Blackman & Vigna document and SmallCrush does not resolve.

### The batteries that were built for bits

SmallCrush over a `UInt32` stream is a smoke test on the integer path, not the
instrument designed for it. TestU01's own bit-level batteries are **Alphabit**
and **Rabbit**, and both run on demand:

```bash
julia scripts/testu01/validate.jl --battery=alphabit --suite=bits
julia scripts/testu01/validate.jl --battery=rabbit --suite=bits --bits=4294967296
```

They differ from the Crush batteries in taking a *size*: `--bits` is how much
output to consume, 2³⁰ by default. At that size Alphabit costs about half a
minute — nine tests, seventeen statistics — and Rabbit about ten minutes, for
twenty-six tests and some forty. Both are close to linear in the size above a
fixed overhead, so four times the bits costs four times the time; Rabbit skips
its larger `MatrixRank` tests when the size is too small to feed them, which is
one reason not to shrink it far.

Alphabit examines bits *r+1* through *r+s* of each word, and the `bits` suite
delivers whole `UInt32`s, so the default (0, 32) uses all of them. Nothing stops
either battery running on the `single` and `interleaved` suites — `unif01`
extracts bits from a double as readily as from a word — and the campaign matrix
covers all three.

A replay has to consume the same number of bits as the run it replicates, so
`validate.jl` records `# bits = N` in the log header and `replay.jl` reads it
back from there; `--bits` overrides it.

## Running Crush and BigCrush

These are **not** part of `Pkg.test()` and must not become part of it. The
package test suite answers "does this installation work", and SmallCrush at
about a minute per battery is the most that belongs there. Crush takes hours
and BigCrush the better part of a day per generator: that is a validation
exercise, for a release or for a paper, not an installation check.

`scripts/testu01/validate.jl` runs any battery over any of the three suites, on
demand:

```bash
# what would run, without running it
julia scripts/testu01/validate.jl --list --battery=bigcrush --suite=all

# one generator, one battery, one suite
julia scripts/testu01/validate.jl --battery=crush --generator=Xoshiro256p --suite=bits

# the full single-stream sweep
julia scripts/testu01/validate.jl --battery=bigcrush --suite=single
```

| option | values | default |
|---|---|---|
| `--battery` | `smallcrush`, `crush`, `bigcrush`, `alphabit`, `rabbit` | `smallcrush` |
| `--suite` | `single`, `interleaved`, `bits`, `all` | `single` |
| `--generator` | a generator name, or `all` | `all` |
| `--streams`, `--values` | shape of the interleaved suite | 64, 1 |
| `--bits` | output consumed by `alphabit` and `rabbit` | 2³⁰ |
| `--out` | directory for logs | `testu01-results` |
| `--list` | print the plan and exit | |
| `--quiet` | summary only, no per-test reports | |

Each run writes three things. TestU01's own report goes to its own log:
per-test reports are on by default, because they are the canonical artefact to
archive and because they make a long run observable — the log fills as tests
complete, so progress can be followed with `tail -f` and an interrupted run
still leaves its completed tests behind. `summary.tsv` gets one line for the
run: battery, suite, generator, wall time, Julia version, CPU, the commit, and
how many statistics the battery reported, singled out and failed outright, so
the question "did anything happen" is answered without opening a log.
`pvalues.tsv` gets one line per statistic — index, name, p-value, class —
taken from `bbattery_pVal[]` rather than from the printed report, which shows
p-values in TestU01's own formatting (`eps`, `1 -  3.0e-9`). Classifying
against 10⁻¹⁰ is not something to do on text.

Run it under `nohup` or `tmux`:

```bash
nohup julia scripts/testu01/validate.jl --battery=bigcrush --suite=all > run.out 2>&1 &
tail -f testu01-results/bigcrush-*.log
```

## Reading a summary report

Every battery ends with a list of the p-values it singled out. That list is not
a verdict, and reading it as one is the easiest way to publish a wrong claim.

The threshold TestU01 prints against is `gofw_Suspectp`, whose default is
0.001: "any p-value less than α or larger than 1−α is considered suspect and is
*singled out*". It is a **printing** threshold. A battery of Crush's size
reports 144 statistics, so a campaign of 51 runs produces 7344 of them, and
0.002 × 7344 ≈ 15 lines are expected from chance alone — before any generator
has done anything wrong. (The statistics within a run are not independent, as
several tests compute more than one from the same stream; that widens the
spread but leaves the expected count unchanged.) The count also assumes that a
good generator's p-values are uniform on every statistic, and four of BigCrush's
are not: see [CollisionOver on small alphabets](#CollisionOver-on-small-alphabets).

The decision rule is in the TestU01 guide, under *Rejecting H₀*:

> When a p-value is extremely close to 0 or to 1 (for example, if it is less
> than 10⁻¹⁰), one can obviously conclude that the generator fails the test. If
> the p-value is suspicious but failure is not clear enough, (p = 0.0005, for
> example), then the test can be replicated independently until either failure
> becomes obvious or suspicion disappears.

So there are three outcomes, not two:

| | criterion | what to do |
|---|---|---|
| **failure** | p < 10⁻¹⁰ or p > 1 − 10⁻¹⁰ | report it; no replay needed |
| **suspect** | outside [0.001, 0.999] but inside [10⁻¹⁰, 1 − 10⁻¹⁰] | replay before calling it anything |
| **pass** | anything else | TestU01 does not print it |

Fix the number of replications before looking at them, and report the p-values
rather than a reject/do-not-reject verdict — the guide is explicit that this
"provides more information".

### Replaying the suspects

`scripts/testu01/replay.jl` re-applies only the tests a run singled out:

```bash
# one run: the newest complete log in that directory, its flagged tests, 3 times each
julia scripts/testu01/replay.jl --log=<out>/runs/crush-interleaved-Threefry4x64_20

# or by hand
julia scripts/testu01/replay.jl --battery=crush --suite=bits \
      --generator=Xoshiro256ss --tests=19 --reps=5

# the whole campaign at once
./scripts/run-campaign.sh replay --battery=crush --reps=3
./scripts/run-campaign.sh replay --battery=crush --dry-run   # what it would do
```

Three things about it are worth knowing.

**It costs minutes, not hours.** `bbattery_RepeatCrush(gen, rep)` applies test
*i* of the battery `rep[i]` times and skips the rest. Resolving the three
suspects of one Crush run took 106 seconds against the 30 minutes the run
itself had taken — which is what makes the same discipline affordable at
BigCrush, where a re-run would otherwise cost hours.

**The output must be disjoint.** `seed_words` is a fixed seed on purpose, so a
run can be reproduced exactly; re-running a suite the ordinary way therefore
reproduces the same p-values and settles nothing. `--skip` advances the stream
generator past everything the original consumed — 1024 streams by default, well
clear of the 64 the interleaved suite takes by default, and as many streams as a
larger shape takes — so the replay is the independent replication the guide
asks for. An interleaved suspect is replayed in the shape it arose in, which the
replay reads from the run's log. This is the package's own stream facility doing
the work the method requires.

**The p-values come back exact.** A replay writes `replay-pvalues.tsv` beside
its log, with every statistic it produced — not only the ones a battery would
single out, since what settles a suspect is the spread of the replications, and
the guide is explicit that reporting the p-values "provides more information"
than a verdict. The console prints the same list.

**The number beside a suspect is the test, not the statistic.** A battery
reports more statistics than it has tests (144 against 96 for Crush), and
TestU01 prints the *test* number, so several lines can carry the same one: five
lines reading `10  RandomWalk1 ...` are one test with five statistics, not
five tests. That number goes straight into `rep`, which is why no mapping table
is needed — but it also means "three suspect lines" and "three suspect tests"
are different counts.

### The Crush campaign, worked through

The 2026 Crush campaign — seventeen generators, three suites, 51 runs, with
Crush itself on the `bits` suite, as the campaign was shaped before Alphabit and
Rabbit were reachable — gives the rule something to bite on. (Its `U(0,1)`
results for the fifteen word-based generators predate the switch to the open
interval; they were measured again, below.) **No p-value came within four orders of
magnitude of 10⁻¹⁰**, so nothing was a failure outright. Eighteen lines were
singled out across fifteen runs, against ≈15 expected: an unremarkable count,
but a count is not an answer for any particular line.

So all eighteen were replayed, three times each on disjoint streams — 54
replications, about twelve minutes of work. **Not one produced a p-value
outside [0.001, 0.999].** The suspects were chance, every one of them, which is
what a campaign of this size is expected to manufacture and exactly what the
replay exists to establish rather than assume.

Two of them had looked worth the trouble. The most extreme p-value,
6.7 × 10⁻⁶ (`Xoshiro256ss`, bits, ClosePairs mNP), is more extreme than one
would typically see among 7344 statistics — probability ≈ 9%. And
`Threefry4x64-20` carried five of the eighteen across two of its three suites,
three of them in the interleaved run alone, where three or more in one run
arises in about 15% of campaigns this size. Replayed on streams 1025 and up,
those three gave:

| test | original | replay 1 | replay 2 | replay 3 |
|---|---|---|---|---|
| 19 ClosePairs mNP2, t = 3 | 1 − 2.9 × 10⁻⁵ | 0.9988 | 0.11 | 0.32 |
| 60 MatrixRank, 1200 × 1200 | 1.3 × 10⁻⁴ | 0.25 | 0.96 | 0.73 |
| 90 HammingIndep, L = 1200 | 8.6 × 10⁻⁴ | 0.79 | 0.65 | 0.45 |

Suspicion gone, in 106 seconds against the 30 minutes the run itself took.
That is the shape the write-up should take for each suspect: the original
p-value, the replications, and the p-values themselves — not a verdict.

### After the switch to the open interval

The `U(0,1)` suites of the fifteen word-based generators — xoshiro and
xoroshiro, PCG, the counter-based families — measured the conversion they used
before every float draw moved to `(0, 1)` (see
[Implementation Notes](implementation.md)). Those thirty runs, `single` and
`interleaved`, were repeated on the new conversion: Crush, at commit `3f67bc0`,
on an Intel Core i5-13500H, five at a time, ten hours in all. The MRG families,
whose `Float64` draws did not change, were not re-run. This was a first pass
with Crush; [the BigCrush sweep](#BigCrush-over-the-whole-matrix) followed.

Of 4320 statistics, **none came within five orders of magnitude of 10⁻¹⁰** —
the most extreme lay 4.3 × 10⁻⁵ from 1 — and ten lines were singled out,
against ≈8.6 expected from chance. All ten were replayed three times on
disjoint streams; not one replication left [0.001, 0.999]:

| generator | suite | test | original | replay 1 | replay 2 | replay 3 |
|---|---|---|---|---|---|---|
| PCG64 | single | CollisionOver, t = 20 | 1 − 4.3 × 10⁻⁵ | 0.20 | 0.73 | 0.61 |
| PCG64 | single | RandomWalk1 J, L = 10000 | 2.7 × 10⁻⁴ | 0.97 | 0.92 | 0.52 |
| Xoshiro256pp | single | WeightDistrib, r = 24 | 1.4 × 10⁻⁴ | 0.12 | 0.48 | 0.54 |
| Xoshiro256pp | interleaved | SerialOver, t = 4 | 3.6 × 10⁻⁴ | 0.62 | 0.75 | 0.76 |
| Xoroshiro128ss | single | MatrixRank, 300 × 300 | 1 − 1.6 × 10⁻⁴ | 0.81 | 0.63 | 0.32 |
| Xoroshiro128ss | single | RandomWalk1 J, L = 10000 | 1 − 1.7 × 10⁻⁴ | 0.041 | 0.33 | 0.088 |
| Xoshiro256p | interleaved | ClosePairs mNP1, t = 7 | 1 − 7.1 × 10⁻⁴ | 0.86 | 0.20 | 0.63 |
| Xoshiro256ss | single | ClosePairs NJumps, t = 2 | 1 − 7.4 × 10⁻⁴ | 0.25 | 0.69 | 0.81 |
| Threefry4x64-20 | single | AutoCor, d = 30 | 1 − 9.1 × 10⁻⁴ | 0.043 | 0.65 | 0.21 |
| Xoroshiro128p | single | RandomWalk1 J, L = 90 | 8.4 × 10⁻⁴ | 0.17 | 0.70 | 0.46 |

The same picture as the first campaign: a count at the level chance predicts,
and suspects that scatter on independent output.

### BigCrush over the whole matrix

The full matrix — the seventeen generators, BigCrush on the `single` and
`interleaved` suites, Alphabit and Rabbit at 2³⁰ bits on `bits`, 68 runs — was
run at commit `2b485cd` (v0.3.0), from a clean tree, on an Intel Core Ultra 9
285 (8 performance and 16 efficiency cores, no SMT) with Julia 1.12.1, 24 runs
at a time. A BigCrush run took 2.6 to 5.0 hours, mostly according to the kind of
core it landed on, with a peak resident set of 1.3 GB; Alphabit took half a
minute and Rabbit two to five. The sweep took eight hours of wall-clock time,
after a calibration run of 2.6 hours.

| battery, suite | runs | statistics | outside [0.001, 0.999] | failures |
|---|---|---|---|---|
| BigCrush, `single` | 17 | 2720 | 3 | 0 |
| BigCrush, `interleaved` (64 streams × 1) | 17 | 2720 | 5 | 0 |
| Alphabit, `bits` | 17 | 289 | 1 | 0 |
| Rabbit, `bits` | 17 | 680 | 2 | 0 |

Of 6409 statistics, **none was a failure outright**: the most extreme lay
2.8 × 10⁻⁶ from 1 (`PCG64DXSM`, Rabbit, MultinomialBitsOver), four orders of
magnitude short of 10⁻¹⁰. Eleven lines were singled out, against ≈12.8 expected
from chance. All eleven were replayed three times on streams 1025 and up, and
**none of the 33 replications left [0.001, 0.999]**:

| generator | suite | test | original | replay 1 | replay 2 | replay 3 |
|---|---|---|---|---|---|---|
| PCG64DXSM | bits, Rabbit | MultinomialBitsOver | 1 − 2.8 × 10⁻⁶ | 0.0033 | 0.96 | 0.79 |
| Xoshiro256p | bits, Rabbit | MultinomialBitsOver | 1.5 × 10⁻⁴ | 0.31 | 0.87 | 0.94 |
| Xoshiro512pp | single | CollisionOver, t = 21 | 1.5 × 10⁻⁴ | 0.97 | 0.37 | 0.12 |
| Philox4x32-10 | single | WeightDistrib, r = 0 | 1 − 1.7 × 10⁻⁴ | 0.035 | 0.41 | 0.30 |
| Xoshiro256p | single | Gap, r = 0 | 2.4 × 10⁻⁴ | 0.32 | 0.84 | 0.12 |
| Threefry4x32-20 | bits, Alphabit | RandomWalk1 J, L = 64 | 4.6 × 10⁻⁴ | 0.76 | 0.67 | 0.23 |
| MRG63k3a | interleaved | SampleMean, r = 10 | 1 − 6.4 × 10⁻⁴ | 0.12 | 0.13 | 0.57 |
| Xoroshiro128p | interleaved | ClosePairs NJumps, t = 5 | 7.3 × 10⁻⁴ | 0.74 | 0.62 | 0.36 |
| Xoroshiro128p | interleaved | CollisionOver, t = 14 | 1 − 4.2 × 10⁻⁴ | 0.29 | 0.51 | 0.71 |
| Philox4x32-10 | interleaved | HammingCorr, L = 1200 | 1 − 8.0 × 10⁻⁴ | 0.013 | 0.58 | 0.35 |
| Xoshiro256ss | interleaved | CollisionOver, t = 7 | 1 − 6.2 × 10⁻⁴ | 0.63 | 0.74 | 0.12 |

A replay recomputes every statistic of the tests it re-applies, 60 here, and one
that had not been singled out left the interval once: ClosePairs NP, t = 5, for
`Xoroshiro128p` interleaved, 0.21 in the original run and 0.58, 0.28 and 0.9994
in the replays — one extreme value in four, for a statistic unremarkable the
first time, against ≈0.12 such lines expected among 60.

### The interleaved suite in four shapes

L'Ecuyer et al. (2021, §6) suggest a power of two between 16 and 1024 streams,
and one to eight values from each in turn. The 28 power-of-two shapes in that
range would be 476 BigCrush runs; three were fixed instead, before anything was
computed — (16, 8), (256, 2) and (1024, 1), a diagonal from a few streams each
giving a run of consecutive values to the most stream starts side by side, one
value each. They ran at commit `a869931`, for the seventeen generators: 51
BigCrush runs of the `interleaved` suite, on the same machine and 24 at a time,
from 2.6 to 5.1 hours each. A suspect is replayed in the shape it arose in.

| shape (streams × values) | runs | statistics | outside [0.001, 0.999] | failures |
|---|---|---|---|---|
| 16 × 8 | 17 | 2720 | 9 | 0 |
| 64 × 1 (the sweep above) | 17 | 2720 | 5 | 0 |
| 256 × 2 | 17 | 2720 | 4 | 0 |
| 1024 × 1 | 17 | 2720 | 5 | 0 |

Of the 8160 statistics of the three new shapes, **none was a failure**: the most
extreme was 2.8 × 10⁻⁵ (`Xoshiro256ss`, 16 × 8, CollisionOver, t = 21). Eighteen
were singled out, against ≈16.3 expected, and each was replayed three times in
its own shape, on streams 1025 and up:

| generator | shape | test | original | replay 1 | replay 2 | replay 3 |
|---|---|---|---|---|---|---|
| Xoshiro256ss | 16 × 8 | CollisionOver, t = 21 | 2.8 × 10⁻⁵ | 0.40 | 0.67 | 0.32 |
| Xoroshiro128ss | 16 × 8 | MatrixRank, L = 5000 | 6.9 × 10⁻⁵ | 0.41 | 0.86 | 0.59 |
| Philox4x32-10 | 16 × 8 | CollisionOver, t = 14 | 8.3 × 10⁻⁵ | 0.044 | 0.58 | 0.38 |
| Xoroshiro128ss | 16 × 8 | RandomWalk1 J, L = 10000, r = 0 | 1.4 × 10⁻⁴ | 0.68 | 0.62 | 0.53 |
| Xoshiro512pp | 256 × 2 | ClosePairs mNP2, t = 3 | 1 − 2.7 × 10⁻⁴ | 0.71 | 0.39 | 0.73 |
| MRG32k3a | 16 × 8 | CollisionOver, t = 21 | 1 − 4.6 × 10⁻⁴ | 0.16 | 0.72 | 0.20 |
| MRG63k3a | 256 × 2 | MaxOft, t = 8 | 1 − 5.1 × 10⁻⁴ | 0.49 | 0.44 | 0.21 |
| Xoshiro256ss | 1024 × 1 | WeightDistrib, r = 28 | 5.9 × 10⁻⁴ | 0.076 | 0.68 | 0.21 |
| Xoshiro512pp | 256 × 2 | RandomWalk1 J, L = 50, r = 25 | 6.2 × 10⁻⁴ | 0.75 | 0.37 | 0.26 |
| Xoroshiro128pp | 1024 × 1 | SampleCorr, k = 2 | 6.3 × 10⁻⁴ | 0.34 | 0.16 | 0.13 |
| Xoshiro512p | 1024 × 1 | BirthdaySpacings, t = 7 | 6.7 × 10⁻⁴ | 0.45 | 0.57 | 0.63 |
| Xoroshiro128pp | 16 × 8 | SampleCorr, k = 2 | 7.6 × 10⁻⁴ | 0.76 | 0.64 | 1 − 1.1 × 10⁻³ |
| MRG32k3a | 16 × 8 | RandomWalk1 J, L = 10000, r = 15 | 1 − 7.6 × 10⁻⁴ | 0.029 | 0.75 | 0.66 |
| PCG64 | 16 × 8 | ClosePairs mNP2S, t = 16 | 8.3 × 10⁻⁴ | 0.27 | 0.31 | 0.83 |
| Xoroshiro128p | 256 × 2 | MatrixRank, L = 5000 | 8.8 × 10⁻⁴ | 0.064 | 0.023 | 0.20 |
| PCG64DXSM | 16 × 8 | MatrixRank, L = 30, r = 0 | 1 − 9.0 × 10⁻⁴ | 0.76 | 0.67 | 0.34 |
| Xoshiro256p | 1024 × 1 | Permutation, t = 7 | 9.0 × 10⁻⁴ | 0.17 | **8.0 × 10⁻⁴** | 0.13 |
| PCG64 | 1024 × 1 | RandomWalk1 J, L = 10000, r = 0 | 9.6 × 10⁻⁴ | 0.77 | 0.21 | 0.20 |

The replays recompute 135 statistics, and one left the interval, against ≈0.27
expected: the second replay of Permutation, t = 7, for `Xoshiro256p` at
1024 × 1, on the same side as the original. Together with SampleCorr, k = 2, for
`Xoroshiro128pp` — the only (generator, statistic) pair singled out in two
shapes, against ≈0.03 such pairs expected, and whose last replay at 16 × 8 came
close on the other side — that left three series that had not scattered cleanly
enough to stop at three. Each was replicated twenty more times, on streams 2049
and up (`replay.jl --skip=2048`), disjoint from the run and from the first
replay:

| generator | shape | test | min | median | max | Kolmogorov–Smirnov |
|---|---|---|---|---|---|---|
| Xoshiro256p | 1024 × 1 | Permutation, t = 7 | 0.034 | 0.47 | 0.997 | D = 0.14, p ≈ 0.81 |
| Xoroshiro128pp | 16 × 8 | SampleCorr, k = 2 | 0.14 | 0.51 | 0.92 | D = 0.14, p ≈ 0.78 |
| Xoroshiro128pp | 1024 × 1 | SampleCorr, k = 2 | 0.0020 | 0.54 | 0.95 | D = 0.11, p ≈ 0.97 |

All sixty stayed inside [0.001, 0.999], and each series is consistent with
uniformity: **no suspect persists**. With the 64 × 1 sweep, the interleaved
suite has passed BigCrush in four shapes, 10 880 statistics, for every
generator.

### The substreams suite in four shapes

The other construction of L'Ecuyer et al. (2021), the substreams of a single
stream, is the one cuRAND uses. It ran next, fixed before the sweep:

- **runs:** 20 BigCrush runs of the [`substreams`
  suite](#Dependence-between-streams), at commit `638e13c`, from a clean tree;
- **generators:** the four counter-based generators and `MRG32k3a`;
- **shapes:** the four shapes above;
- **machine:** the same one, 20 runs at a time, 4.1 to 5.2 hours each.

The streams are substreams 1 to S of each generator's first stream. For
`Philox4x32-10` that is the key (0, 0), and for `MRG32k3a` the seed 12345 × 6:
cuRAND's seed 0 in both cases, so these substreams are its subsequences, bit for
bit.

| shape (substreams × values) | runs | statistics | outside [0.001, 0.999] | failures |
|---|---|---|---|---|
| 16 × 8 | 5 | 800 | 3 | 0 |
| 64 × 1 | 5 | 800 | 2 | 0 |
| 256 × 2 | 5 | 800 | 3 | 0 |
| 1024 × 1 | 5 | 800 | 3 | 0 |

Of the 3200 statistics, **none was a failure**. The most extreme lay
5.6 × 10⁻⁵ from 1 (`Philox4x32-10`, 1024 × 1, HammingIndep, L = 300). Eleven were
singled out, against ≈6.4 expected. For comparison, the interleaved suite of the
same five generators in the same four shapes singled out four of its 3200. Three
of the eleven are CollisionOver at t = 21; see [CollisionOver on small
alphabets](#CollisionOver-on-small-alphabets). Each was replayed three times in
its own shape, on substreams 1025 and up of the same stream:

| generator | shape | test | original | replay 1 | replay 2 | replay 3 |
|---|---|---|---|---|---|---|
| Philox4x32-10 | 1024 × 1 | HammingIndep, L = 300, r = 0 | 1 − 5.6 × 10⁻⁵ | 0.065 | 0.070 | 0.33 |
| Philox4x64-10 | 64 × 1 | CollisionOver, t = 21 | 1.0 × 10⁻⁴ | 0.88 | 0.975 | 0.15 |
| MRG32k3a | 16 × 8 | CollisionOver, t = 21 | 1.9 × 10⁻⁴ | 0.075 | 0.81 | 0.99 |
| MRG32k3a | 256 × 2 | RandomWalk1 J, L = 50, r = 0 | 2.0 × 10⁻⁴ | 0.19 | 0.0015 | 0.76 |
| Philox4x32-10 | 1024 × 1 | ClosePairs mNP1, t = 5 | 1 − 2.0 × 10⁻⁴ | 0.68 | 0.59 | 0.9971 |
| Threefry4x64-20 | 16 × 8 | AppearanceSpacings, r = 0 | 4.3 × 10⁻⁴ | 0.0063 | 0.78 | 0.38 |
| MRG32k3a | 64 × 1 | ClosePairs NP, t = 9 | 7.5 × 10⁻⁴ | 0.15 | 0.30 | 0.053 |
| Philox4x32-10 | 16 × 8 | CollisionOver, t = 21 | 7.8 × 10⁻⁴ | 0.19 | 0.22 | 0.951 |
| Threefry4x32-20 | 256 × 2 | CollisionOver, t = 2 | 8.6 × 10⁻⁴ | 0.52 | 0.25 | 0.64 |
| Threefry4x32-20 | 256 × 2 | RandomWalk1 M, L = 50, r = 25 | 1 − 8.8 × 10⁻⁴ | 0.89 | 0.55 | 0.030 |
| Threefry4x64-20 | 1024 × 1 | RandomWalk1 J, L = 10000, r = 0 | 9.9 × 10⁻⁴ | 0.66 | 0.85 | 0.66 |

None of the 33 replications left [0.001, 0.999], and neither did any of the 99
statistics the replays recompute. Under the rule of the four shapes above, no
series called for more replications: no replay fell outside the interval on the
side of its original, and no (generator, statistic) pair was singled out in two
shapes. **No suspect persists.** Both constructions of the paper have now passed
BigCrush for the four counter-based generators and `MRG32k3a`, in four shapes
each.

### CollisionOver on small alphabets

Eight of the 37 BigCrush suspects above came from CollisionOver at t = 14 or
t = 21, which are four of BigCrush's 160 statistics. Tests 3 to 12 of BigCrush
are `smarsa_CollisionOver`. Each counts the collisions among n = 2 × 10⁷
overlapping t-tuples of symbols from an alphabet of d, in d^t = 2⁴² cells, N = 30
times. The five alphabets are (d, t) = (2²¹, 2), (2¹⁴, 3), (64, 7), (8, 14) and
(4, 21), each at two values of r. TestU01 compares the total with a Poisson law
of mean μ = N n²/(2 d^t) ≈ 1364.2, whose variance is μ too. But the tuples
overlap. A collision between the tuples starting at i and j is followed by one
between i + 1 and j + 1 whenever the next two symbols agree, which they do with
probability 1/d. Collisions therefore come in clumps, and the variance of their
number is (1 + 1/d)/(1 − 1/d) times μ:

- 5/3 for d = 4 (tests 11 and 12, t = 21);
- 9/7 for d = 8 (tests 9 and 10, t = 14);
- 1.03 for d = 64, and indistinguishable from μ beyond.

A calibration fixed in advance, decision rule included, put this to the test.
It ran at commit `405fe65` with `scripts/testu01/calibrate.jl`, and replicated
the tests on two generators:

- TestU01's own AES, which the battery calls with no Julia code in between;
- the package's `Philox4x32-10`, through the callback every campaign uses.

In the table, R is the variance of the count over μ:

| generator | d (tests) | replications | R | 95% interval | predicted |
|---|---|---|---|---|---|
| TestU01's AES | 2²¹ (3, 4) | 80 | 1.06 | [0.80, 1.49] | 1 |
| TestU01's AES | 8 (9, 10) | 280 | 1.34 | [1.14, 1.60] | 1.29 |
| TestU01's AES | 4 (11, 12) | 280 | 1.76 | [1.50, 2.10] | 1.67 |
| Philox4x32-10 | 4 (11, 12) | 72 | 1.85 | [1.36, 2.64] | 1.67 |

The mean count is μ in every arm, but the variance is not. The excess is the
same through the callback as inside the library (F-test, p = 0.78). TestU01's
MRG32k3a and the package's `MRG32k3a()`, which produce the same numbers, counted
exactly the same collisions. The excess therefore belongs to the test, not to
any generator. **For a perfect generator, each of tests 11 and 12 falls outside
[0.001, 0.999] about 1.7% of the time instead of 0.2%, and each of tests 9 and
10 about 0.6%**, by a normal approximation. In the calibration, 3.9% of the
p-values at d = 4 fell below 0.01 and 2.5% above 0.99.

That is what the tables above show. Over their 105 BigCrush runs:

- tests 11 and 12 singled out six statistics, against ≈0.42 expected if the
  test were calibrated and ≈3.5 given the variance measured;
- tests 9 and 10 singled out two, against ≈0.42 and ≈1.3.

None of these suspects persisted on replay. None could have become a failure
this way either: under the larger variance, a count at TestU01's 10⁻¹⁰ is still
4.9 standard deviations out, a few times 10⁻⁷ by the same approximation. The
verdicts stand. What does not stand is reading a CollisionOver p-value at t = 14
or 21 at face value. A correction, which compares the count with a variance of
5/3 μ or 9/7 μ, has not been applied to anything on this page.

```bash
# one job of the calibration: TestU01's AES under key 1, tests 3, 4 and 9 to 12
julia scripts/testu01/calibrate.jl --generator=aes --job=0 \
      --tests=3:2,4:2,9:7,10:7,11:7,12:7 --out=<dir>
```

Each job writes the TestU01 log and `collisions.tsv`, which has one line per
replication: the number of collisions, the p-value and the commit. The
calibration ran 20 such AES jobs (keys 1 to 20) and three `Philox4x32-10` jobs
(streams 4097 to 4099), 24 jobs at a time, in three hours.

## Long campaigns: running for days without a session

`validate.jl` runs one battery in one process, serially. That is the wrong shape
for a BigCrush sweep: the full matrix is 68 runs — seventeen generators on the
`single` and `interleaved` suites, plus Alphabit and Rabbit on the `bits` suite
— most of them taking the better part of a day, so a single process means weeks
of sequential work in which one crash loses everything and a lost SSH session
ends the campaign.

`scripts/testu01/campaign.jl` runs the same matrix as independent OS processes:

```bash
# what would run, and what is already done
julia scripts/testu01/campaign.jl --battery=bigcrush --status

# measure one job before committing to the sweep
julia scripts/testu01/campaign.jl --battery=bigcrush --calibrate

# the sweep, six jobs at a time
julia scripts/testu01/campaign.jl --battery=bigcrush --jobs=6
```

`--battery` is required: there is deliberately no default, so that a stray
invocation cannot start a two-week sweep.

**A campaign need not run the whole matrix.** `--suite` keeps some of the three
suites, and `--shapes=<streams>x<values>[,...]` gives the interleaved suite one
job per shape. `run-campaign.sh` records both next to the results, so that
`status`, `stop` and the watchdog work against the matrix that was started, and
it refuses a different matrix in the same directory rather than mix the two.
With `--jobs` given, `--skip-calibration` starts the sweep without first running
one job alone to measure it — at BigCrush, hours during which every other core
waits.

**The `bits` leg does not run `--battery`.** Crush and BigCrush examine the 30
most significant bits of a `U(0,1)` output, so over an integer stream they are
the wrong instrument — running one there was what the campaign did for want of
an alternative. A campaign now runs **Alphabit and Rabbit** on the `bits` suite
whatever `--battery` says, which is where the extra seventeen runs come from.
`--bits-battery=<name>[,<name>]` overrides that, and passing the same name as
`--battery` restores the old uniform matrix; `--bits=<n>` sets the size those
two consume. `validate.jl` stays orthogonal — it runs any battery over any
suite, and this pairing is the campaign's policy, not the library's.

**Resuming is the point.** `summary.tsv` records one line per finished
`(battery, suite, generator)`, and for the interleaved suite per shape. Re-running the same command skips those and does
the rest, so an interruption costs only the jobs actually in flight — never the
weeks already spent. Each job also writes its own `summary.tsv` in its own
directory, which the driver reconciles, so even a driver killed mid-flight
loses nothing that finished.

**Stopping cleanly.** `touch <out>/STOP` stops new jobs from starting and waits
for the running ones, which is how to end a campaign without discarding a
battery that is twenty hours in. Ctrl-C does the same.

**Surviving logout and reboot.** `setsid nohup` survives a logout; `tmux` does
too and lets you reattach, unless it moves its panes into the systemd user
manager ([below](#When-enable-linger-is-not-yours-to-run)); neither survives a
reboot or restarts a driver that died. For a campaign measured in weeks, use the
systemd *user* unit in `scripts/testu01/randomdatastreams-campaign.service`:

```bash
loginctl enable-linger $USER          # user services keep running after logout
cp scripts/testu01/randomdatastreams-campaign.service ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/randomdatastreams-campaign.service   # set paths
systemctl --user daemon-reload
systemctl --user start randomdatastreams-campaign
journalctl --user -u randomdatastreams-campaign -f
```

The unit stops with `SIGINT` and a six-hour stop timeout, so `systemctl --user
stop` lets the driver wait for its children rather than killing them, and
`Restart=on-failure` brings back a driver that died — it resumes from
`summary.tsv` like any other invocation.

### When `enable-linger` is not yours to run

That first line is the one that needs an administrator. Without it, systemd
kills user services at logout, which is exactly what a two-week sweep must
survive. On a shared machine the answer is often "no", so
`scripts/run-campaign.sh` can supervise the campaign without systemd:

```bash
./scripts/run-campaign.sh run --battery=crush --supervisor=detached --watchdog
./scripts/run-campaign.sh status --battery=crush      # no flag needed afterwards
./scripts/run-campaign.sh stop   --battery=crush
```

`--supervisor=detached` starts the driver with `setsid nohup`, in a session of
its own, and records its pid in `<out>/driver.pid`; its output goes to
`<out>/supervisor.log`. `stop` sends `SIGINT` to its process group, which holds
the jobs as well — what Ctrl-C does in a terminal. `--supervisor=auto`, the
default, uses systemd when linger is on or can be turned on, and falls back to
`detached`. The choice is recorded in the output directory, so `status`, `stop`
and `collect` find the campaign without being told again, and a `start` that
names another supervisor records that one instead.

What neither gives you is what `--watchdog` restores, through a *user* crontab
entry that needs no privileges:

| | systemd unit | detached | detached + `--watchdog` |
|---|---|---|---|
| survives logout | with linger | yes, unless `KillUserProcesses=yes` | same |
| survives a reboot | yes | no | yes (`@reboot`) |
| restarts a driver that died | `Restart=on-failure` | no | yes, within 15 min |
| watch it work | `journalctl -f` | `tail -f <out>/supervisor.log` | same |

**tmux, where it still works.** `--supervisor=tmux` runs the driver in a
detached tmux session you can attach to, and was the fallback until it took a
BigCrush sweep down. A tmux built with systemd support — 3.6 on the machine
where it happened — moves each pane it spawns into a scope of its own under the
systemd user manager, `user@<uid>.service`, so that the pane outlives the
terminal it came from. Without linger that manager is stopped when your last
session ends, and every scope under it with it; `KillUserProcesses=no` does not
help, since it spares what stays in a session's scope and the pane has left it.
The last logout of 2026-10-16 took the driver and the 24 BigCrush jobs it was
running, some five hours in. `start` now asks the pane where it landed, from
`/proc/<pid>/cgroup`, and when that is under the user manager with linger off it
stops the session again, before a job has started, and says to use `detached`.
A detached driver stays in the scope of the login or the cron job that started
it.

Two details the script handles, both of which are silent failures otherwise.
`KillUserProcesses=yes` in `logind.conf` kills a detached driver or tmux session
at logout just as it kills an unlingered service, so neither is a workaround on
such a machine — `run-campaign.sh` reads the setting and says so before you
commit two weeks to it. And tmux keeps its socket under `/run/user/$UID`, a
directory that is removed when your last session ends; the script points
`TMUX_TMPDIR` at a directory under `$HOME` instead, so the session started from
your login shell and the one the watchdog looks for are the same session.

The watchdog respects `touch <out>/STOP`: a campaign you stopped on purpose
stays stopped. It also leaves a finished campaign alone: the driver writes
`<out>/DONE` once every job of the matrix is in `summary.tsv`. `stop` removes
the crontab entry, and any crontab it edits is
backed up to `<out>/crontab.backup` first — your other cron jobs are never
touched.

### A home directory that answers to a ticket

Where `$HOME` is an NFS mount secured with Kerberos — `sec=krb5` in the output
of `findmnt -T ~ -o OPTIONS` — the files answer to a *ticket*, not to your uid.
When the ticket lapses, every process you own loses the directory at the same
instant: the driver in the middle of writing a result, and `cron` before it can
so much as read the watchdog's copy of this script. Tickets are measured in
hours and campaigns in days, so this is not a corner case. It is what happens
by default to a sweep left to run.

Nothing in the campaign reports it, because nothing in the campaign can still
write anywhere to report it. From outside, it looks like a watchdog that
quietly stopped firing; the system's cron log tells the real story, one line
per attempt:

```
CROND[91468]: (CRON) ERROR chdir failed (/u/you): Permission denied
```

`prepare` recognises the case. It refuses to begin without a valid ticket, and
gives the campaign **a ticket cache of its own**: a copy of yours, at
`/tmp/krb5cc_<uid>_randomdatastreams-<battery>`, renewed every 30 minutes by a
`krenew` that no login owns, and named in the launcher:

```bash
export KRB5CCNAME="FILE:/tmp/krb5cc_<uid>_randomdatastreams-bigcrush"
```

Your own cache will not do wherever each login gets one —
`/tmp/krb5cc_<uid>_XXXXXX`, made by `sshd` or `pam_sss` and destroyed at
logout. A driver started under a `krenew` that inherited it fails with
`No credentials cache found` once you have disconnected; a `krenew -b` started
from your shell goes the same way; and `cron` never sees a `KRB5CCNAME` at all,
and looks for `/tmp/krb5cc_<uid>`, which nobody made. That is what held up the
first BigCrush sweep for six hours one night, the calibration finished and every
core idle, with the watchdog failing every fifteen minutes. The NFS client needs
no telling: `rpc.gssd` looks in `/tmp` for a cache you own whose name starts
with `krb5cc_`. The watchdog brings the campaign's `krenew` back if it dies while
the ticket still holds, and `stop` destroys the cache, which existed only for
the campaign. Where your cache is not a file (`KEYRING:`, `KCM:`) there is
nothing to copy, and the driver runs under `krenew` on the cache you have; such
caches usually outlive a login, but check that yours does before leaving a sweep
to run.

Two limits are worth knowing before committing days to a sweep. First,
**renewal has its own deadline**: a renewable ticket may be renewed only until
the `renew until` that `klist -f` prints, typically some weeks after the
`kinit` that created it, and a ticket that is not renewable at all cannot be
rescued — `prepare` warns in both cases. `prepare` and `start` copy the ticket
you have at the time, so a fresh login followed by either restarts that clock.
Second, **a reboot clears `/tmp`** on most machines, and the campaign's cache
with it: the watchdog's `@reboot` line then cannot so much as enter `$HOME`, and
the campaign waits for you to log in and run `start`.

**A lock that never wakes.** One more thing goes wrong on such a home, and has
nothing to do with tickets. Every job used to begin with `Pkg.activate`, which
writes `~/.julia/logs/manifest_usage.toml` under a pid lock and, while another
process holds it, waits on a file watcher. Of 24 BigCrush jobs started within
the same minute on an NFS home, six waited there for nine and a half hours,
until they were killed: the watcher never fired. Julia prints a process's
stacks on `kill -USR1 <pid>`, and again when `SIGTERM` kills it; all six stood
at the same place. The driver resolves the environment once,
before any job starts, so a job whose Manifest already points at the checkout
now sets its active project directly and calls no Pkg at all. Only a stale
Manifest still goes through Pkg, lock included, and the driver's own pass is
what keeps it from being stale.

The alternative dissolves the question rather than answering it — put the
checkout and the results on a filesystem that needs no ticket, such as local
scratch space:

```bash
./scripts/run-campaign.sh prepare --battery=bigcrush --out=/var/tmp/$USER-bigcrush
```

**Every result names the code that produced it.** `summary.tsv` records the
package version, the commit of `HEAD` and whether the working tree was clean,
alongside the Julia version, the CPU and — for the interleaved suite — the
number of streams and values per stream. Each battery log repeats it in its
header. A campaign started from a dirty tree prints a warning before it begins,
because a run measured in weeks cannot be repeated to find out afterwards which
code produced it. The same line appears at the top of the throughput benchmark's
report.

**Do not benchmark and validate at the same time.** The throughput figures in
the implementation notes are minima over samples on an unloaded machine; a host
running six BigCrush processes will produce numbers that measure the scheduler.
Run `scripts/benchmarks/throughput.jl` first, pinned, then start the campaign.
On a hybrid CPU (Intel 12th generation and later) pin the benchmark to a
performance core with `taskset`; the batteries can use any core, since their
results do not depend on how fast they were produced.

### What BigCrush settled, and what it did not

Two things the suite cannot settle at SmallCrush level, and where the
[BigCrush sweep](#BigCrush-over-the-whole-matrix) leaves them:

- **The low bits of the `+` scramblers.** Blackman & Vigna document the lowest
  bits of `xoshiro256+` and `xoroshiro128+` as linearly weak, and the `bits`
  suite hands exactly those bits to the batteries: a `UInt32` is the low half of
  the word. Alphabit and Rabbit at 2³⁰ bits passed on all three `+` generators,
  and BigCrush passed on their `U(0,1)` output. That narrows the question
  without closing it: a pass is not a proof, and a weakness confined to the
  lowest bit or two can be diluted in a test of the whole 32-bit word.
- **Dependence between streams.** The interleaved suite is the construction
  L'Ecuyer et al. (2021) call for, and the paper says batteries for
  counter-based generators still need development. BigCrush passed on it for
  all seventeen generators in [four shapes](#The-interleaved-suite-in-four-shapes),
  from 16 streams of eight values to 1024 streams of one. For the counter-based
  generators each of those streams has a key of its own. The other
  construction the paper describes, the substreams of a single key, is how
  cuRAND hands out its Philox and MRG32k3a streams. It passed BigCrush [in the
  same four shapes](#The-substreams-suite-in-four-shapes) for the four
  counter-based generators and `MRG32k3a`, but has not been run on the other
  twelve.

Results belong in the JOSS submission, not in CI.

## How the batteries are driven

Every battery on this page is a direct call into the C library, through the
`TestU01_jll` artifact and one layer of our own in `test/tu01.jl`. That layer is
what the test suite and both scripts use: it builds and frees the `unif01_Gen`,
runs the batteries, and reads their answers out of `bbattery_pVal[]`,
`bbattery_NTests` and `bbattery_TestNames[]`. It lives under `test/` rather than
beside the scripts because `Pkg.test()` has to be self-contained, while a script
may reach anywhere in the checkout it belongs to.

It is deliberately small — batteries, the repeat batteries, the two printing
globals, and the generator's lifetime — and everything in it is there because
the alternative, [RNGTest.jl](https://github.com/JuliaRandom/RNGTest.jl), gave
one of the following instead. The package depended on it until 2026-09; the
record is kept because each point is a reason the code looks as it does, and
because someone reaching for the wrapper again should know what they are taking
on.

**The batteries returned nothing.** `smallcrushTestU01` and its siblings are
`Cvoid` calls, and the three globals holding the answer are not exposed, so
every p-value had to be recovered from the formatted report — as text, in
TestU01's printing forms. That is a poor thing to classify against a threshold
of 10⁻¹⁰, and it made the log the only artefact a run produced. `pvalues.tsv`
exists because the globals do.

**The test suite could not run the real battery, for that same reason.** It
drove `RNGTest.smallcrushJulia`, a Julia reimplementation that applies
SmallCrush's ten tests one at a time and returns their p-values — the only way
to assert on them. It is not the same object: it keeps a subset of the
statistics (`sstring_HammingIndep` contributes its `Mean` alone), and it
distributes the ten tests with `pmap`, so under `julia -p N` the generator
closure would be serialised to each worker and every test would start from the
same state — ten tests on one stream. The suite now runs `bbattery_SmallCrush`
and asserts on all 15 statistics. That costs about 15% more per battery — 12.1 s
against 10.6 s, measured on one generator in one process — which is the price of
running the battery the literature names rather than something close to it.

**The callback was not rooted.** `@cfunction($f, ...)` returns a
`Base.CFunction`, which Base documents as the "garbage-collection handle" that
must outlive every C call through the pointer; `RNGTest.Unif01` builds it in a
local and drops it when the constructor returns. The callback allocates, so the
GC does run during a battery. It was never seen to crash — but nothing made it
safe, and a battery that dies at hour eighteen of a BigCrush costs a day. `Gen`
holds the handle and the closure for as long as the generator lives.

**A `Function` generator lost its first 100 values.** `Unif01(f::Function,
name)` drew a hundred to check the return type and the range before TestU01 saw
the generator, and each test of `smallcrushJulia` paid it again, since every
test built its own. The layer checks the *inferred* return type instead, which
costs nothing, catches the same mistake — a callback inferred as `Any` is a
segfault, not a type error, once C is calling it — and leaves the stream at its
origin.

**`swrite_Basic` was cleared at load, and written back eight bytes at a time.**
The flag is a `lebool`, which `gdef.h` typedefs to `int`; RNGTest stores a
pointer-sized value into it, as did the copy of that trick this repository was
carrying. `TU01.verbose!` writes a `Cint`.

**The repeat and bit-level batteries had no wrapper at all.** `bbattery_Repeat*`
is what makes [Replaying the suspects](#Replaying-the-suspects) affordable, and
`Rabbit`, `Alphabit` and `BlockAlphabit` are the batteries actually meant for a
bit source, and the `bits` suite had been running SmallCrush over a `UInt32`
stream for want of them. `--battery=alphabit` and `--battery=rabbit` now run
them, replays included; see [The batteries that were built for
bits](#The-batteries-that-were-built-for-bits). `BlockAlphabit` is in the layer
but not wired to the command line: its repeat form takes a block width as well,
and nothing here has asked for it yet.

Two rules the layer enforces that the C library only documents. **One generator
at a time**: TestU01 keeps its state in globals and crashes if a second
`unif01_Gen` is created, so `Gen` refuses to build one and names the generator
still live; it is also why `campaign.jl` runs its matrix as independent OS
processes rather than as tasks. **The callback must be inferred**, as `Float64`
for a `U(0,1)` generator or `UInt32` for a bit generator — the generator tables
here have an abstract element type, so every callback goes through a function
barrier, and `Gen` rejects one that did not.

What the layer does not cover is the individual tests: the `s*` modules hold
some fifty of them, and nothing here calls one directly, because a battery is
what this package validates against. Should one ever be needed, the same
artifact ships `libtestu01extractors`, the shim that reads a p-value out of an
opaque `sres_*` struct, which is the larger part of what RNGTest is for.
