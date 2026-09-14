# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **TestU01 is driven directly, and RNGTest.jl is no longer a dependency.**
  `test/tu01.jl` calls the same `TestU01_jll` artifact the wrapper used, and
  covers what this package runs: the three Crush batteries, the `Repeat`
  batteries, the bit-level batteries, and the three C globals — `bbattery_pVal`,
  `bbattery_NTests`, `bbattery_TestNames` — that hold a battery's answer and
  that RNGTest exposes none of. Every p-value is now the double the library
  computed rather than a number parsed back out of a formatted report, which is
  what a threshold of `1e-10` needs. Each run writes `pvalues.tsv` (one line per
  statistic, with its class), `summary.tsv` gained the counts and the most
  extreme p-value, and a replay reports the p-values of the tests it repeated.
  The test suite now runs `bbattery_SmallCrush` itself and asserts on all 15 of
  its statistics, in place of RNGTest's Julia reimplementation — about 15% more
  per battery, and the battery the literature names. Two hazards inherited from
  the wrapper are gone with it: the `@cfunction` handle TestU01 calls through is
  now rooted for the generator's lifetime, and `swrite_Basic`, a four-byte
  `int`, is no longer written eight bytes wide.

- **A campaign's `bits` suite runs Alphabit and Rabbit, not `--battery`.** Crush
  and BigCrush examine the 30 most significant bits of a `U(0,1)` output and
  were never built for an integer stream; they were what the `bits` leg ran for
  want of an alternative. `campaign.jl` now pairs that leg with the bit-level
  batteries — 68 runs instead of 51 — and `--bits-battery` overrides it, with
  the same name as `--battery` restoring the old uniform matrix. `validate.jl`
  stays orthogonal: any battery, any suite.

### Added

- **Alphabit and Rabbit**, TestU01's bit-level batteries, which RNGTest never
  wrapped. `validate.jl --battery=alphabit|rabbit`, sized by `--bits` (2^30 by
  default), with replays: a replay consumes the same number of bits as the run
  it replicates, which the log header records. Measured on one generator at that
  size, Alphabit is 17 statistics in about half a minute and Rabbit some forty
  in about ten.

- **A campaign supervisor that does not need `loginctl enable-linger`.**
  `scripts/run-campaign.sh` takes `--supervisor=auto|systemd|tmux`: systemd when
  linger is on or can be turned on, tmux otherwise, for the common case where
  linger is an administrator's call and the answer is no.
  `--watchdog` adds a *user* crontab entry that restores what systemd was
  giving — restart after a reboot, restart after a driver that died — with no
  privileges. The supervisor is recorded next to the results, so `status`,
  `stop` and `collect` need no extra flag; `stop` removes the watchdog again;
  and `touch <out>/STOP` is respected, so a campaign stopped on purpose is not
  restarted fifteen minutes later. Two silent failures are handled rather than
  left to be discovered: `KillUserProcesses=yes` kills a detached tmux session
  at logout exactly as it kills an unlingered service, which the script reports
  before a two-week sweep rather than after; and tmux's socket lives under
  `/run/user/$UID`, which disappears with your last session, so the sockets are
  moved under `$HOME`.

- **MRG63k3a** (L'Ecuyer 1999, Table II, fourth entry): `MRG63k3a` and
  `MRG63k3aGen`, the 64-bit-arithmetic member of the MRG32k3a family. Two
  moduli just under `2^63`, period ≈ `2^377`, and just under 63 random bits per
  step against 32 for MRG32k3a — so a `UInt64` costs two steps instead of four,
  and its 32-bit chunks depart from uniformity by `2^-31` rather than `2^-16`.
  Output matches L'Ecuyer's own C implementation value for value. The package
  was testing MRG32k3a's assembled 64-bit words without shipping the generator
  that has 64 bits to give; now it ships both.

  The stream and substream distances, `2^250` and `2^150`, are this package's
  choice: L'Ecuyer published jump matrices for MRG32k3a only. They are his
  `2^127` and `2^76` scaled by the ratio of the two periods, and the matrices
  are computed at precompilation rather than tabulated. `advance_state!`,
  `next_substream!`, `next_stream!`, `get_state`/`set_state!`, `srand!` and
  `Random.seed!` behave exactly as for MRG32k3a.

  The draw avoids a 128-bit remainder: both moduli are `2^63 - c` with `c`
  small, so the reduction is a shift, a multiply, an add and one conditional
  subtraction — a plain remainder is ten times slower, measured. All three of
  Vigna's optimizations for MRG32k3a are in place, including the one this
  package had measured and rejected for the 32-bit generator: keeping the state
  one step ahead so the output overlaps the next state is worth nothing there
  (236 against 235 M/s) and 7% here (196 against 183). The shift is hidden
  behind the API — `get_state` and `show` step back, the constructors and
  `Random.seed!` step forward, and the jump matrices commute with it, so
  streams, substreams and `advance_state!` are untouched.

  It joins every generator list in the suite: SmallCrush on one stream, on 64
  interleaved streams, and on the `UInt32` and `UInt64` bit paths; the threads
  suite; the uniform-interface matrix; the on-demand Crush/BigCrush harness; and
  `notebooks/streams_tour.ipynb`, which now also shows the two MRG word sizes
  side by side.

### Fixed

- **Ranges on the xoshiro and PCG families were sampled by folding.**
  `rand(rng::LinRNG, ::UnitRange{Int64})` and its `PCGRNG` twin reduced one
  `UInt64` draw with `%`, which biases the low values of a range whose length
  does not divide `2^64` by about `n/2^64`, and which threw `DivideError` on an
  empty range where every other family — and the standard library — throws
  `ArgumentError`. Both methods are gone; ranges now go through `Random`'s
  sampler, which rejects rather than folds, for all seventeen variants. The
  uniform-interface matrix checks the bounds and the empty-range error.

- **`show` was multi-line on two arguments.** `show(io, x)` is what string
  interpolation, `@show`, container display and error messages call, so
  `"$rng"` spread one generator over four lines and a `Vector` of them was
  unreadable. The two-argument method is now a one-line summary; the three
  anchors moved to `show(io, ::MIME"text/plain", x)`, so REPL and notebook
  display are unchanged. All ten types, streams and generator objects alike.

## [0.2.0]

Two new generator families, and one breaking change to seeding.

### Breaking

- **The all-zero state is now refused everywhere.** It is a fixed point of the
  xoshiro and xoroshiro transitions: a generator seeded with it emits zeros
  forever. Constructors, `srand!`, `set_state!` and `Random.seed!` now throw an
  `ArgumentError` instead of accepting it. The documentation already claimed
  this was rejected; the code did not do it. Code that seeded with an all-zero
  vector was producing constant output and will now fail loudly.

### Added

- **PCG** (O'Neill 2014): `PCG64` and `PCG64DXSM`, with `PCG64Gen` and
  `PCG64DXSMGen`. Raw 64-bit outputs match NumPy's, whose `default_rng()` is
  PCG64, for the same state. Streams and substreams use the closed-form LCG
  advance, so `advance_state!` costs `O(log n)` multiplications at any
  distance, forwards or backwards.
- **Counter-based generators** (Salmon et al. 2011): `PhiloxRNG` (4x32-10),
  `Philox4x64RNG`, `Threefry4x32RNG` and `Threefry4x64RNG`, with their
  generator objects, validated against the authors' test vectors. Streams are
  distinct keys rather than segments of one orbit, so any draw is addressable
  from `(key, substream, index)` without replaying anything.
- **A uniform seeding rule across every family.** Each generator now accepts an
  integer seed, expanded through splitmix64, and a raw seed in its own natural
  form. `MRG32k3a(12345)`, `Xoshiro256ppGen(12345)`, `PCG64(12345)` and
  `Philox4x64Gen(12345)` all work and mean the same thing.
- `set_state!` for `MRG32k3aGen`, and `srand!` for generator objects, closing
  gaps that made the interface non-uniform.
- `scripts/testu01/campaign.jl`: a resumable, parallel driver for the TestU01
  batteries, with a systemd unit for campaigns that outlive a login session.
- `scripts/benchmarks/throughput.jl`: a reproducible throughput benchmark that
  states its measurement method.
- Provenance in every result file: the commit, whether the tree was dirty, and
  the package version that produced each measurement.
- Documentation: a validation page, a generator comparison with a related-work
  section, an FAQ, and `notebooks/streams_tour.ipynb`, a runnable tour of the
  stream model across all four families.

### Fixed

- `advance_state!(rng::MRG32k3a, e, c)` returned the value of its last
  assignment instead of the generator, unlike every other family. It now
  returns `rng`.
- PCG streams used power-of-two jump distances, which for a linear congruential
  generator makes the jump multiplier congruent to 1 modulo a large power of
  two, leaving the streams correlated. Each stream passed SmallCrush alone
  while sixty-four interleaved failed with fourteen of fifteen p-values at
  zero. Stream and substream distances are now odd, `2^32(2^64+1)+1` and
  `2^64+1`. Found by the interleaved battery, which is in the test suite.

## 0.1.0

Initial release: `MRG32k3a` and the xoshiro/xoroshiro families, with the stream
and substream object model of L'Ecuyer et al. (2002). Registered without a git
tag, which is why there is no comparison link for it below.

[Unreleased]: https://github.com/JLChartrand/RandomDataStreams.jl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/JLChartrand/RandomDataStreams.jl/releases/tag/v0.2.0
