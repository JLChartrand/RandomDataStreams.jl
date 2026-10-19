# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0]

Normals and exponentials are drawn by inversion on every generator, with the
same bits on the CPU and the GPU, and the package reproduces cuRAND's Philox
and MRG32k3a streams word for word.

### Breaking

- **Every normal and exponential sequence changes.** `randn`, `randexp` and
  their `!` forms draw by inversion, on the CPU and, for the counter-based
  generators, on the GPU; the uniforms do not change. Code that must reproduce
  normals or exponentials drawn under 0.3 needs to stay on 0.3.
- **On the GPU, `randn!` is no longer Box-Muller.** Code that wants the old
  Box-Muller fill calls `randn_boxmuller!` (Philox4x32, `Float64` and
  `Float32`, as before).

### Changed

- **`randn` and `randexp` draw by inversion, on every generator.** `randn(rng)`
  is now `norminv(rand(rng, Float64))` and `randexp(rng)` is
  `-log(1 - rand(rng, Float64))`, in place of the standard library's Ziggurat:
  one uniform per variate, through a monotone map. Antithetic variates, common
  random numbers and randomized quasi-Monte Carlo now carry through from the
  uniforms to the variates -- on the word generators' grid, reflecting a uniform
  negates its normal exactly. **Every normal and exponential sequence changes**;
  the uniforms do not. A normal costs more: about 4.5 ns over its uniform on
  Xoshiro256++, against 1.2 ns for the Ziggurat. The manual's "Normal variates
  and variance reduction" tabulates the comparison.

- **On the GPU, `randn!` and `randexp!` are the CPU's, bit for bit**, for the
  four counter-based generators and `Float64`, `Float32` and `Float16`, from
  any position, mid-block included. They were Box-Muller, Philox4x32 only and
  block-aligned, matching nothing on the CPU. Both maps are computed with
  correctly rounded operations only -- an fdlibm logarithm instead of libm's or
  libdevice's, and every product feeding a sum written as an fma, since the GPU
  compiler fuses what the CPU's does not. A `Float32` normal is computed in
  `Float64`, as on the CPU, which a consumer GPU runs slowly: about 0.6 billion
  a second on an RTX A1000.

- **GPU fills compute each block of the generator once.** A large `rand!` or
  `randn32!` into a `CuArray` runs one thread per block of the counter-based
  generator, writing every draw whose words lie in it, where one thread per
  element recomputed the whole block for each word. On an RTX A1000, `rand!`
  in `Float32` from Philox4x32 doubles to the memory bound (41 billion a
  second, cuRAND's 45), the 64-bit ciphers run three to four times faster in
  `Float32` (Threefry4x64 three times in `Float64` too, Philox4x64 1.4 times),
  and `randn32!` 1.3 to 3 times. Fills under 2^16 blocks, and `randn!`/`randexp!`,
  whose `Float64` inversion is the work, keep one thread per element. The
  values do not change. `scripts/benchmarks/gpu_fills.jl` measures every fill.

- **Faster `Float32` and `Float16` normal fills on the CPU.** `randn!` and
  `randexp!` into a narrower array went through the standard library's
  one-draw-at-a-time fallback, which cost a counter-based generator twice its
  `Float64` rate. A counter-based generator now fills block by block, through
  the function its GPU kernel uses, and the others draw eight uniforms onto the
  stack at a time. On one P-core of an Intel Core Ultra 9 285, `randn!` into a
  `Float32` array went from 14.7 to 10.5 ns a value with Philox4x32, 21.4 to
  11.4 with Threefry4x64, 12.4 to 8.5 with MRG32k3a, and 5.3 to 4.7 with
  Xoshiro256++, and `randexp!` from 6.9 to 2.9 with Xoshiro256++; still without
  allocation, and with the same values. `randn32!`, whose uniforms already fit
  the array, keeps `rand!` and Φ⁻¹ in place, except where the uniforms are dear
  enough for one pass to win: Threefry4x64 11.8 to 9.4 ns, Philox4x64 8.4 to
  7.2, MRG32k3a 8.9 to 8.4.

- **Box-Muller moves to `randn_boxmuller!`** (Philox4x32, `Float64` and
  `Float32`, as before), and **`randn_inversion!` is deprecated**: it forwards to
  `randn!`, which is inversion now.

### Added

- **`randn32` and `randn32!`: single-precision normals by inversion**, the
  `Float32` Φ⁻¹ (AS 241's PPND7) of `rand(rng, Float32)`, in `Float32`
  arithmetic throughout, on every generator and, through the CUDA extension, on
  a `CuArray{Float32}` with the CPU's values bit for bit. About eighteen times
  `randn!`'s `Float32` rate on an RTX A1000 (about 11 billion a second with
  Philox4x32), for tails that stop at ±5.29 instead of ±8.21. `norminv` on a
  `Float32` is this single-precision map.

- **`norminv(u)`**, the standard normal quantile function `randn` uses
  (Wichura's AS 241, relative error below 5e-16), exported so that the points of
  a randomized quasi-Monte Carlo rule, or a kernel's own uniforms, map the same
  way, with the same bits on a GPU.

- **Campaigns over several interleaved shapes.** `scripts/testu01/campaign.jl`
  takes `--shapes=<streams>x<values>[,...]` and runs the interleaved suite once
  per shape, which is how the heavier configurations of L'Ecuyer et al. (2021)
  — up to 1024 streams, up to 8 values per stream — reach a campaign rather than
  one `validate.jl` call at a time. A finished job is now recognised by its
  shape as well, and a summary line without one is read as the 64 × 1 shape
  every earlier campaign ran, so existing results directories resume as before.
  `scripts/run-campaign.sh` passes `--suite` and `--shapes` through, records
  them next to the results for `status`, `stop` and the watchdog, and refuses
  another matrix in the same directory. Its `--skip-calibration` starts a sweep
  with `--jobs` given without first running one job alone.

- **A calibration harness for BigCrush's CollisionOver tests.**
  `scripts/testu01/calibrate.jl` replicates tests 3 to 12 of BigCrush --
  `smarsa_CollisionOver` in its ten parameter sets -- on TestU01's own AES,
  which no Julia code touches, or on a package generator, and records the
  number of collisions and the p-value of every replication. The TestU01 layer
  gains TestU01's native generators (`TU01.aes`, `TU01.ulec_mrg32k3a`), and the
  test suite checks that TestU01's MRG32k3a and the package's give identical
  SmallCrush statistics, double for double: what the batteries see through the
  Julia callback is what they would see from inside the library.

- **A `substreams` suite.** `scripts/testu01/validate.jl --suite=substreams`
  interleaves the substreams of one stream, in the same shapes as
  `interleaved`, rather than separate streams: the other of the two
  constructions in L'Ecuyer et al. (2021), and the one cuRAND uses — for a
  counter-based generator one key and consecutive values of the counter's high
  half. `campaign.jl` runs it once per shape, and `replay.jl` replays a suspect
  on substreams of the same stream past those the run used. `--suite=all`
  still means the three suites it meant, so existing results directories
  resume unchanged. `scripts/run-campaign.sh` also takes `--generator`, and
  records it with the rest of the matrix.

- **A detached supervisor, now the fallback without linger.**
  `scripts/run-campaign.sh --supervisor=detached` starts the driver with
  `setsid nohup`, in a session of its own, records its pid in
  `<out>/driver.pid`, and stops it with `SIGINT` to its process group, jobs
  included. `--supervisor=auto` falls back to it rather than to tmux where
  linger is refused, and the watchdog works with it as with tmux.

### Fixed

- **A tmux supervisor died with the last logout.** A tmux built with systemd
  support moves each pane into a scope under the systemd user manager, which
  without linger is stopped, scopes and all, when the last session ends: on
  2026-10-16 that took a BigCrush sweep's driver and its 24 jobs, five hours
  in. `start` now looks where the pane landed and, under the user manager with
  linger off, stops it again before a job has started and points to
  `--supervisor=detached`. `start` also records the supervisor it starts, so
  `status`, `stop` and the watchdog follow a campaign moved from one to another.

- **Six jobs waited nine and a half hours on a Pkg lock.** Every job began with
  `Pkg.activate`, which takes a pid lock on `~/.julia/logs/manifest_usage.toml`
  and waits on a file watcher while another process holds it; on an NFS home,
  six of 24 jobs started together were never woken. A job whose Manifest already
  points at the checkout now sets its active project directly and calls no Pkg
  at all, and so does every other script that goes through `scripts/env.jl`.

- **A job killed by a signal was reported as `FAILED (exit 0)`**, since such a
  process has no exit code of its own. `campaign.jl` now names the signal:
  `FAILED (killed by SIGTERM)`.

- **A replay of an interleaved suspect ran in the default shape.** `replay.jl`
  read the battery, suite, generator and bit count from a run's log but not its
  streams and values, so a suspect from any shape but 64 × 1 would have been
  replayed in 64 × 1. It now reads them from the log's header, and skips at
  least as many streams as the run used, so that a shape over 1024 streams still
  replays on disjoint output. No campaign so far ran another shape.

- **A second replay took the first one's log for the run.** A replay writes its
  log beside the run's, ending with the same summary line, and was the newest
  complete log there; `replay.jl` now passes over replay logs when it picks the
  run's.

- **A campaign lost its Kerberos ticket at logout.** Where every login gets a
  ticket cache of its own and loses it at logout, the launcher's `krenew`
  inherited a cache that was gone by the time the driver started, and cron,
  which has no `KRB5CCNAME`, found none at all: the first BigCrush sweep sat
  idle for six hours with the watchdog failing every fifteen minutes.
  `scripts/run-campaign.sh` now gives the campaign a cache of its own, copied
  from yours, renewed by a `krenew` that no login owns, named in the launcher,
  revived by the watchdog if it dies, and destroyed by `stop`.

- **GNU time went "not found" now and then, and `--jobs` with it.** Under
  `pipefail`, `time -v true | grep -q ...` fails whenever `grep -q` exits before
  `time` has finished writing: one probe in eight, measured, on a binary sitting
  in `~/.local/bin`. The calibration then had no peak RSS and fell back to
  cores/4. The same race could report a renewable ticket as not renewable, and
  keep `stop` from removing the watchdog; all four such pipelines now read
  their input to the end.

- **Kerberos went undetected on a results directory that did not exist yet**,
  which is the usual case on a first `prepare`: the check now asks the nearest
  directory that does.

### Validation

- **`PhiloxRNG` reproduces cuRAND's `Philox4_32_10`**, word for word: cuRAND's
  seed is the key, low word first, and its subsequence is the package's
  substream. Checked against cuRAND 10.4 (CUDA 13.4) through the host API, on
  4 194 304 words for four seeds, and through the device API's
  `curand_init(seed, subsequence, offset)`, on 1 310 720 words with
  subsequences up to `2^64 - 1` and offsets mid-block; the test suite keeps
  words cuRAND generated. The FAQ says what carries over and what does not:
  the words, not cuRAND's floats, which include 1.0, nor its Box-Muller
  normals.

- **`MRG32k3a` reproduces cuRAND's `MRG32k3a`**, on the combination `z` both
  then scale: seed 0 is the published state `MRG32k3a()` starts from, other
  seeds go through `curand_init`'s map, and cuRAND's subsequence `k` is the
  package's substream `k`, `k·2^76` steps in. Checked against cuRAND 10.4 on
  4 × 2 621 440 values of the host API and 524 288 of the device API, with
  subsequences up to `2^51 − 1`, the end of the range cuRAND declares valid,
  and offsets up to 10⁶; the test suite keeps values cuRAND generated.

- **BigCrush over the whole matrix**, at v0.3.0: the seventeen generators,
  BigCrush on the `single` and `interleaved` suites, Alphabit and Rabbit on
  `bits`, 68 runs and 6409 statistics. No failure; eleven suspects against ≈12.8
  expected, and none of their 33 replications on disjoint streams left
  [0.001, 0.999]. See `docs/src/validation.md`.

- **The interleaved suite in three more shapes**, 16 × 8, 256 × 2 and 1024 × 1,
  fixed before the sweep: 51 BigCrush runs at `a869931`, 8160 statistics, no
  failure. Eighteen suspects against ≈16.3 expected, each replayed three times
  in its own shape; the three series that did not scatter cleanly were
  replicated twenty more times, all sixty values inside [0.001, 0.999]. No
  suspect persists, and with 64 × 1 the interleaved suite has passed BigCrush in
  four shapes. See `docs/src/validation.md`.

- **The substreams suite in the same four shapes**, the construction cuRAND
  uses, fixed before the sweep: 20 BigCrush runs at `638e13c`, covering the four
  counter-based generators and `MRG32k3a` on substreams of their first stream,
  which is cuRAND's seed 0 for Philox4x32 and MRG32k3a. 3200 statistics, no
  failure. Eleven suspects against ≈6.4 expected; all 33 replications, and
  all 99 statistics the replays recompute, stayed inside [0.001, 0.999]. No
  suspect persists: both constructions of L'Ecuyer et al. (2021) have passed
  BigCrush for these five generators. See `docs/src/validation.md`.

- **BigCrush's CollisionOver tests on small alphabets are miscalibrated**, for
  every generator. TestU01 compares the number of collisions among overlapping
  t-tuples with a Poisson law, but such collisions come in clumps. At d = 4
  (tests 11 and 12, t = 21) the variance is about 5/3 of Poisson's, and at d = 8
  (tests 9 and 10, t = 14) about 9/7. A calibration fixed in advance confirmed
  it at `405fe65`, on TestU01's own AES, with no Julia code involved:
  R = 1.76, 95% interval [1.50, 2.10], from 280 replications at d = 4.
  Each of tests 11 and 12 therefore singles out a perfect generator about 1.7%
  of the time instead of 0.2%. Eight of the 37 BigCrush suspects on the
  validation page came from these four tests; none persisted, and no verdict
  changes. See
  "CollisionOver on small alphabets" in `docs/src/validation.md`.

## [0.3.0]

Every float draw moves to the open interval `(0, 1)`, and the counter-based
generators gain GPU fills through a CUDA extension.

### Breaking

- **`close_open01` (`[0, 1)`) is removed; `open01` (`(0, 1)`) replaces it.**
  The CPU no longer uses the old construction, so code calling
  `close_open01` directly needs the new one instead.

### Changed

- **Julia 1.9 is the minimum**, for package extensions, which carry the CUDA
  support without making CUDA.jl a dependency. CI tests 1.9 and the current
  release.

- **Every floating-point draw lies in the open interval `(0, 1)`**, for all
  seventeen generators, in `Float64`, `Float32` and `Float16`, and through
  `rand(rng)`, `rand(rng, T)`, arrays and `rand!` alike, so that inversion is
  finite for every draw. The word-based families (xoshiro, PCG, counter-based)
  now return `(2k + 1) · 2^-p`, with `k` the top `p − 1` bits of the word and
  `p` the precision of the type: exact values, symmetric under `u ↦ 1 − u`, at
  the cost of one bit of resolution. **The values drawn change** for these
  fifteen generators in every float type, and for the MRG families in
  `Float32` and `Float16`. The scalar `Float64` draws of the MRG families are
  unchanged, but for the MRG63k3a cap below, and array fills now match them.
  TestU01 results measured on `Float64` for the word-based families predate
  the change; those of the MRG families do not.
  Several paths had reached an end of the interval:
  - `rand(rng)` on xoshiro divided by `typemax(UInt64)`, returning exactly 1
    for the top `2^10` outputs;
  - `rand(rng, Float32)` was `Float32(rand(rng))`, returning `1.0f0` once in
    `2^25` draws (`Float16(1)` once in `2^12`);
  - PCG, the counter-based families and the standard library's `Float32` path
    filled the mantissa of 1.0 and subtracted 1, which can return 0;
  - arrays from the MRG families went through `(1 + u) − 1`, which for
    MRG63k3a returned 0 below `2^-53` and 1 above `1 − 2^-53`;
  - MRG63k3a's own `k / (m1 + 1)` rounds to exactly 1.0 for the twelve largest
    `k`, as in L'Ecuyer's C code; those draws are now capped at
    `prevfloat(1.0)`.
  The narrow floats were also defined on `::Type` only, so `rand(rng, Float32,
  n)` bypassed them, read the low bits of the word, and disagreed with the
  scalar draws. All paths are now sampler methods, and the test suite checks
  scalar, array and `rand!` against each other and against the documented
  construction for every generator.

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

- **The threading examples take one stream per replication, not per thread.**
  The README, the FAQ, *Streams & Substreams* and the `next_stream!(gen, n)`
  docstring allocated `nthreads()` streams, which ties what a replication draws
  to the number of threads. Indexed by replication, the results are the same on
  one thread or on sixty-four — the requirement Passerat-Palmbach, Mazel & Hill
  (2012) set for parallel streams.

### Added

- **GPU fills for the counter-based generators, bit for bit the CPU's.** With
  CUDA.jl loaded — a weak dependency, through a package extension —
  `rand!(rng, A)` on a `CuArray{Float64}`, `CuArray{Float32}` or
  `CuArray{Float16}` fills `A` on the device for Philox4x32, Philox4x64,
  Threefry4x32 and Threefry4x64. The values are exactly those `rand!` gives on
  the CPU from the same position, mid-block included, and the generator object,
  which stays on the host, is left where the CPU fill would leave it. Element
  `k` is a pure function of the key, the counter and `k`, checked against the
  stream object on the CPU and against the CPU fill on the device; it does not
  depend on the launch configuration. `randn!` on a `CuArray` (Box-Muller,
  Philox4x32), with `randn_inversion!` and `randn_polar!` for comparison,
  continues the same stream but is not the CPU's `randn`. `philox4x32_10`,
  `philox4x32_counter` and `open01` are exported for writing kernels.

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

**Never registered.** `Project.toml` said 0.2.0 from `e6376ae` (2026-08-31)
until 0.3.0, but no 0.2.0 reached the General registry. Its changes first
reached users with 0.3.0, so anyone upgrading from 0.1.0 meets the breaking
change below there. The comparison link runs from `6449bd8`, the commit 0.1.0
was registered from, to `e6376ae`.

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

## [0.1.0]

Initial release: `MRG32k3a` and the xoshiro/xoroshiro families, with the stream
and substream object model of L'Ecuyer et al. (2002). Registered on 2026-08-29
from `6449bd8`, without a git tag; `v0.1.0` was added on 2026-10-18.

[Unreleased]: https://github.com/JLChartrand/RandomDataStreams.jl/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/JLChartrand/RandomDataStreams.jl/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/JLChartrand/RandomDataStreams.jl/compare/e6376ae...v0.3.0
[0.2.0]: https://github.com/JLChartrand/RandomDataStreams.jl/compare/6449bd8...e6376ae
[0.1.0]: https://github.com/JLChartrand/RandomDataStreams.jl/tree/v0.1.0
