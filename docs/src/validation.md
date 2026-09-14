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
to 8 values per stream) are in `scripts/testu01_bigcrush.jl`.

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
spread but leaves the expected count unchanged.)

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
clear of the 64 the interleaved suite takes — so the replay is the independent
replication the guide asks for. This is the package's own stream facility doing
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
Rabbit were reachable — gives the rule something to bite on. **No p-value came within four orders of
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
`(battery, suite, generator)`. Re-running the same command skips those and does
the rest, so an interruption costs only the jobs actually in flight — never the
weeks already spent. Each job also writes its own `summary.tsv` in its own
directory, which the driver reconciles, so even a driver killed mid-flight
loses nothing that finished.

**Stopping cleanly.** `touch <out>/STOP` stops new jobs from starting and waits
for the running ones, which is how to end a campaign without discarding a
battery that is twenty hours in. Ctrl-C does the same.

**Surviving logout and reboot.** `nohup` survives a logout, `tmux` survives a
logout and lets you reattach, and neither survives a reboot or restarts a driver
that died. For a campaign measured in weeks, use the systemd *user* unit in
`scripts/testu01/randomdatastreams-campaign.service`:

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
`scripts/run-campaign.sh` can supervise the campaign with **tmux** instead:

```bash
./scripts/run-campaign.sh run --battery=crush --supervisor=tmux --watchdog
./scripts/run-campaign.sh status --battery=crush      # no flag needed afterwards
./scripts/run-campaign.sh stop   --battery=crush
```

`--supervisor=auto`, the default, uses systemd when linger is on or can be
turned on, and falls back to tmux. The choice is recorded in the output
directory, so `status`, `stop` and `collect` find the campaign without being
told again.

What tmux does not give you is what `--watchdog` restores, through a *user*
crontab entry that needs no privileges:

| | systemd unit | tmux | tmux + `--watchdog` |
|---|---|---|---|
| survives logout | with linger | yes, unless `KillUserProcesses=yes` | same |
| survives a reboot | yes | no | yes (`@reboot`) |
| restarts a driver that died | `Restart=on-failure` | no | yes, within 15 min |
| attach and watch it work | `journalctl -f` | `tmux attach` | `tmux attach` |

Two details the script handles, both of which are silent failures otherwise.
`KillUserProcesses=yes` in `logind.conf` kills a detached tmux session at logout
just as it kills an unlingered service, so tmux is *not* a workaround on such a
machine — `run-campaign.sh` reads the setting and says so before you commit two
weeks to it. And tmux keeps its socket under `/run/user/$UID`, a directory that
is removed when your last session ends; the script points `TMUX_TMPDIR` at a
directory under `$HOME` instead, so the session started from your login shell
and the one the watchdog looks for are the same session.

The watchdog respects `touch <out>/STOP`: a campaign you stopped on purpose
stays stopped. `stop` removes the crontab entry, and any crontab it edits is
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
wraps the driver in `krenew`, which renews the ticket for as long as the
campaign runs:

```bash
exec nice -n 10 ionice -c 3 /usr/bin/krenew -K 30 -- julia ... campaign.jl
```

Two limits are worth knowing before committing days to a sweep. First,
**renewal has its own deadline**: a renewable ticket may be renewed only until
the `renew until` that `klist -f` prints, typically some weeks after the
`kinit` that created it, and a ticket that is not renewable at all cannot be
rescued — `prepare` warns in both cases. A fresh login before a long campaign
restarts that clock. Second, **the watchdog needs a ticket of its own**: the
launcher's `krenew` runs only while the campaign does, and the watchdog exists
for the times it does not. Keep one renewed independently of it:

```bash
krenew -K 30 -b        # a background daemon, renewing every 30 minutes
```

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

### What still needs BigCrush

Two things the suite cannot settle at SmallCrush level:

- **The low bits of the `+` scramblers.** Blackman & Vigna document the lowest
  bits of `xoshiro256+` and `xoroshiro128+` as linearly weak. SmallCrush does
  not resolve it, so the `bits` suite passing at that level is not a
  clearance; the question needs Crush or BigCrush, and possibly does not show
  even there. Alphabit and Rabbit on the bit stream are the cheaper instrument
  aimed at exactly this, and are worth running first.
- **Dependence between streams at full strength.** The interleaved suite is
  the construction L'Ecuyer et al. (2021) call for, and the paper says
  batteries for counter-based generators still need development. A SmallCrush
  pass is a smoke test; the claim worth publishing is a BigCrush one.

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
