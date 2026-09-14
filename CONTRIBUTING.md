# Contributing to RandomDataStreams.jl

Thank you for considering a contribution. This document says how to report a
problem, how to get help, and what a change has to satisfy before it can be
merged.

## Getting help

- **Questions about using the package**: open an
  [issue](https://github.com/JLChartrand/RandomDataStreams.jl/issues) labelled
  `question` — there is no separate forum, and a question that was worth asking
  is usually worth answering in public. Check the
  [FAQ](https://jlchartrand.github.io/RandomDataStreams.jl/dev/faq/) first —
  it covers the questions we are asked most often, including why PCG's own
  increment-based "streams" are deliberately not exposed.
- **Documentation that is unclear or wrong** is a bug. Report it as one.

## Reporting a bug

Open an [issue](https://github.com/JLChartrand/RandomDataStreams.jl/issues) and
include:

1. the output of `versioninfo()` and the package version;
2. the generator involved, and whether you were using the generator object
   (`MRG32k3aGen`) or the stream object (`MRG32k3a`);
3. a self-contained snippet that reproduces the behaviour, with the seed;
4. what you expected and what happened.

For a statistical complaint — "these streams look correlated" — say which
battery, which suite, and which p-values. A single suspicious p-value is
expected: TestU01 flags anything outside `[1e-3, 1-1e-3]`, and a battery runs
enough tests that a few will land there by chance. Re-run with a different seed
before reporting; a genuine defect persists.

## Development setup

```julia
using Pkg
Pkg.develop(path = "path/to/RandomDataStreams.jl")
Pkg.test("RandomDataStreams")
```

The test suite takes about six minutes, most of it the batteries. It includes a
SmallCrush run per generator,
which is an installation check rather than a validation: it confirms the build
works, not that a generator is sound. The batteries do not run on Apple
Silicon, where Julia cannot build the C callback TestU01 needs from a closure;
the suite skips them and says so.

Heavier validation is on demand and outside `Pkg.test()`:

```bash
julia scripts/testu01/validate.jl --list
julia scripts/testu01/validate.jl --battery=crush --generator=Xoshiro256p --suite=interleaved
julia scripts/testu01/validate.jl --battery=alphabit --suite=bits   # the bit-level battery
julia scripts/testu01/campaign.jl --battery=crush --jobs=6          # the full matrix
julia -O3 scripts/benchmarks/throughput.jl                          # timings
```

The batteries are `smallcrush`, `crush`, `bigcrush`, `alphabit` and `rabbit`.
A campaign runs the last two on its `bits` suite whatever `--battery` says,
since Crush and BigCrush were never built for an integer stream.

Crush takes hours per generator and BigCrush the better part of a day, so use
`campaign.jl` — it runs the matrix as independent processes, records what
finished in `summary.tsv`, and resumes rather than restarting.

On a machine dedicated to a campaign, `scripts/run-campaign.sh` wraps the whole
sequence: it benchmarks before the machine gets busy, calibrates one job, sizes
`--jobs` from the measured peak RSS, and installs a systemd user unit that
survives a logout and a reboot.

```bash
./scripts/run-campaign.sh run     --battery=crush   # prepare, then start
./scripts/run-campaign.sh status  --battery=crush
./scripts/run-campaign.sh collect --battery=crush   # bundle results for a write-up
```

## What a change has to satisfy

**Every generator obeys one contract.** The point of this package is that code
written against the abstract interface never names a family. A change that
makes one generator behave differently from the others under `next_stream!`,
`next_substream!`, `reset_stream!`, `reset_substream!`, `advance_state!`,
`get_state`, `set_state!` or `Random.seed!` will be asked to change. The
contract is stated in the docstring of `AbstractStreamableRNG` and in the
[API reference](https://jlchartrand.github.io/RandomDataStreams.jl/dev/api/).

**Substream boundaries are anchored.** `next_substream!` lands in the same
place regardless of how much of the current substream was consumed. This is
what makes a replication reproducible when the model changes how much
randomness it draws, and it is not negotiable.

**New generators need external reference values.** An implementation is
accepted when it reproduces the output of an independent reference — the
Salmon et al. test vectors for counter-based families, the C code from
`xoshiro.di.unimi.it`, NumPy for PCG — as a test in `test/runtests.jl`, not
as a claim in a comment.

**New stream mechanisms need evidence, not argument.** Streams that are only
plausibly independent do not go in. If a family's streams come from jumps,
they must pass the interleaved battery in `test/test_streams_interleaved.jl`,
which runs many streams against each other rather than one at a time. That
battery is what caught the PCG defect where power-of-two jump distances
produced correlated streams: each stream passed SmallCrush alone, while
sixty-four interleaved failed with fourteen of fifteen p-values at zero.

## Style

Match the surrounding code. Beyond that:

- Comments explain *why*, not *what*. A constant that looks arbitrary should
  say where it comes from — a paper, a section, an equation.
- Hot paths must not allocate. Check with `@allocated` before submitting.
- The package supports Julia ≥ 1.6, which CI enforces. Avoid syntax added
  later; a backslash-newline continuation inside a string literal, for
  instance, is a syntax error there and not on current Julia.
- Public functions get docstrings. `docs/make.jl` runs with
  `checkdocs = :exports`, so an exported symbol without one fails the build.

## Submitting

1. Branch from `master`.
2. Add tests. A bug fix gets a test that fails without the fix.
3. Run `Pkg.test("RandomDataStreams")` locally.
4. Update `CHANGELOG.md`.
5. Open a pull request describing what changes for a user, not only what
   changed in the code. CI runs the suite on Linux, macOS and Windows, on
   Julia 1.6 and current, and builds the documentation.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
