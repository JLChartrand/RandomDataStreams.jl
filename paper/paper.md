---
title: 'RandomDataStreams.jl: streams and substreams for Julia random number generators'
tags:
  - Julia
  - random number generation
  - simulation
  - common random numbers
  - variance reduction
authors:
  - name: Fabian Bastin
    orcid: 0000-0000-0000-0000        # TODO
    affiliation: 1
  # TODO: confirm co-authors, in particular the original author of the package
affiliations:
  - name: Département d'informatique et de recherche opérationnelle, Université de Montréal, Canada
    index: 1
date: 30 August 2026
bibliography: paper.bib
---

# Summary

Stochastic simulation rarely needs one stream of random numbers; it needs many,
each reproducible on its own and provably disjoint from the others.
`RandomDataStreams.jl` provides the stream and substream object model of
L'Ecuyer et al. [@lecuyer:2002] for Julia, applied uniformly to four families of
generator: the combined multiple recursive generator MRG32k3a, the F2-linear
xoshiro and xoroshiro families [@blackman:2021], PCG [@oneill:2014], and the
counter-based generators Philox and Threefry [@salmon:2011].

Every generator answers to the same calls — `next_stream!`, `next_substream!`,
`reset_substream!`, `reset_stream!`, `advance_state!`, `get_state`,
`set_state!` — with the same meanings, so code written against the abstract
interface never names a family.

# Statement of need

Common random numbers — driving two competing system designs with the same
randomness, so that the difference between them is not swamped by the difference
between their random inputs — requires that each source of randomness in a model
have its own sequence, disjoint from the others and rewindable independently of
them. None of that follows from seeding.

Seeding two generators differently makes them unlikely to collide, which is a
birthday-problem estimate rather than a guarantee, and it gives no way to rewind
one source of randomness while leaving the others in place — precisely what is
needed when a model variant consumes a different amount of randomness than the
variant it is compared against. The model of L'Ecuyer et al. [@lecuyer:2002]
answers both: streams are separated by jumps too large to overlap in any
feasible run, and substream boundaries are anchored, so returning to the start
of a substream lands in the same place however much of it was consumed.

Julia's standard library offers no such abstraction, and neither does its
package ecosystem (see *State of the field*). A simulation practitioner moving
to Julia from SSJ, Arena or Simio therefore loses the stream model that the
simulation literature has assumed since 2002 — and gains no replacement.

Counter-based generators add a property no recurrence-based generator has: a
draw is a keyed bijection of its index, so replicate $i$ of stream $s$ can be
computed without replaying anything before it. That is what allows a GPU kernel
or a remote worker to reproduce a stream it was assigned, holding no state and
communicating with nothing. `RandomDataStreams.jl` places both kinds of
generator behind the same interface, so a model written against it can move
between them without changing a line.

# State of the field

Julia's package ecosystem already offers a wide selection of random number
generators, most of them collected under the `JuliaRandom` organisation:
`RandomNumbers.jl` [@sunoru:2019] gathers PCG, several members of the xorshift
family and MT19937 behind a common `AbstractRNG` interface; `Random123.jl`
provides the counter-based generators of Salmon et al. [@salmon:2011];
`StableRNGs.jl` supplies a Lehmer generator whose stream is stable across Julia
releases; `VSL.jl` binds Intel's Vector Statistics Library. These packages are
mature, registered and widely used, and none of them addresses the problem this
package exists for.

No package in the organisation implements streams or substreams in the sense of
L'Ecuyer et al. [@lecuyer:2002]. As of `RandomNumbers.jl` v1.6.0 and
`Random123.jl` v1.7.1, the words *substream* and *jump* do not occur in either
source tree, and no operation partitions a generator's period into
non-overlapping, individually rewindable segments. Jump-ahead exists only for PCG, through `advance!`: the
xoshiro jump polynomials of Blackman and Vigna [@blackman:2021] are absent, as
is any skip-ahead for MT19937 or for the counter-based families, which expose
only `set_counter!`. The gap is not one of underlying capability: `VSL.jl` wraps a library providing `vslSkipAheadStream`
and `vslLeapfrogStream`, and binds neither. MRG32k3a [@lecuyer:2002], the
reference generator of the simulation literature and the generator whose object
model defines the stream abstraction, is absent throughout. (The "stable
streams" of `StableRNGs.jl` are unrelated: the term there means reproducibility
of one sequence across Julia versions.)

The one stream mechanism that is offered is the one L'Ecuyer et al.
[@lecuyer:2021] caution against. `RandomNumbers.jl` exposes PCG's
increment-based "stream variations", including `PCGStateUnique`, whose increment
is derived from the memory address of the generator object; two processes seeded
identically therefore produce different sequences. More generally, distinct
increments do not yield provably independent sequences: for a linear
congruential generator modulo $2^k$ with multiplier $a \equiv 1 \pmod 4$, the
sequences with increments $c_1$ and $c_2$ are translates of one another,
$t_n = s_n + h$, whenever $4 \mid (c_1 - c_2)$ — half of all pairs of odd
increments.

`RandomDataStreams.jl` supplies what is missing rather than another collection
of generators: the stream object model, the jump machinery each family requires,
and a test suite that exercises streams against each other rather than one at a
time. It builds on the ecosystem rather than replacing it — its statistical
validation runs through the organisation's own `RNGTest.jl` interface to
TestU01. That inter-stream testing is not incidental. It is what revealed that
power-of-two jump distances produce correlated PCG streams — each stream passes
SmallCrush alone, while sixty-four of them interleaved fail it with fourteen of
fifteen $p$-values at zero — a defect that cannot arise in a package with no
jump-based streams to test.

# Validation

Every generator is pinned to an independent reference: the test vectors of
Salmon et al. [@salmon:2011] for Philox and Threefry, the C implementations
published at `xoshiro.di.unimi.it` for the xoshiro and xoroshiro families, and
NumPy — whose `default_rng()` is PCG64 — for both PCG variants, matching outputs
and final state. These are tests in the suite, not claims in a comment.

Statistical testing runs through `RNGTest.jl`, the Julia interface to TestU01
[@lecuyer:2007], along three axes rather than one. The conventional axis tests a
single stream. The second interleaves many streams in round-robin, which tests
the streams against each other and is the construction WSC 2021 [@lecuyer:2021]
calls for; it is what revealed the PCG jump-distance defect described above. The
third tests the integer paths at bit level, which the $U(0,1)$ batteries never
examine, since Crush inspects only the high bits of a floating-point output.

The test suite runs SmallCrush on all three axes as an installation check.
Crush and BigCrush are validation exercises rather than installation checks, and
run on demand through a campaign driver that executes the matrix as independent
processes and resumes where it left off.

<!-- TODO: quote the campaign results here, each with the commit it was
     measured at, once the BigCrush sweep completes. -->

# Acknowledgements

<!-- TODO -->

# References
