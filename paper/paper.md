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
each reproducible on its own and provably disjoint from the others. That is what
makes common random numbers possible, the variance reduction technique on which
comparisons between system designs usually rest, and what makes a replication
reproducible when the model under study changes how much randomness it consumes.
`RandomDataStreams.jl` provides the stream and substream object model of
L'Ecuyer et al. [@lecuyer:2002] for Julia, applied uniformly to four families of
generator: the combined multiple recursive generator MRG32k3a, the F2-linear
xoshiro and xoroshiro families [@blackman:2021], PCG [@oneill:2014], and the
counter-based generators Philox and Threefry [@salmon:2011].

Every generator answers to the same calls — `next_stream!`, `next_substream!`,
`reset_substream!`, `reset_stream!`, `advance_state!`, `get_state`,
`set_state!` — with the same meanings, so code written against the abstract
interface never names a family. Substream boundaries are anchored: moving to the
next substream lands in the same place no matter how much of the current one was
consumed, which is the property that makes a replication reproducible across
model variants.

# Statement of need

<!-- TODO: expand. Points to make:
     - CRN requires assigning disjoint, individually addressable streams
     - Julia's Random has no such abstraction; seeding is not enough
     - the ecosystem gap (see State of the field)
     - counter-based generators additionally allow a draw to be addressed by
       index, which is what lets a GPU kernel or a remote worker reproduce a
       stream it was assigned                                                  -->

# State of the field

Julia's package ecosystem already offers a wide selection of random number
generators. `RandomNumbers.jl` [@sunoru:2019] collects PCG, several members of
the xorshift family and MT19937 behind a common `AbstractRNG` interface, and
`Random123.jl` provides the counter-based generators of Salmon et al.
[@salmon:2011]. Both are mature, registered and widely used, and neither
addresses the problem this package exists for.

Neither package implements streams or substreams. As of `RandomNumbers.jl`
v1.6.0 and `Random123.jl` v1.7.1, the words *substream* and *jump* do not occur
in either source tree, and no operation partitions a generator's period into
non-overlapping, individually rewindable segments. Jump-ahead exists only for
PCG, through `advance!`; the xoshiro jump polynomials of Blackman and Vigna
[@blackman:2021] are absent, as is any skip-ahead for MT19937 or for the
counter-based families, which expose only `set_counter!`. MRG32k3a
[@lecuyer:2002], the reference generator of the simulation literature and the
generator whose object model defines the stream abstraction, is absent from both.

The one stream mechanism that is offered is the one L'Ecuyer et al.
[@lecuyer:2021] caution against. `RandomNumbers.jl` exposes PCG's
increment-based "stream variations", including `PCGStateUnique`, whose increment
is derived from the memory address of the generator object; two processes seeded
identically therefore produce different sequences. More generally, distinct
increments do not yield provably independent sequences: for a linear
congruential generator modulo $2^k$ with multiplier $a \equiv 1 \pmod 4$, the
sequences with increments $c_1$ and $c_2$ satisfy $t_n = s_n + h$ for a constant
$h$ whenever $4 \mid (c_1 - c_2)$, which holds for half of all pairs of odd
increments.

`RandomDataStreams.jl` supplies what is missing rather than another collection
of generators: the stream object model, the jump machinery each family requires,
and a test suite that exercises streams against each other rather than one at a
time. That inter-stream testing is not incidental. It is what revealed that
power-of-two jump distances produce correlated PCG streams — each stream passes
SmallCrush alone, while sixty-four of them interleaved fail it with fourteen of
fifteen $p$-values at zero — a defect that cannot arise in a package with no
jump-based streams to test.

# Validation

<!-- TODO: SmallCrush in the test suite, Crush/BigCrush on demand, the
     interleaved battery, the bit-level battery, and the known-answer vectors
     each family is pinned against (Salmon et al., xoshiro.di.unimi.it, NumPy).
     Numbers to be quoted with the commit they were measured at.               -->

# Acknowledgements

<!-- TODO -->

# References
