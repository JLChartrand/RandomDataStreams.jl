# Docstrings

Auto-generated documentation from the package docstrings.

## Abstract types

```@docs
AbstractRNGStream
AbstractStreamableRNG
```

## MRG32k3a

```@docs
MRG32k3a
MRG32k3aGen
checkseed
```

## MRG63k3a

```@docs
MRG63k3a
MRG63k3aGen
checkseed63
```

## Xoshiro / xoroshiro families

```@docs
RandomDataStreams.LinRNG
Xoroshiro128p
Xoroshiro128ss
Xoroshiro128pp
Xoshiro256p
Xoshiro256ss
Xoshiro256pp
Xoshiro512p
Xoshiro512ss
Xoshiro512pp
```

### Stream generators

```@docs
RandomDataStreams.LinGen
Xoroshiro128pGen
Xoroshiro128ssGen
Xoroshiro128ppGen
Xoshiro256plusGen
Xoshiro256pGen
Xoshiro256ssGen
Xoshiro256ppGen
Xoshiro512pGen
Xoshiro512ssGen
Xoshiro512ppGen
```

## Counter-based generators

```@docs
RandomDataStreams.CBGen
```

### Functions

```@docs
rand(::MRG32k3a)
rand(::MRG63k3a)
rand(::RandomDataStreams.LinRNG)
rand(::RandomDataStreams.LinRNG, ::Type{Float32})
rand(::RandomDataStreams.LinRNG, ::Type{Float16})
rand(::RandomDataStreams.CBRNG)
rand(::RandomDataStreams.CBRNG, ::Type{Float32})
rand(::RandomDataStreams.CBRNG, ::Type{Float16})
rand(::PCGRNG)
rand(::PCGRNG, ::Type{Float32})
rand(::PCGRNG, ::Type{Float16})
rand(::MRG32k3a, ::RandomDataStreams.Random.SamplerTrivial{RandomDataStreams.Random.CloseOpen12_64})
rand(::MRG63k3a, ::RandomDataStreams.Random.SamplerTrivial{RandomDataStreams.Random.CloseOpen12_64})
short_jump!
long_jump!
advance_state!
srand!
get_state
set_state!
next_stream!
reset_stream!
reset_substream!
next_substream!
```
