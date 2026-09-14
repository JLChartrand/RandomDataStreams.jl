using RandomDataStreams
using Test
using Random

statewords(::Type{RandomDataStreams.LinRNG{N,S}}) where {N,S} = N

@testset "RandomDataStreams.jl" begin

    @testset "the all-zero xoshiro state is refused everywhere" begin
        # It is a fixed point of every xoshiro/xoroshiro transition, so a
        # generator holding it returns zeros forever -- and used to, silently.
        # The FAQ has always said constructors reject it; now they do, and so
        # does every other way the state can be set.
        z4 = fill(UInt64(0), 4)
        @test !RandomDataStreams.checkseed(z4)
        @test RandomDataStreams.checkseed(UInt64[0, 0, 0, 1])
        @test_throws ArgumentError Xoshiro256pp(z4)
        @test_throws ArgumentError RandomDataStreams.Xoshiro256ppGen(z4)
        @test_throws ArgumentError srand!(Xoshiro256pp(), z4)
        @test_throws ArgumentError set_state!(Xoshiro256pp(), z4)
        @test_throws ArgumentError Random.seed!(Xoshiro256pp(), z4)
        @test_throws ArgumentError RandomDataStreams.Xoroshiro128p(fill(UInt64(0), 2))
        @test_throws ArgumentError RandomDataStreams.Xoshiro512p(fill(UInt64(0), 8))
        # and no integer seed can produce one
        @test all(RandomDataStreams.checkseed(RandomDataStreams._seed_words_nonzero(UInt64(k), 4))
                  for k in 0:200)
    end


    @testset "checkseed" begin
        @test checkseed(RandomDataStreams.DEFAULT_SEED)
        @test !checkseed([1, 2, 3, 4, 5])                       # wrong length
        @test !checkseed([0, 0, 0, 1, 1, 1])                    # all-zero first component
        @test !checkseed([1, 1, 1, 0, 0, 0])                    # all-zero second component
        @test !checkseed([-1, 1, 1, 1, 1, 1])                   # negative entry
        @test !checkseed([RandomDataStreams.PMF.m1, 1, 1, 1, 1, 1])          # >= m1 in first half
        @test !checkseed([1, 1, 1, RandomDataStreams.PMF.m2, 1, 1])          # >= m2 in second half
        @test checkseed([RandomDataStreams.PMF.m1 - 1, RandomDataStreams.PMF.m2 - 1, 7, RandomDataStreams.PMF.m2 - 1, 3, 9])
    end

    @testset "MRG32k3a constructors" begin
        rng = MRG32k3a()
        @test rng.Cg == RandomDataStreams.DEFAULT_SEED && rng.Bg == RandomDataStreams.DEFAULT_SEED && rng.Ig == RandomDataStreams.DEFAULT_SEED

        seed = [42, 1, 2, 3, 4, 5]
        rng = MRG32k3a(seed)
        seed[1] = -100                    # constructor must copy the seed
        @test rng.Cg[1] == 42

        x = [1, 2, 3, 4, 5, 6]
        y = [10, 20, 30, 40, 50, 60]
        z = [100, 200, 300, 400, 500, 600]
        rng = MRG32k3a(x, y, z)
        @test rng.Cg == x && rng.Bg == y && rng.Ig == z
        x[1] = 999                        # no aliasing
        @test rng.Cg[1] == 1

        @test_throws AssertionError MRG32k3a([0, 0, 0, 1, 1, 1])
    end

    @testset "MRG32k3a reference values" begin
        rng = next_stream!(MRG32k3aGen())
        @test [rand(rng) for _ in 1:5] ==
              [0.12701112204657714, 0.3185275653967945,
               0.3091860155832701, 0.8258468629271136, 0.2216299157820229]

        for (T, ref) in (
            (UInt8,   UInt8[0xed, 0x82, 0x51]),
            (UInt16,  UInt16[0xcced, 0x0582, 0xd051]),
            (UInt32,  UInt32[0xcced0582, 0xd051b288, 0xbcca9934]),
            (UInt64,  UInt64[0xcced0582d051b288, 0xbcca99340444f89c, 0x22d387a29770f58a]),
            (UInt128, UInt128[0xcced0582d051b288bcca99340444f89c,
                              0x22d387a29770f58a57416a5915ef65d6,
                              0x9ef65959a1651d5418ebdf4023a8b6d5]),
            (Int8,    Int8[-19, -126, 81]),
            (Int16,   Int16[-13075, 1410, -12207]),
            (Int32,   Int32[-856881790, -799952248, -1127573196]),
            (Int64,   Int64[-3680279261092924792, -4842890000594569060, 2509498549770712458]),
            (Int128,  Int128[-67889169649162077992496674455215081316,
                             46292077500965604301225650373683799510,
                             -128985226324011672590792899314054547755]),
            (Float32, Float32[0.8517306, 0.63826084, 0.5828004]),
            (Float16, Float16[0.2314, 0.377, 0.0791]),
            (Bool,    Bool[1, 0, 1]),
        )
            r = next_stream!(MRG32k3aGen())
            @test [rand(r, T) for _ in 1:3] == ref
        end
    end

    @testset "MRG32k3a output properties" begin
        rng = MRG32k3a()
        u = [rand(rng) for _ in 1:100_000]
        @test all(0.0 .<= u .< 1.0)
        @test abs(sum(u) / length(u) - 0.5) < 0.01

        r = next_stream!(MRG32k3aGen())
        v = [rand(r, 1:10) for _ in 1:10_000]
        @test all(1 .<= v .<= 10)
        @test count(==(1), v) > 500 && count(==(10), v) > 500
    end

    @testset "MRG32k3a streams & substreams" begin
        gen = MRG32k3aGen()
        rng1 = next_stream!(gen)
        rng2 = next_stream!(gen)
        first1 = rand(MRG32k3a(copy(rng1.Ig), copy(rng1.Ig), copy(rng1.Ig)))
        first2 = rand(MRG32k3a(copy(rng2.Ig), copy(rng2.Ig), copy(rng2.Ig)))

        # successive streams are disjoint
        a = [rand(rng1) for _ in 1:100]
        b = [rand(rng2) for _ in 1:100]
        @test Set(a) ∩ Set(b) == Set{Float64}()

        # reset_stream! rewinds to the very beginning
        rand(rng1); rand(rng1)
        reset_stream!(rng1)
        @test rand(rng1) == first1

        # reset_substream!: no next_substream! yet -> back to the stream start
        reset_substream!(rng2)
        @test rand(rng2) == first2

        # next_substream! opens a new, different block; reset_substream! rewinds it
        v1 = rand(rng2)
        next_substream!(rng2)
        w1 = rand(rng2)
        reset_substream!(rng2)
        @test rand(rng2) == w1          # start of the new substream
        @test v1 != w1
    end

    @testset "MRG32k3a state handling" begin
        rng = next_stream!(MRG32k3aGen())
        state = get_state(rng)
        xs = [rand(rng) for _ in 1:5]

        clone = MRG32k3a(state, state, state)
        @test rand(clone) == xs[1]

        st = get_state(rng)
        st[1] += 1                      # get_state must return an independent copy
        @test get_state(rng) != st

        c = copy(rng)
        rand(c); rand(c)
        @test rand(rng) != rand(c)      # copies evolve independently
    end

    @testset "MRG32k3a advance_state!" begin
        ref = next_stream!(MRG32k3aGen())
        vals = [rand(ref) for _ in 1:4]

        rng = MRG32k3a([12345, 12345, 12345, 12345, 12345, 12345])
        advance_state!(rng, Int64(2), Int64(-1))     # skip n = 2^2 - 1 = 3 values
        @test rand(rng) == vals[4]

        # backward jump: after consuming vals[4] the position is 4; n = -2^2 = -4
        advance_state!(rng, Int64(-2), Int64(0))
        @test rand(rng) == vals[1]

        # e = 0, c = k: plain forward jump
        rng2 = MRG32k3a([12345, 12345, 12345, 12345, 12345, 12345])
        advance_state!(rng2, Int64(0), Int64(2))
        @test rand(rng2) == vals[3]
    end

    @testset "MRG32k3aGen" begin
        gen = MRG32k3aGen()
        @test gen.seed == RandomDataStreams.DEFAULT_SEED
        @test get_state(gen) == RandomDataStreams.DEFAULT_SEED

        custom = MRG32k3aGen([7, 7, 7, 8, 8, 8])
        @test get_state(custom) == [7, 7, 7, 8, 8, 8]
        @test_throws AssertionError MRG32k3aGen([0, 0, 0, 1, 1, 1])

        g2 = MRG32k3aGen([1, 2, 3, 4, 5, 6])
        r = next_stream!(g2)
        @test get_state(g2) != [1, 2, 3, 4, 5, 6]   # internal seed advanced
    end

    @testset "checkseed63" begin
        P = RandomDataStreams.PMF63
        @test checkseed63(RandomDataStreams.DEFAULT_SEED63)
        @test !checkseed63([1, 2, 3, 4, 5])                     # wrong length
        @test !checkseed63([0, 0, 0, 1, 1, 1])                  # all-zero first component
        @test !checkseed63([1, 1, 1, 0, 0, 0])                  # all-zero second component
        @test !checkseed63([-1, 1, 1, 1, 1, 1])                 # negative entry
        @test !checkseed63([P.m1, 1, 1, 1, 1, 1])               # >= m1 in first half
        @test !checkseed63([1, 1, 1, P.m2, 1, 1])               # >= m2 in second half
        @test checkseed63([P.m1 - 1, P.m2 - 1, 7, P.m2 - 1, 3, 9])
        # the moduli are the ones of Table II, fourth entry
        @test P.m1 == 2^63 - 6645 && P.m2 == 2^63 - 21129
        @test P.norm == 1.0842021724855052e-19                  # as in the reference C code
    end

    @testset "MRG63k3a modular reduction" begin
        # The step avoids a 128-bit remainder by folding at bit 63. What makes
        # that valid is a bound on the argument, so both the identity and the
        # bound are checked, over the exact range the step can produce and at
        # its endpoints.
        P = RandomDataStreams.PMF63
        hi1 = Int128(P.a12 + P.a13n) * P.m1        # largest component-1 argument
        hi2 = Int128(P.a21 + P.a23n) * P.m2        # largest component-2 argument
        @test hi1 < Int128(1) << 96
        @test hi2 < Int128(1) << 99

        for t in (Int128(0), Int128(1), Int128(P.m1) - 1, Int128(P.m1),
                  Int128(P.m1) + 1, Int128(1) << 63, hi1, hi1 - 1)
            @test P.mod_m1(t) == mod(t, Int128(P.m1))
        end
        for t in (Int128(0), Int128(1), Int128(P.m2) - 1, Int128(P.m2),
                  Int128(P.m2) + 1, Int128(1) << 63, hi2, hi2 - 1)
            @test P.mod_m2(t) == mod(t, Int128(P.m2))
        end

        rng = MRG32k3a(20260901)                   # an independent source
        bad = 0
        for _ in 1:20_000
            t1 = Int128(rand(rng, UInt64)) * rand(rng, UInt32) % (hi1 + 1)
            t2 = Int128(rand(rng, UInt64)) * rand(rng, UInt32) % (hi2 + 1)
            P.mod_m1(t1) == mod(t1, Int128(P.m1)) || (bad += 1)
            P.mod_m2(t2) == mod(t2, Int128(P.m2)) || (bad += 1)
        end
        @test bad == 0
    end

    @testset "MRG63k3a reference values" begin
        # L'Ecuyer's own C implementation of MRG63k3a (Operations Research 47,
        # 1 (1999), Table II, fourth entry), run with its own default seed:
        # every state variable at 123456789.
        rng = MRG63k3a(fill(123456789, 6))
        @test [rand(rng) for _ in 1:10] ==
              [0.6437422003439717,  0.930832893185952,   0.7842438837551243,
               0.31770814625739846, 0.6939887409593761,  0.27038650781966916,
               0.7341627140120408,  0.07111609732185895, 0.18504500900319684,
               0.1989539243417336]

        # ... and still in step a million draws later, which is the part that
        # would break if the modular reduction were merely almost right.
        for _ in 11:1_000_000
            rand(rng)
        end
        @test rand(rng) == 0.16846629392794513

        # The integer z in [1, m1] behind those doubles, for three seeds: the
        # default one, a tiny one, and one against the top of each modulus.
        # Taken from the same C program, printed before the normalisation.
        P = RandomDataStreams.PMF63
        zs(seed, n) = (r = MRG63k3a(seed);
                       [RandomDataStreams.combine63(RandomDataStreams.next_pair!(r)...)
                        for _ in 1:n])
        @test zs(fill(123456789, 6), 6) ==
              [5937473809595949476, 8585418077995931278, 7233373107501396343,
               2930340432071454314, 6400916347256758513, 2493875355366750018]
        @test zs([1, 2, 3, 4, 5, 6], 6) ==
              [9223371873653682447, 4676622457246299043, 6666800424851934373,
               94748724783950719,   30418792941617299,   2346650931207009463]
        @test zs([P.m1 - 1, 0, 1, P.m2 - 1, 1, 0], 6) ==
              [9223372033837736831, 8338928663777515826, 4220650215078059018,
               1264970491331603831, 8782366725965330982, 4939471023159614097]

        # The integer paths are ours, not L'Ecuyer's -- he specifies a double.
        # Locked here the way MRG32k3a's are, as a regression net.
        r0 = next_stream!(MRG63k3aGen())
        @test [rand(r0) for _ in 1:5] ==
              [0.999964376179128, 0.3293712031670167, 0.6728066002975757,
               0.8707612110911583, 0.7121206375374564]

        for (T, ref) in (
            (UInt8,   UInt8[0x18, 0xa2, 0x16]),
            (UInt16,  UInt16[0xc518, 0x59a2, 0xd016]),
            (UInt32,  UInt32[0x6d5cc518, 0xec5a59a2, 0x67cfd016]),
            (UInt64,  UInt64[0x6d5cc518ec5a59a2, 0x67cfd01621852115, 0x83d87361f770fd6e]),
            (UInt128, UInt128[0x6d5cc518ec5a59a267cfd01621852115,
                              0x83d87361f770fd6ef596a755820cadca,
                              0x2cd6bb0f94bd342704d866ea9f77f327]),
            (Int8,    Int8[24, -94, 22]),
            (Int16,   Int16[-15080, 22946, -12266]),
            (Int32,   Int32[1834796312, -329623134, 1741672470]),
            (Int64,   Int64[7880390158826756514, 7480426299555914005,
                            -8946273795171091090]),
            (Int128,  Int128[145367540460856542937830089570103140629,
                             -165029623112855383596246194477897830966,
                             59600977387313586812902884216453722919]),
            (Float32, Float32[0.7247648, 0.7058604, 0.6235378]),
            (Float16, Float16[0.2734, 0.4082, 0.02148]),
            (Bool,    Bool[0, 0, 0]),
        )
            r = next_stream!(MRG63k3aGen())
            @test [rand(r, T) for _ in 1:3] == ref
        end

        # a wide word is the concatenation of consecutive 32-bit chunks: two
        # steps for a UInt64 here, where MRG32k3a needs four
        a, b = MRG63k3a(), MRG63k3a()
        w = rand(a, UInt64)
        @test w == (UInt64(rand(b, UInt32)) << 32) | rand(b, UInt32)
        @test get_state(a) == get_state(b)
    end

    @testset "both MRGs match L'Ecuyer's C over 10^7 draws" begin
        # The reference vectors above pin the first few values. This pins the
        # whole run: an FNV-1a digest over the exact combined integer z of ten
        # million draws, for each of several seeds, taken from L'Ecuyer's own C
        # code (MRG32k3a.c and MRG63k3a.c) instrumented to print z before the
        # normalisation. A divergence anywhere -- a reduction that is wrong
        # only for rare arguments, a reordering that drops a step -- moves the
        # digest, where the first five values would not notice.
        #
        # Regenerate by running the same digest over the C implementations:
        #   h = 14695981039346656037; for each z: h = (h XOR z) * 1099511628211
        # in unsigned 64-bit arithmetic.
        fnv(h::UInt64, x::UInt64) = (h ⊻ x) * 0x100000001b3

        function digest(mk, zfun, n = 10_000_000)
            r = mk()
            h = 0xcbf29ce484222325
            z = UInt64(0)
            for _ in 1:n
                z = UInt64(zfun(r))
                h = fnv(h, z)
            end
            return h, z
        end

        z32(r) = RandomDataStreams.combine(RandomDataStreams.next_pair!(r)...)
        z63(r) = RandomDataStreams.combine63(RandomDataStreams.next_pair!(r)...)

        P32, P63 = RandomDataStreams.PMF, RandomDataStreams.PMF63

        cases = [
            ("MRG32k3a, seed 12345 x6",   () -> MRG32k3a(),
             z32, 6383324985553746743, 3871081252),
            ("MRG32k3a, seed 1..6",       () -> MRG32k3a([1, 2, 3, 4, 5, 6]),
             z32, 17608962676454277005, 2376766909),
            ("MRG32k3a, top of moduli",   () -> MRG32k3a([P32.m1 - 1, 0, 1, P32.m2 - 1, 1, 0]),
             z32, 1151014898845193120, 3582580783),
            ("MRG63k3a, seed 12345 x6",   () -> MRG63k3a(),
             z63, 212485652778388375, 1433135117478512863),
            ("MRG63k3a, seed 123456789 x6 (the C default)", () -> MRG63k3a(fill(123456789, 6)),
             z63, 17948122136114750705, 3784388861730332075),
            ("MRG63k3a, seed 1..6",       () -> MRG63k3a([1, 2, 3, 4, 5, 6]),
             z63, 8165515108107542832, 2852637435258056948),
            ("MRG63k3a, top of moduli",   () -> MRG63k3a([P63.m1 - 1, 0, 1, P63.m2 - 1, 1, 0]),
             z63, 444689602571418870, 7757865061190074490),
        ]

        for (name, mk, zfun, ref_digest, ref_last) in cases
            @testset "$name" begin
                h, z = digest(mk, zfun)
                @test h == UInt64(ref_digest)
                @test z == UInt64(ref_last)
            end
        end
    end

    @testset "MRG63k3a output properties" begin
        rng = MRG63k3a()
        u = [rand(rng) for _ in 1:100_000]
        @test all(0.0 .< u .< 1.0)
        @test abs(sum(u) / length(u) - 0.5) < 0.01

        r = next_stream!(MRG63k3aGen())
        v = [rand(r, 1:10) for _ in 1:10_000]
        @test all(1 .<= v .<= 10)
        @test count(==(1), v) > 500 && count(==(10), v) > 500

        # 63 bits per step, against 32 for MRG32k3a: the top half of a word is
        # not a near-deterministic function of the bottom half, and the whole
        # range of a UInt32 is reached
        w = [rand(r, UInt32) for _ in 1:10_000]
        @test maximum(w) > 0xff00_0000 && minimum(w) < 0x00ff_ffff
    end

    @testset "MRG63k3a jump matrices" begin
        # These are not L'Ecuyer's: he published jump matrices for MRG32k3a
        # only. They are computed at precompilation, so what is checked here is
        # the arithmetic that produced them.
        P = RandomDataStreams.PMF63
        I3 = [1 0 0; 0 1 0; 0 0 1]

        # the backward step really is the inverse of the forward one
        @test P.MatMatModM(P.A1p0, P.InvA1, P.m1) == I3
        @test P.MatMatModM(P.A2p0, P.InvA2, P.m2) == I3

        # the same formulas, applied to MRG32k3a's coefficients, reproduce the
        # InvA1 and InvA2 that L'Ecuyer published -- an independent check on
        # the derivation, since those constants are tabulated, not computed
        let m = RandomDataStreams.PMF.m1, a12 = 1403580, a13n = 810728
            iv = invmod(a13n, m)
            @test [(a12 * iv) % m, 0, m - iv] == RandomDataStreams.PMF.InvA1[1, :]
        end
        let m = RandomDataStreams.PMF.m2, a21 = 527612, a23n = 1370589
            iv = invmod(a23n, m)
            @test [0, (a21 * iv) % m, m - iv] == RandomDataStreams.PMF.InvA2[1, :]
        end

        # the substream matrix squared 100 times is the stream matrix:
        # (2^150)^(2^100) = 2^250
        @test P.MatTwoPowModM(P.A1p150, Int64(100), P.m1) == P.A1p250
        @test P.MatTwoPowModM(P.A2p150, Int64(100), P.m2) == P.A2p250

        # and the jumps mean what the documentation says they mean
        a = MRG63k3a(777); b = MRG63k3a(777)
        next_substream!(a)
        advance_state!(b, Int64(P.SUBSTREAM_EXPONENT), Int64(0))
        @test get_state(a) == get_state(b)

        gen = MRG63k3aGen(2024)
        s1 = next_stream!(gen); s2 = next_stream!(gen)
        c = copy(s1)
        advance_state!(c, Int64(P.STREAM_EXPONENT), Int64(0))
        @test get_state(c) == get_state(s2)
    end

    @testset "MRG63k3a keeps its state one step ahead" begin
        # Vigna's third optimization: a draw returns the pair already in the
        # state and computes the next one, so the internal vector is one step
        # ahead of the position. Everything public has to hide that.
        P = RandomDataStreams.PMF63
        seed = [1, 2, 3, 4, 5, 6]

        rng = MRG63k3a(seed)
        @test get_state(rng) == seed                    # the seed comes back out
        @test rng.Cg != seed                            # ... but is not what is stored
        @test RandomDataStreams._unstep63(rng.Cg) == seed
        @test RandomDataStreams._step63(RandomDataStreams._unstep63(rng.Cg)) == rng.Cg

        # the stored vector is exactly the seed advanced one step, i.e. it
        # already holds the first output in positions 3 and 6
        first = rand(copy(rng))
        @test first == RandomDataStreams.combine63(rng.Cg[3], rng.Cg[6]) * P.norm

        # get_state tracks the position, one draw at a time
        for k in 1:5
            r = MRG63k3a(seed)
            for _ in 1:k
                rand(r)
            end
            direct = MRG63k3a(seed)
            advance_state!(direct, Int64(0), Int64(k))
            @test get_state(r) == get_state(direct)
        end

        # and the seed representation is what crosses every public boundary
        @test occursin(string(seed[1]), sprint(show, MRG63k3a(seed)))
        g = MRG63k3a(); Random.seed!(g, seed)
        @test get_state(g) == seed
        @test get_state(next_stream!(MRG63k3aGen(seed))) == seed
    end

    @testset "MRG63k3a streams & substreams" begin
        gen = MRG63k3aGen()
        rng1 = next_stream!(gen)
        rng2 = next_stream!(gen)
        # the first value of each stream, taken from a copy so the streams
        # themselves stay untouched (`Ig` is internal here: the MRG63k3a state
        # runs one step ahead of the position, so it is not a seed)
        first1 = rand(copy(rng1))
        first2 = rand(copy(rng2))

        # successive streams are disjoint
        a = [rand(rng1) for _ in 1:100]
        b = [rand(rng2) for _ in 1:100]
        @test Set(a) ∩ Set(b) == Set{Float64}()

        # reset_stream! rewinds to the very beginning
        rand(rng1); rand(rng1)
        reset_stream!(rng1)
        @test rand(rng1) == first1

        # reset_substream!: no next_substream! yet -> back to the stream start
        reset_substream!(rng2)
        @test rand(rng2) == first2

        # next_substream! opens a new, different block; reset_substream! rewinds it
        v1 = rand(rng2)
        next_substream!(rng2)
        w1 = rand(rng2)
        reset_substream!(rng2)
        @test rand(rng2) == w1          # start of the new substream
        @test v1 != w1
    end

    @testset "MRG63k3a state handling" begin
        rng = next_stream!(MRG63k3aGen())
        state = get_state(rng)
        xs = [rand(rng) for _ in 1:5]

        clone = MRG63k3a(state, state, state)
        @test rand(clone) == xs[1]

        st = get_state(rng)
        st[1] += 1                      # get_state must return an independent copy
        @test get_state(rng) != st

        c = copy(rng)
        rand(c); rand(c)
        @test rand(rng) != rand(c)      # copies evolve independently

        @test_throws ArgumentError set_state!(MRG63k3a(), [0, 0, 0, 1, 1, 1])
        @test_throws ArgumentError srand!(MRG63k3a(), [0, 0, 0, 1, 1, 1])
    end

    @testset "MRG63k3a advance_state!" begin
        ref = next_stream!(MRG63k3aGen())
        vals = [rand(ref) for _ in 1:4]

        rng = MRG63k3a(fill(12345, 6))
        advance_state!(rng, Int64(2), Int64(-1))     # skip n = 2^2 - 1 = 3 values
        @test rand(rng) == vals[4]

        # backward jump: after consuming vals[4] the position is 4; n = -2^2 = -4
        advance_state!(rng, Int64(-2), Int64(0))
        @test rand(rng) == vals[1]

        # e = 0, c = k: plain forward jump
        rng2 = MRG63k3a(fill(12345, 6))
        advance_state!(rng2, Int64(0), Int64(2))
        @test rand(rng2) == vals[3]

        # a jump far beyond anything reachable by stepping, and back
        rng3 = MRG63k3a(4242)
        s0 = get_state(rng3)
        advance_state!(rng3, Int64(300), Int64(11))
        @test get_state(rng3) != s0
        advance_state!(rng3, Int64(-300), Int64(-11))
        @test get_state(rng3) == s0
    end

    @testset "MRG63k3aGen" begin
        gen = MRG63k3aGen()
        @test gen.seed == RandomDataStreams.DEFAULT_SEED63
        @test get_state(gen) == RandomDataStreams.DEFAULT_SEED63

        custom = MRG63k3aGen([7, 7, 7, 8, 8, 8])
        @test get_state(custom) == [7, 7, 7, 8, 8, 8]
        @test_throws AssertionError MRG63k3aGen([0, 0, 0, 1, 1, 1])
        @test_throws ArgumentError srand!(MRG63k3aGen(), [0, 0, 0, 1, 1, 1])

        g2 = MRG63k3aGen([1, 2, 3, 4, 5, 6])
        r = next_stream!(g2)
        @test get_state(g2) != [1, 2, 3, 4, 5, 6]   # internal seed advanced

        # the two members of the family are different generators, and neither
        # is silently substituted for the other
        @test rand(next_stream!(MRG63k3aGen(99))) != rand(next_stream!(MRG32k3aGen(99)))
    end

    @testset "Xoshiro256p reference values" begin
        seed = UInt64[0x01, 0x02, 0x03, 0x04]
        x = Xoshiro256p(seed)
        @test [RandomDataStreams.next(x) for _ in 1:5] == UInt64[
            0x0000000000000005, 0x0000c00000000007, 0x0000c00018000007,
            0x8001600018040302, 0x8061900024040305]

        x = Xoshiro256p(seed); short_jump!(x)
        @test RandomDataStreams.next(x) == 1153146630064993313
        x = Xoshiro256p(seed); long_jump!(x)
        @test RandomDataStreams.next(x) == 4237864540600467441
    end

    @testset "Xoshiro256p outputs" begin
        x = Xoshiro256p(UInt64[1, 2, 3, 4])
        u = [rand(x) for _ in 1:100_000]
        @test all(0.0 .<= u .< 1.0)
        @test abs(sum(u) / length(u) - 0.5) < 0.01

        x = Xoshiro256p(UInt64[1, 2, 3, 4])
        @test rand(x, UInt64) isa UInt64
        @test rand(x, Float32) isa Float32
        @test rand(x, Float16) isa Float16

        v = [rand(x, Int64(1):Int64(10)) for _ in 1:10_000]
        @test all(1 .<= v .<= 10)
        counts = [count(==(i), v) for i in 1:10]
        @test maximum(counts) < 1500 && minimum(counts) > 500
    end

    @testset "Xoshiro256p streams & substreams" begin
        seed = UInt64[0x01, 0x02, 0x03, 0x04]
        x = Xoshiro256p(seed)
        u0 = RandomDataStreams.next(x)

        srand!(x, seed)
        @test RandomDataStreams.next(x) == u0

        x = Xoshiro256p(seed)
        short_jump!(x)
        after_short = RandomDataStreams.next(x)
        reset_substream!(x)
        @test RandomDataStreams.next(x) == after_short

        x = Xoshiro256p(seed)
        long_jump!(x)
        after_long = RandomDataStreams.next(x)
        reset_stream!(x)
        @test RandomDataStreams.next(x) == after_long

        next_substream!(x)
        ns = RandomDataStreams.next(x)
        reset_substream!(x)
        @test RandomDataStreams.next(x) == ns
    end

    @testset "Xoshiro256p state" begin
        seed = UInt64[0x01, 0x02, 0x03, 0x04]
        x = Xoshiro256p(seed)
        st = get_state(x)
        @test st isa Vector{UInt64} && length(st) == 4
        expected = RandomDataStreams.next(x)
        y = Xoshiro256p(st)
        @test RandomDataStreams.next(y) == expected

        c = copy(x)
        RandomDataStreams.next(c)
        @test RandomDataStreams.next(x) != RandomDataStreams.next(c)
    end

    @testset "Xoshiro256plusGen" begin
        seed = UInt64[0x0d, 0x0e, 0x0a, 0x0d]
        gen = Xoshiro256plusGen(seed)
        @test get_state(gen) == seed

        seed[1] = 0xff                     # generator must own its copy
        @test get_state(gen)[1] == 0x0d

        r1 = next_stream!(gen)
        r2 = next_stream!(gen)
        @test get_state(r1) != get_state(r2)

        srand!(gen, UInt64[1, 1, 1, 1])
        @test get_state(gen) == UInt64[1, 1, 1, 1]
        @test_throws ArgumentError Xoshiro256plusGen(UInt64[1, 2, 3])
    end


    @testset "Philox reference values" begin
        # Known-answer tests from the Random123 reference implementation
        # (Salmon, Moraes, Dror & Shaw, SC 2011), Philox4x32-10.
        z = ntuple(_ -> UInt32(0), 4)
        @test RandomDataStreams.philox(z, (UInt32(0), UInt32(0))) ==
              (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)

        f = typemax(UInt32)
        @test RandomDataStreams.philox((f, f, f, f), (f, f)) ==
              (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)

        @test RandomDataStreams.philox((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
                                       (0xa4093822, 0x299f31d0)) ==
              (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)

        # the RNG wrapper must hand out those same words, in order
        rng = next_stream!(PhiloxGen())
        @test [rand(rng, UInt32) for _ in 1:4] ==
              UInt32[0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]

        # 64-bit draws pair two consecutive 32-bit words, low word first
        rng = next_stream!(PhiloxGen())
        @test RandomDataStreams.next(rng) == 0xe169c58d6627e8d5

        # ... and the same vectors for Philox4x64-10
        z64 = ntuple(_ -> UInt64(0), 4)
        @test RandomDataStreams.philox(z64, (UInt64(0), UInt64(0))) ==
              (0x16554d9eca36314c, 0xdb20fe9d672d0fdc,
               0xd7e772cee186176b, 0x7e68b68aec7ba23b)

        g = typemax(UInt64)
        @test RandomDataStreams.philox((g, g, g, g), (g, g)) ==
              (0x87b092c3013fe90b, 0x438c3c67be8d0224,
               0x9cc7d7c69cd777b6, 0xa09caebf594f0ba0)

        @test RandomDataStreams.philox((0x243f6a8885a308d3, 0x13198a2e03707344,
                                        0xa4093822299f31d0, 0x082efa98ec4e6c89),
                                       (0x452821e638d01377, 0xbe5466cf34e90c6c)) ==
              (0xa528f45403e61d95, 0x38c72dbd566e9788,
               0xa5a1610e72fd18b5, 0x57bd43b5e52b7fe6)

        rng = next_stream!(Philox4x64Gen())
        @test [rand(rng, UInt64) for _ in 1:4] ==
              UInt64[0x16554d9eca36314c, 0xdb20fe9d672d0fdc,
                     0xd7e772cee186176b, 0x7e68b68aec7ba23b]
    end

    @testset "Philox outputs" begin
        rng = next_stream!(PhiloxGen())
        v = [rand(rng) for _ in 1:10_000]
        @test all(0 .<= v .< 1)
        @test 0.48 < sum(v) / length(v) < 0.52

        rng = next_stream!(PhiloxGen())
        w = [rand(rng, 1:10) for _ in 1:10_000]
        @test all(1 .<= w .<= 10)
        @test count(==(1), w) > 500 && count(==(10), w) > 500

        # all four words of a block are consumed before the counter moves on
        rng = next_stream!(PhiloxGen())
        for _ in 1:4
            rand(rng, UInt32)
        end
        @test get_state(rng)[1] == 1
    end

    @testset "Philox streams & substreams" begin
        gen = PhiloxGen()
        s1 = next_stream!(gen)
        s2 = next_stream!(gen)
        @test get_state(s1)[2] != get_state(s2)[2]        # distinct keys
        @test isempty(intersect([rand(s1) for _ in 1:100], [rand(s2) for _ in 1:100]))

        # a substream is a jump of 2^64 blocks in the counter
        reset_stream!(s1)
        v1 = rand(s1, UInt32)
        next_substream!(s1)
        @test get_state(s1)[1] == UInt128(1) << 64
        w1 = rand(s1, UInt32)
        @test v1 != w1

        rand(s1, UInt32)
        reset_substream!(s1)
        @test rand(s1, UInt32) == w1                      # start of the substream
        reset_stream!(s1)
        @test rand(s1, UInt32) == v1                      # start of the stream
    end

    @testset "Philox state" begin
        rng = PhiloxRNG((UInt32(3), UInt32(7)))
        ctr, key, _, idx = get_state(rng)
        @test ctr == 0 && key == (UInt32(3), UInt32(7)) && idx == 5

        srand!(rng, UInt32[1, 2])
        @test get_state(rng)[2] == (UInt32(1), UInt32(2))
        @test_throws ArgumentError srand!(rng, UInt32[1, 2, 3])

        gen = PhiloxGen()
        srand!(gen, UInt32[4, 5])
        @test get_state(gen) == (UInt32(4), UInt32(5))
        @test get_state(next_stream!(gen))[2] == (UInt32(4), UInt32(5))

        # advance_state! counts rand(rng) draws, like every other generator --
        # including across block boundaries. A block of Philox4x32-10 holds four
        # 32-bit words, hence two draws.
        base = let q = next_stream!(PhiloxGen()); [rand(q) for _ in 1:20] end
        for n in (1, 2, 3, 5, 9)
            q = next_stream!(PhiloxGen())
            advance_state!(q, 0, n)
            @test [rand(q) for _ in 1:3] == base[n+1:n+3]
        end

        q = next_stream!(PhiloxGen())              # one draw = two 32-bit words
        advance_state!(q, 0, 1)
        ctr, _, _, idx = get_state(q)
        @test ctr == 1 && idx == 3

        q = next_stream!(PhiloxGen())              # backwards from mid-block
        for _ in 1:7
            rand(q)
        end
        advance_state!(q, 0, -5)
        @test [rand(q) for _ in 1:2] == base[3:4]

        q = next_stream!(PhiloxGen())              # a substream is 2^65 draws
        advance_state!(q, 65, 0)
        @test get_state(q)[1] == UInt128(1) << 64

        a = next_stream!(PhiloxGen())
        b = copy(a)
        @test rand(a) == rand(b)                          # copies are independent
        rand(b)
        @test rand(a) != rand(b)
    end



    @testset "Threefry reference values" begin
        # Known-answer tests from the Random123 reference implementation
        # (Salmon, Moraes, Dror & Shaw, SC 2011), Threefry-4x-20.
        z32 = ntuple(_ -> UInt32(0), 4)
        @test RandomDataStreams.threefry(z32, z32) ==
              (0x9c6ca96a, 0xe17eae66, 0xfc10ecd4, 0x5256a7d8)

        f = typemax(UInt32)
        @test RandomDataStreams.threefry((f, f, f, f), (f, f, f, f)) ==
              (0x2a881696, 0x57012287, 0xf6c7446e, 0xa16a6732)

        @test RandomDataStreams.threefry((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
                                         (0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89)) ==
              (0x59cd1dbb, 0xb8879579, 0x86b5d00c, 0xac8b6d84)

        z64 = ntuple(_ -> UInt64(0), 4)
        @test RandomDataStreams.threefry(z64, z64) ==
              (0x09218ebde6c85537, 0x55941f5266d86105,
               0x4bd25e16282434dc, 0xee29ec846bd2e40b)

        g = typemax(UInt64)
        @test RandomDataStreams.threefry((g, g, g, g), (g, g, g, g)) ==
              (0x29c24097942bba1b, 0x0371bbfb0f6f4e11,
               0x3c231ffa33f83a1c, 0xcd29113fde32d168)

        @test RandomDataStreams.threefry((0x243f6a8885a308d3, 0x13198a2e03707344,
                                          0xa4093822299f31d0, 0x082efa98ec4e6c89),
                                         (0x452821e638d01377, 0xbe5466cf34e90c6c,
                                          0xbe5466cf34e90c6c, 0xc0ac29b7c97c50dd)) ==
              (0xa7e8fde591651bd9, 0xbaafd0c30138319b,
               0x84a5c1a729e685b9, 0x901d406ccebc1ba4)

        # the RNG wrappers must hand out those same words, in order
        rng = next_stream!(Threefry4x64Gen())
        @test [rand(rng, UInt64) for _ in 1:4] ==
              UInt64[0x09218ebde6c85537, 0x55941f5266d86105,
                     0x4bd25e16282434dc, 0xee29ec846bd2e40b]

        rng = next_stream!(Threefry4x32Gen())
        @test [rand(rng, UInt32) for _ in 1:4] ==
              UInt32[0x9c6ca96a, 0xe17eae66, 0xfc10ecd4, 0x5256a7d8]

        # the round count is a real parameter, not a constant tuned for 20:
        # the 13-round variant matches its own Random123 vector
        @test RandomDataStreams.threefry(z64, z64, Val(13)) ==
              (0x4071fabee1dc8e05, 0x02ed3113695c9c62,
               0x397311b5b89f9d49, 0xe21292c3258024bc)
        @test RandomDataStreams.threefry(z32, z32, Val(13)) ==
              (0x531c7e4f, 0x39491ee5, 0x2c855a92, 0x3d6abf9a)

        # no multiplication anywhere: the key schedule is pure xor
        @test RandomDataStreams._threefry_ks((UInt32(1), UInt32(2), UInt32(4), UInt32(8)))[5] ==
              UInt32(1) ⊻ UInt32(2) ⊻ UInt32(4) ⊻ UInt32(8) ⊻ 0x1BD11BDA
    end

    @testset "counter-based variant matrix" begin
        # every stream/state routine must behave identically for each variant
        variants = [
            (PhiloxRNG,       PhiloxGen,       UInt32, 2, "Philox4x32-10"),
            (Philox4x64RNG,   Philox4x64Gen,   UInt64, 2, "Philox4x64-10"),
            (Threefry4x32RNG, Threefry4x32Gen, UInt32, 4, "Threefry4x32-20"),
            (Threefry4x64RNG, Threefry4x64Gen, UInt64, 4, "Threefry4x64-20"),
        ]
        for (T, G, W, K, name) in variants
            key = W[3 + i for i in 1:K]
            toolong = W[1 for _ in 1:(K + 1)]
            gen = G()
            s1 = next_stream!(gen)
            s2 = next_stream!(gen)
            @test s1 isa T && s1 isa AbstractRNG
            @test get_state(s1)[2] != get_state(s2)[2]
            @test isempty(intersect([rand(s1) for _ in 1:100], [rand(s2) for _ in 1:100]))

            r = T()
            v1 = rand(r, W)
            next_substream!(r)
            @test get_state(r)[1] == UInt128(1) << 64
            w1 = rand(r, W)
            rand(r, W)
            reset_substream!(r)
            @test rand(r, W) == w1
            reset_stream!(r)
            @test rand(r, W) == v1

            srand!(r, key)
            @test get_state(r)[2] == NTuple{K,W}(key)
            @test_throws ArgumentError srand!(r, toolong)

            srand!(gen, key)
            @test get_state(gen) == NTuple{K,W}(key)
            @test get_state(next_stream!(gen))[2] == NTuple{K,W}(key)

            a = T(); b = copy(a)
            @test rand(a) == rand(b)
            rand(b)
            @test rand(a) != rand(b)

            y = T(); z = T()
            Random.seed!(y, 987); Random.seed!(z, 987)
            @test [rand(y) for _ in 1:20] == [rand(z) for _ in 1:20]

            io = IOBuffer()
            show(io, T()); @test occursin(name, String(take!(io)))
            show(io, G()); @test occursin(name, String(take!(io)))
        end
    end


    @testset "PCG reference values" begin
        # Known-answer vectors produced by NumPy 1.26.4, whose PCG64 is the
        # default bit generator of `numpy.random.default_rng`. Both the outputs
        # and the resulting state must agree, which pins the multiplier, the
        # increment, the output permutation and the order of step and output.
        S0 = (UInt128(0x0123456789abcdef) << 64) | UInt128(0x0123456789abcdef)

        r = PCG64(S0)
        @test [rand(r, UInt64) for _ in 1:6] == UInt64[
            0xa12dea8c95158441, 0x242041db494e6da8, 0x2cb3dccd41360faa,
            0x4ceae7e3765e3633, 0x65ddd0b932ceeb6b, 0x5bd4867ba1e071d4]
        @test get_state(r) == (UInt128(0xbfd04c4cd56d6e75) << 64) | UInt128(0x873a61a69650be85)

        d = PCG64DXSM(S0)
        @test [rand(d, UInt64) for _ in 1:6] == UInt64[
            0x5a3d0ba6a739bb5e, 0xa2fe1f98fc08aa3a, 0x624216f30d9f745d,
            0x90f0d7069d4cf377, 0x6a6c25608fb7c01d, 0x64220ea4ada77336]
        @test get_state(d) == (UInt128(0x4fa4d3f3a35dcfff) << 64) | UInt128(0xf57dd659648a1dd5)

        # the two variants really are different generators
        @test PCG64(S0) |> x -> rand(x, UInt64) != rand(PCG64DXSM(S0), UInt64)
    end


    @testset "PCG jumps are the closed-form LCG advance" begin
        # The whole point of putting PCG here: the jump is Brown's formula, so
        # any distance is O(log n) and must coincide exactly with taking that
        # many draws. Checked for both variants, since they use different
        # multipliers.
        for T in (PCG64, PCG64DXSM)
            for n in (1, 2, 37, 1021)
                a = T(12345); b = T(12345)
                for _ in 1:n
                    rand(a)
                end
                advance_state!(b, 0, n)
                @test rand(a) == rand(b)
            end

            # backwards costs the same and lands in the same place
            a = T(999)
            x = [rand(a) for _ in 1:50]
            advance_state!(a, 0, -50)
            @test [rand(a) for _ in 1:50] == x

            # substreams are 2^64 + 1 apart, streams 2^32*(2^64 + 1) + 1.
            # The distances are odd on purpose: a power-of-two jump gives a
            # jump multiplier congruent to 1 modulo a large power of two, and
            # the resulting streams fail the interleaved battery.
            a, b = T(7), T(7)
            next_substream!(a); advance_state!(b, 64, 1)
            @test rand(a) == rand(b)

            a, b = T(7), T(7)
            long_jump!(a); advance_state!(b, 96, 2^32 + 1)
            @test rand(a) == rand(b)

            # what makes those distances safe: v2(a^n - 1) = 2 + v2(n), so an
            # odd distance keeps the jump multiplier away from the identity
            mult = T === PCG64 ? RandomDataStreams.PCG_MULT_128 : RandomDataStreams.PCG_MULT_CM
            for d in (RandomDataStreams._PCG_SHORT_JUMP, RandomDataStreams._PCG_LONG_JUMP)
                @test isodd(d)
                jm = powermod(big(mult), big(d), big(2)^128) % UInt128
                @test trailing_zeros(jm - one(UInt128)) <= 8
            end

            # boundaries are anchored: how much was consumed does not matter
            a = T(3); next_substream!(a)
            u = [rand(a) for _ in 1:4]
            b = T(3)
            for _ in 1:37
                rand(b)
            end
            next_substream!(b)
            @test [rand(b) for _ in 1:4] == u
        end

        g = PCG64Gen(20260830)
        s1, s2 = next_stream!(g), next_stream!(g)
        @test isempty(intersect([rand(s1, UInt64) for _ in 1:5000],
                                [rand(s2, UInt64) for _ in 1:5000]))
    end


    @testset "PCG stream and substream addressing" begin
        # The jump identities above say a jump of n lands n draws ahead. This
        # says the stream/substream *composition* lands where the layout
        # claims: stream k at k*D_stream, its substream j at
        # k*D_stream + j*D_substream, with the substreams of one stream fitting
        # inside the stream spacing.
        DS = RandomDataStreams._PCG_SHORT_JUMP
        DL = RandomDataStreams._PCG_LONG_JUMP

        # 2^32 substreams of 2^64 draws must end below the next stream's start
        @test (UInt128(2)^32 - 1) * DS + UInt128(2)^64 < DL

        for (T, G, mult) in ((PCG64,     PCG64Gen,     RandomDataStreams.PCG_MULT_128),
                             (PCG64DXSM, PCG64DXSMGen, RandomDataStreams.PCG_MULT_CM))
            seed = UInt128(20260830)

            gen = G(seed)
            for k in 0:3
                rng = next_stream!(gen)
                @test get_state(rng) ==
                      RandomDataStreams._lcg_advance(seed, (UInt128(k) * DL) % UInt128, mult)
            end

            gen = G(seed)
            for k in 0:2
                rng = next_stream!(gen)
                for j in 0:3
                    @test get_state(rng) == RandomDataStreams._lcg_advance(
                        seed, (UInt128(k) * DL + UInt128(j) * DS) % UInt128, mult)
                    rand(rng); rand(rng)          # consumption must not move the anchor
                    next_substream!(rng)
                end
            end

            # the three checkpoints move exactly as the interface promises
            r = T(seed); short_jump!(r)
            @test r.Cg == r.Bg && r.Ig == seed
            r = T(seed); long_jump!(r)
            @test r.Cg == r.Bg == r.Ig != seed

            # streams and substreams are disjoint in practice, not just in theory
            gen = G(seed)
            vals = [[rand(s, UInt64) for _ in 1:5000] for s in (next_stream!(gen), next_stream!(gen),
                                                                next_stream!(gen), next_stream!(gen))]
            @test length(unique(vcat(vals...))) == 4 * 5000

            rng = next_stream!(G(seed))
            subs = UInt64[]
            for _ in 1:4
                append!(subs, [rand(rng, UInt64) for _ in 1:5000])
                next_substream!(rng)
            end
            @test length(unique(subs)) == 4 * 5000

            # srand! on the generator object rewinds what next_stream! hands out
            gen = G(seed)
            first_stream = get_state(next_stream!(gen))
            next_stream!(gen); next_stream!(gen)
            srand!(gen, seed)
            @test get_state(next_stream!(gen)) == first_stream
        end
    end


    @testset "PCG increment-based streams are not independent" begin
        # Why the package fixes the increment instead of exposing it as a
        # stream parameter, as PCG and NumPy do. Writing t_n = s_n + h and
        # matching the two recurrences gives h*(a - 1) = c1 - c2, solvable
        # whenever 4 divides c1 - c2 -- half of all pairs of odd increments.
        # For those pairs the two "independent streams" are one sequence
        # translated by a constant, which no output permutation undoes.
        a  = RandomDataStreams.PCG_MULT_128
        c1 = RandomDataStreams.PCG_INCREMENT

        @test a % 4 == 1                                  # forced by full period
        @test (a - one(UInt128)) % 4 == 0                  # so v2(a - 1) >= 2
        @test (a - one(UInt128)) % 8 != 0                  # and exactly 2

        h  = one(UInt128)
        c2 = c1 - (a - one(UInt128))                       # makes h = 1 a solution
        @test isodd(c2)                                    # still a valid increment

        s = (UInt128(0x9e3779b97f4a7c15) << 64) | UInt128(0xf39cc0605cedc835)
        t = s + h
        separated = false
        for _ in 1:1000
            s = s * a + c1
            t = t * a + c2
            t - s == h || (separated = true; break)
        end
        @test !separated
    end


    @testset "uniform constructor and generator-object API" begin
        # The package promises one interface across every generator. That claim
        # is only as good as its weakest family, so it is asserted here for all
        # seventeen, rather than for the handful the interface testset samples.
        #
        # The seeding rule under test: a value in the family's own
        # representation (its seed vector, or a UInt128 for PCG) IS the state or
        # key; any other integer is a seed, expanded through splitmix64. So
        # `T(12345)` means the same kind of thing everywhere, and matches
        # `Random.seed!(T(), 12345)`.
        R = RandomDataStreams
        families = [
            ("MRG32k3a",        MRG32k3a,          MRG32k3aGen,        [1, 2, 3, 4, 5, 6]),
            ("MRG63k3a",        MRG63k3a,          MRG63k3aGen,        [1, 2, 3, 4, 5, 6]),
            ("Xoroshiro128p",   R.Xoroshiro128p,   R.Xoroshiro128pGen,  UInt64[1, 2]),
            ("Xoroshiro128ss",  R.Xoroshiro128ss,  R.Xoroshiro128ssGen, UInt64[1, 2]),
            ("Xoroshiro128pp",  R.Xoroshiro128pp,  R.Xoroshiro128ppGen, UInt64[1, 2]),
            ("Xoshiro256p",     Xoshiro256p,       Xoshiro256plusGen,   UInt64[1, 2, 3, 4]),
            ("Xoshiro256ss",    R.Xoshiro256ss,    R.Xoshiro256ssGen,   UInt64[1, 2, 3, 4]),
            ("Xoshiro256pp",    R.Xoshiro256pp,    R.Xoshiro256ppGen,   UInt64[1, 2, 3, 4]),
            ("Xoshiro512p",     R.Xoshiro512p,     R.Xoshiro512pGen,    UInt64[1, 2, 3, 4, 5, 6, 7, 8]),
            ("Xoshiro512ss",    R.Xoshiro512ss,    R.Xoshiro512ssGen,   UInt64[1, 2, 3, 4, 5, 6, 7, 8]),
            ("Xoshiro512pp",    R.Xoshiro512pp,    R.Xoshiro512ppGen,   UInt64[1, 2, 3, 4, 5, 6, 7, 8]),
            ("PCG64",           PCG64,             PCG64Gen,            UInt64[1, 2]),
            ("PCG64DXSM",       PCG64DXSM,         PCG64DXSMGen,        UInt64[1, 2]),
            ("Philox4x32-10",   PhiloxRNG,         PhiloxGen,           UInt32[1, 2]),
            ("Philox4x64-10",   Philox4x64RNG,     Philox4x64Gen,       UInt64[1, 2]),
            ("Threefry4x32-20", Threefry4x32RNG,   Threefry4x32Gen,     UInt32[1, 2, 3, 4]),
            ("Threefry4x64-20", Threefry4x64RNG,   Threefry4x64Gen,     UInt64[1, 2, 3, 4]),
        ]
        @test length(families) == 17

        for (name, T, G, seed) in families
            @testset "$name" begin
                # three constructor forms, on both the stream and the generator
                @test G()      isa R.AbstractRNGStream
                @test G(12345) isa R.AbstractRNGStream
                @test G(seed)  isa R.AbstractRNGStream
                @test T()      isa R.AbstractStreamableRNG
                @test T(12345) isa R.AbstractStreamableRNG
                @test T(seed)  isa R.AbstractStreamableRNG

                # an integer seed is deterministic, and means the same thing as
                # seeding after construction
                @test rand(T(777)) == rand(T(777))
                @test rand(next_stream!(G(777))) == rand(next_stream!(G(777)))
                r = T()
                Random.seed!(r, 777)
                r2 = T(777)                       # one instance, not one per draw
                @test [rand(r) for _ in 1:4] == [rand(r2) for _ in 1:4]

                # the generator object saves, restores and reseeds its position
                g = G(seed)
                st = get_state(g)
                firststream = get_state(next_stream!(g))
                next_stream!(g); next_stream!(g)
                set_state!(g, st)
                @test get_state(next_stream!(g)) == firststream
                srand!(g, seed)
                @test get_state(next_stream!(g)) == firststream
                # two-argument `show` is what interpolation and container
                # display call, so it has to stay on one line; the REPL's
                # `text/plain` rendering is the one that may spread out
                @test occursin(r"\S", sprint(show, g))
                @test !occursin('\n', sprint(show, g))
                @test occursin('\n', sprint(show, MIME"text/plain"(), g))

                # the portable navigation contract, returning the object each time
                x = next_stream!(G(seed))
                @test next_substream!(x)  === x
                @test reset_substream!(x) === x
                @test reset_stream!(x)    === x
                @test advance_state!(x, 0, 5)  === x
                @test advance_state!(x, 0, -5) === x
                @test set_state!(x, get_state(x)) === x
                @test copy(x) !== x
                @test occursin(r"\S", sprint(show, x))
                @test !occursin('\n', sprint(show, x))
                @test occursin('\n', sprint(show, MIME"text/plain"(), x))

                # ranges go through `Random`'s sampler for every family: in
                # bounds, and the same error as the standard library on an
                # empty range rather than a `DivideError` from folding with `%`
                @test all(3 .<= [rand(x, 3:9) for _ in 1:200] .<= 9)
                @test_throws ArgumentError rand(x, 1:0)
            end
        end
    end


    @testset "uniform stream interface" begin
        # code written against AbstractStreamableRNG must work for every
        # generator: same navigation, same get_state/set_state! round-trip,
        # same srand! semantics.
        makers = [
            ("MRG32k3a",      () -> MRG32k3a([42, 1, 2, 3, 4, 5]), [7, 7, 7, 8, 8, 8]),
            ("MRG63k3a",      () -> MRG63k3a([42, 1, 2, 3, 4, 5]), [7, 7, 7, 8, 8, 8]),
            ("Xoshiro256pp",  () -> Xoshiro256pp(fill(UInt64(42), 4)), UInt64[1, 2, 3, 4]),
            ("Xoroshiro128ss",() -> Xoroshiro128ss(fill(UInt64(42), 2)), UInt64[5, 6]),
            ("Philox4x32-10", () -> next_stream!(PhiloxGen()), UInt32[3, 4]),
            ("Philox4x64-10", () -> next_stream!(Philox4x64Gen()), UInt64[3, 4]),
            ("Threefry4x64-20", () -> next_stream!(Threefry4x64Gen()), UInt64[3, 4, 5, 6]),
            ("Threefry4x32-20", () -> next_stream!(Threefry4x32Gen()), UInt32[3, 4, 5, 6]),
            ("PCG64",         () -> next_stream!(PCG64Gen(42)), 98765),
            ("PCG64DXSM",     () -> next_stream!(PCG64DXSMGen(42)), 98765),
        ]
        for (name, mk, seed) in makers
            @testset "$name" begin
                r = mk()
                @test r isa RandomDataStreams.AbstractStreamableRNG

                # get_state / set_state! are inverses
                for _ in 1:3
                    rand(r)
                end
                st = get_state(r)
                expected = [rand(r) for _ in 1:5]
                set_state!(r, st)
                @test [rand(r) for _ in 1:5] == expected

                # ... and restoring does not disturb the substream anchor
                set_state!(r, st)
                reset_substream!(r)
                v0 = rand(r)
                reset_substream!(r)
                @test rand(r) == v0

                # advance_state! counts draws, both ways, for every generator
                a = mk()
                base = [rand(a) for _ in 1:12]
                b = mk()
                advance_state!(b, 0, 5)
                @test [rand(b) for _ in 1:3] == base[6:8]
                advance_state!(b, 0, -8)
                @test [rand(b) for _ in 1:3] == base[1:3]

                # srand! rewinds the stream and substream boundaries
                c = mk()
                srand!(c, seed)
                first = [rand(c) for _ in 1:4]
                next_substream!(c)
                rand(c)
                srand!(c, seed)
                @test [rand(c) for _ in 1:4] == first
                reset_stream!(c)
                @test [rand(c) for _ in 1:4] == first
            end
        end
    end


    @testset "array fill matches repeated draws" begin
        # rand! on a counter-based generator writes whole blocks straight into
        # the array. It must produce exactly what the scalar path would have
        # produced, in the same order and leaving the same state -- including
        # when the generator does not start on a block boundary, and for
        # lengths that do not divide the block size.
        variants = [
            ("Philox4x32-10",   () -> next_stream!(PhiloxGen()),       UInt32),
            ("Philox4x64-10",   () -> next_stream!(Philox4x64Gen()),   UInt64),
            ("Threefry4x32-20", () -> next_stream!(Threefry4x32Gen()), UInt32),
            ("Threefry4x64-20", () -> next_stream!(Threefry4x64Gen()), UInt64),
        ]
        for (name, mk, W) in variants
            @testset "$name" begin
                for T in (Float64, UInt64, W), n in (0, 1, 3, 4, 5, 8, 9, 17), pre in (0, 1, 2, 3)
                    a, b = mk(), mk()
                    for _ in 1:pre                 # start off a block boundary
                        rand(a, W); rand(b, W)
                    end
                    v = Vector{T}(undef, n)
                    rand!(a, v)
                    @test v == [rand(b, T) for _ in 1:n]
                    @test get_state(a)[1] == get_state(b)[1]     # counter
                    @test get_state(a)[4] == get_state(b)[4]     # index in block
                end
            end
        end

        # MRG32k3a defined its wide unsigned draws on `::Type` only, so the
        # Sampler machinery behind rand! had no method and this threw. Both
        # members of the MRG family assemble words the same way, so both are
        # checked.
        for M in (MRG32k3a, MRG63k3a)
            a, b = M(), M()
            v = Vector{UInt32}(undef, 5)
            rand!(a, v)
            @test v == [rand(b, UInt32) for _ in 1:5]

            a, b = M(), M()
            w = Vector{UInt128}(undef, 3)
            rand!(a, w)
            @test w == [rand(b, UInt128) for _ in 1:3]

            a, b = M(), M()
            u = Vector{UInt64}(undef, 4)
            rand!(a, u)
            @test u == [rand(b, UInt64) for _ in 1:4]
        end
    end


    @testset "stateless addressing matches the stream object" begin
        # A counter-based draw must be recomputable from (key, substream,
        # index) alone, with no stream object: that identity is what lets a
        # kernel, or any external consumer, address the same sequence the host
        # assigned. Documented in the streams page; locked here.
        function direct_word(key::NTuple{2,UInt32}, substream::Integer, i::Integer)
            block, word = divrem(i, 4)               # four 32-bit words per block
            ctr = (UInt128(substream) << 64) | UInt128(block)
            c = (ctr % UInt32, (ctr >> 32) % UInt32,
                 (ctr >> 64) % UInt32, (ctr >> 96) % UInt32)
            return RandomDataStreams.philox(c, key)[word + 1]
        end

        gen = PhiloxGen()
        for _ in 1:3
            rng = next_stream!(gen)
            key = get_state(rng)[2]
            for s in (0, 1, 5)
                reset_stream!(rng)
                for _ in 1:s
                    next_substream!(rng)
                end
                @test [rand(rng, UInt32) for _ in 0:9] == [direct_word(key, s, i) for i in 0:9]
            end
        end

        # the counterpart for recurrence-based generators: the host hands out
        # as many non-overlapping starting states as there are tasks
        g = Xoshiro256plusGen(UInt64[1, 2, 3, 4])
        seeds = [get_state(next_stream!(g)) for _ in 1:4]
        @test length(unique(seeds)) == 4
    end

    @testset "show methods" begin
        io = IOBuffer()
        show(io, MRG32k3a())
        @test occursin("MRG32k3a", String(take!(io)))
        show(io, MRG32k3aGen())
        @test occursin("MRG32k3a", String(take!(io)))
        show(io, MRG63k3a())
        @test occursin("MRG63k3a", String(take!(io)))
        show(io, MRG63k3aGen())
        @test occursin("MRG63k3a", String(take!(io)))
        show(io, Xoshiro256p(UInt64[1, 2, 3, 4]))
        @test occursin("Xoshiro256plus", String(take!(io)))
        show(io, Xoshiro256plusGen(UInt64[1, 2, 3, 4]))
        @test occursin("Xoshiro256plus", String(take!(io)))
        show(io, PhiloxRNG())
        @test occursin("Philox", String(take!(io)))
        show(io, PhiloxGen())
        @test occursin("Philox", String(take!(io)))
    end

    @testset "Random API integration" begin
        m = MRG32k3a()
        x = Xoshiro256p(UInt64[1, 2, 3, 4])
        @test m isa AbstractRNG && x isa AbstractRNG

        m1 = copy(m); m2 = copy(m)
        @test rand(m1) == rand(m2)      # identical copies produce identical values
        rand(m2)
        @test rand(m1) != rand(m2)
    end

    @testset "xoshiro/xoroshiro families" begin
        # Reference values validated byte-for-byte against the original C
        # implementations (http://xoshiro.di.unimi.it), seeded via splitmix64(0x12345).
        function splitmix_seed(nwords)
            sm = UInt64(0x12345)
            out = Vector{UInt64}(undef, nwords)
            for w in 1:nwords
                sm += 0x9E3779B97f4A7C15
                z = sm
                z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
                z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
                v = z ⊻ (z >> 31)
                out[w] = v == 0 ? UInt64(1) : v
            end
            out
        end

        refs = [
            ("x128p", RandomDataStreams.Xoroshiro128p,
             [16078810691027958445, 16308033409981753773, 2459436067589474410],
             UInt64(3256517475130558849), UInt64(9914398136390617976)),
            ("x128ss", RandomDataStreams.Xoroshiro128ss,
             [1368380786106859145, 4084486448798807325, 5543098162342028434],
             UInt64(13270538264984994966), UInt64(11920770361629577860)),
            ("x128pp", RandomDataStreams.Xoroshiro128pp,
             [1673598725564225250, 4573856857203627046, 16115075492912290848],
             UInt64(6730732423063272729), UInt64(3990584750745030224)),
            ("x256p", RandomDataStreams.Xoshiro256p,
             [9347869054091033467, 4771635083329966370, 15370539120085914707],
             UInt64(5160113462924517804), UInt64(15646076907341602619)),
            ("x256ss", RandomDataStreams.Xoshiro256ss,
             [1368380786106859145, 14675459340006078963, 5910136538917692306],
             UInt64(10763142405286072350), UInt64(16578569578879967412)),
            ("x256pp", RandomDataStreams.Xoshiro256pp,
             [17674668620475692482, 5475324084790473842, 16549419549621501831],
             UInt64(16231616900645524224), UInt64(13257993063221304277)),
            ("x512p", RandomDataStreams.Xoshiro512p,
             [17616468189221365395, 12780371331858795586, 9670281856951386802],
             UInt64(16537263715482372943), UInt64(2902670574299405512)),
            ("x512ss", RandomDataStreams.Xoshiro512ss,
             [1368380786106859145, 14675459340006078963, 11156859077635828455],
             UInt64(2926543538459652119), UInt64(10490133114023169214)),
            ("x512pp", RandomDataStreams.Xoshiro512pp,
             [4070133962183833323, 16966997717490759717, 12310346514993803299],
             UInt64(13604293740491481571), UInt64(151834235382922295)),
        ]
        for (name, T, seq, shortv, longv) in refs
            seed = splitmix_seed(statewords(T))
            x = T(seed)
            @test [RandomDataStreams.next(x) for _ in 1:3] == seq
            y = T(copy(seed)); short_jump!(y)
            @test RandomDataStreams.next(y) == shortv
            z = T(copy(seed)); long_jump!(z)
            @test RandomDataStreams.next(z) == longv
        end

        allvariants = [
            ("Xoroshiro128plus",     RandomDataStreams.Xoroshiro128p,    RandomDataStreams.Xoroshiro128pGen),
            ("Xoroshiro128starstar", RandomDataStreams.Xoroshiro128ss,   RandomDataStreams.Xoroshiro128ssGen),
            ("Xoshiro256plusplus",   RandomDataStreams.Xoshiro256pp,     RandomDataStreams.Xoshiro256ppGen),
            ("Xoshiro512plus",       RandomDataStreams.Xoshiro512p,      RandomDataStreams.Xoshiro512pGen),
            ("Xoshiro512starstar",   RandomDataStreams.Xoshiro512ss,     RandomDataStreams.Xoshiro512ssGen),
        ]
        for (shown_name, T, G) in allvariants
            n = statewords(T)
            x = T(fill(UInt64(0xdecafbad), n))
            u = [rand(x) for _ in 1:10_000]
            @test all(0.0 .<= u .< 1.0)
            @test abs(sum(u) / length(u) - 0.5) < 0.02
            @test rand(x, UInt64) isa UInt64
            @test rand(x, Float32) isa Float32
            v = [rand(x, Int64(1):Int64(100)) for _ in 1:1000]
            @test all(1 .<= v .<= 100)

            x = T(fill(UInt64(0xbeef), n))
            w0 = RandomDataStreams.next(x)
            next_substream!(x)
            w1 = RandomDataStreams.next(x)
            reset_substream!(x)
            @test RandomDataStreams.next(x) == w1
            reset_stream!(x)
            @test RandomDataStreams.next(x) == w0
            srand!(x, fill(UInt64(0xbeef), n))
            @test get_state(x) == fill(UInt64(0xbeef), n)

            io = IOBuffer()
            show(io, x)
            @test occursin(shown_name, String(take!(io)))

            gen = G(fill(UInt64(0xfeed), n))
            r1 = next_stream!(gen)
            r2 = next_stream!(gen)
            s1 = Set([RandomDataStreams.next(r1) for _ in 1:200])
            s2 = Set([RandomDataStreams.next(r2) for _ in 1:200])
            @test isempty(intersect(s1, s2))
            @test get_state(gen) != fill(UInt64(0xfeed), n)
        end

        @test_throws ArgumentError Xoroshiro128pGen(fill(UInt64(1), 3))
        @test_throws ArgumentError Xoshiro512pGen(fill(UInt64(1), 4))
    end

    @testset "advance_state! (xoshiro families)" begin
        seed256 = fill(UInt64(0x31), 4)

        # fixed-distance jumps agree with short_jump! from a fresh generator
        for (T, e) in ((Xoroshiro128p, 64), (Xoshiro256ss, 128), (Xoshiro512p, 256))
            n = statewords(T)
            seed = fill(UInt64(0xabcdef), n)
            a = T(seed); b = T(seed)
            short_jump!(a)
            advance_state!(b, e, 0)
            @test a.Cg == b.Cg
        end

        # small forward jumps match manual stepping
        for k in (1, 2, 3, 17, 1000)
            y = Xoshiro256p(seed256)
            advance_state!(y, 0, k)
            st = RandomDataStreams._lin_step(Xoshiro256p(seed256).Cg)
            for _ in 2:k
                st = RandomDataStreams._lin_step(st)
            end
            @test y.Cg == st
        end

        # round trip: +k then -k restores the exact position
        for T in (Xoroshiro128pp, Xoshiro256p, Xoshiro512ss)
            x = T(fill(UInt64(7), statewords(T)))
            foreach(_ -> RandomDataStreams.next(x), 1:13)
            orig = x.Cg
            advance_state!(x, 0, 12345)
            @test x.Cg != orig
            advance_state!(x, 0, -12345)
            @test x.Cg == orig
        end

        # backward jump then re-draw reproduces earlier values
        x = Xoshiro512pp(fill(UInt64(9), 8))
        v1 = RandomDataStreams.next(x)
        RandomDataStreams.next(x)
        advance_state!(x, 0, -2)
        @test RandomDataStreams.next(x) == v1

        # distance convention n = 2^e + c (e > 0), -2^-e + c (e < 0), c (e = 0)
        x = Xoshiro256p(seed256); advance_state!(x, 0, 7); advance_state!(x, -1, -1)   # -3
        z = Xoshiro256p(seed256); advance_state!(z, 0, 4)
        @test x.Cg == z.Cg

        a = Xoshiro256p(seed256); advance_state!(a, -2, 5)                             # +1
        @test a.Cg == RandomDataStreams._lin_step(Xoshiro256p(seed256).Cg)

        # distances beyond the period are reduced modulo the period
        x = Xoshiro256p(fill(UInt64(5), 4)); advance_state!(x, 1000, 0)
        y = Xoshiro256p(fill(UInt64(5), 4)); advance_state!(y, 232, 0)
        @test x.Cg == y.Cg

        # boundaries are not moved by arbitrary jumps
        r = Xoshiro256p(fill(UInt64(11), 4))
        bg_before = r.Bg; ig_before = r.Ig
        advance_state!(r, 10, 3)
        @test r.Bg == bg_before && r.Ig == ig_before
    end

    @testset "Random API parity" begin
        # drop-in substitutability with the standard library's Xoshiro:
        # every operation below must work for every RandomDataStreams generator.
        mk = [
            () -> MRG32k3a([42, 1, 2, 3, 4, 5]),
            () -> MRG63k3a([42, 1, 2, 3, 4, 5]),
            () -> Xoshiro256pp(fill(UInt64(42), 4)),
            () -> Xoroshiro128ss(fill(UInt64(42), 2)),
            () -> Xoshiro512p(fill(UInt64(42), 8)),
            () -> next_stream!(PhiloxGen()),
            () -> next_stream!(Philox4x64Gen()),
            () -> next_stream!(Threefry4x64Gen()),
            () -> next_stream!(Threefry4x32Gen()),
        ]
        for mkm in mk
            r = mkm()
            @test rand(r) isa Float64
            @test rand(r, Bool) isa Bool
            for T in (Int, UInt8, UInt64, Int128, Char)
                v = rand(r, T)
                @test v isa T
            end
            @test all(1 .<= rand(r, 1:10) .<= 10)
            v = rand(r, 5); @test length(v) == 5
            @test size(rand(r, Int, 2, 2)) == (2, 2)
            d = Vector{Float64}(undef, 4); rand!(r, d); @test all(0 .<= d .< 1)
            n = randn(r); @test n isa Float64
            e = randexp(r); @test e >= 0
            @test sort(shuffle(r, collect(1:8))) == collect(1:8)
            p = randperm(r, 6); @test sort(p) == collect(1:6)
            @test Random.randstring(r, 5) |> length == 5
        end

        # seed! reproducibility: same seed -> identical sequence
        for (mka, seed) in (
            (() -> MRG32k3a(), 12345),
            (() -> MRG63k3a(), 12345),
            (() -> Xoshiro256pp(), 6789),      # the state is replaced by seed! below
            (() -> PhiloxRNG(), 6789),
        )
            r1 = mka(); r2 = mka()
            Random.seed!(r1, seed); Random.seed!(r2, seed)
            a1 = [rand(r1) for _ in 1:20]
            a2 = [rand(r2) for _ in 1:20]
            @test a1 == a2
        end

        # seed! accepts vectors and validates them
        r = MRG32k3a()
        Random.seed!(r, [7, 7, 7, 8, 8, 8]); @test r.Cg == [7, 7, 7, 8, 8, 8]
        @test_throws ArgumentError Random.seed!(r, [0, 0, 0, 1, 1, 1])
        r63 = MRG63k3a()
        Random.seed!(r63, [7, 7, 7, 8, 8, 8]); @test get_state(r63) == [7, 7, 7, 8, 8, 8]
        @test_throws ArgumentError Random.seed!(r63, [0, 0, 0, 1, 1, 1])
        x = Xoshiro256p()
        Random.seed!(x, UInt64[1, 2, 3, 4]); @test get_state(x) == UInt64[1, 2, 3, 4]
        @test_throws ArgumentError Random.seed!(x, fill(UInt64(0), 4))

        # streams still work through the generic API
        gen = Xoshiro256plusGen(UInt64[1, 2, 3, 4])
        s1 = next_stream!(gen); s2 = next_stream!(gen)
        @test !isempty(intersect([rand(s1) for _ in 1:100], [rand(s2) for _ in 1:100])) ==
              false
    end

    @testset "stream API matrix across all variants" begin
        # every stream/substream routine must work, with correct semantics,
        # for every xoshiro/xoroshiro variant.
        statewords(::Type{RandomDataStreams.LinRNG{N,S}}) where {N,S} = N

        variants = [
            (RandomDataStreams.Xoroshiro128p,    RandomDataStreams.Xoroshiro128pGen),
            (RandomDataStreams.Xoroshiro128ss,   RandomDataStreams.Xoroshiro128ssGen),
            (RandomDataStreams.Xoroshiro128pp,   RandomDataStreams.Xoroshiro128ppGen),
            (Xoshiro256p,           Xoshiro256plusGen),
            (RandomDataStreams.Xoshiro256ss,     RandomDataStreams.Xoshiro256ssGen),
            (RandomDataStreams.Xoshiro256pp,     RandomDataStreams.Xoshiro256ppGen),
            (RandomDataStreams.Xoshiro512p,      RandomDataStreams.Xoshiro512pGen),
            (RandomDataStreams.Xoshiro512ss,     RandomDataStreams.Xoshiro512ssGen),
            (RandomDataStreams.Xoshiro512pp,     RandomDataStreams.Xoshiro512ppGen),
        ]
        for (T, G) in variants
            n = statewords(T)
            seed = fill(UInt64(0xabc), n)

            gen = G(seed)
            s1 = next_stream!(gen); s2 = next_stream!(gen)
            @test RandomDataStreams.get_state(s1) != RandomDataStreams.get_state(s2)

            x = T(seed); srand!(x, seed)
            @test get_state(x) == seed

            y = T(seed); z = T(seed)
            Random.seed!(y, 987); Random.seed!(z, 987)
            @test rand(y) == rand(z)

            a = T(seed); b = copy(a)
            RandomDataStreams.next(b); RandomDataStreams.next(b)
            @test a.Cg != b.Cg

            c = T(seed)
            u0 = RandomDataStreams.next(c)
            RandomDataStreams.next(c); RandomDataStreams.next(c)
            reset_stream!(c)
            @test RandomDataStreams.next(c) == u0

            d = T(seed)
            next_substream!(d)
            w1 = RandomDataStreams.next(d)
            reset_substream!(d)
            @test RandomDataStreams.next(d) == w1

            e1 = T(seed); e2 = T(seed)
            short_jump!(e1)
            next_substream!(e2)
            @test e1.Bg == e2.Bg && e1.Cg == e2.Cg

            f = T(seed)
            long_jump!(f)
            @test f.Bg == f.Cg && f.Ig == f.Cg

            g = T(seed)
            bg0, ig0 = g.Bg, g.Ig
            advance_state!(g, 0, 999)
            advance_state!(g, 0, -999)
            @test g.Cg == T(seed).Cg && g.Bg == bg0 && g.Ig == ig0
        end
    end

    @testset "deprecated names" begin
        seed = fill(UInt64(0x77), 4)
        a = Xoshiro256p(seed)
        @test_logs (:warn, r"deprecated") match_mode = :any begin
            b = short_jump(a)          # old name still works...
            @test b === a              # ...and mutates in place
        end
        @test_logs (:warn, r"deprecated") match_mode = :any begin
            srand(a, seed)
            @test get_state(a) == seed
        end
        gen = MRG32k3aGen()
        @test_logs (:warn, r"deprecated") match_mode = :any next_stream(gen) isa MRG32k3a
    end

end

include("readme.jl")

include("test_threads.jl")

include("test_testu01.jl")

include("test_streams_interleaved.jl")

include("test_bits.jl")
