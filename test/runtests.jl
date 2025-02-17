using CHMMera
using Test
using Random: seed!
seed!(1)

include("naive_algorithms.jl")

@testset "CHMMera.jl" begin
    @testset "utils.jl" begin
        @test CHMMera.as_ints("ACGT-N") == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
        @test CHMMera.vovtomatrix([[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]]) == [0x01 0x02 0x03 0x04 0x05 0x06; 0x01 0x02 0x03 0x04 0x05 0x06]
    end

    @testset "hmm.jl" begin
        fullhmm = CHMMera.FullHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(["ACGT-N", "ACGT-N"])), [0.01, 0.02], 0.01)
        @test fullhmm.N == 4
        @test fullhmm.n == 2
        @test fullhmm.L == 6
        @test fullhmm.K == 2
        @test CHMMera.get_bs(fullhmm, CHMMera.as_ints("ACGT-N"), [0.01, 0.02]) == [0.99 0.99 0.99 0.99 1.0 1.0;
                                                                                0.98 0.98 0.98 0.98 1.0 1.0;
                                                                                0.99 0.99 0.99 0.99 1.0 1.0;
                                                                                0.98 0.98 0.98 0.98 1.0 1.0]

        approximatehmm = CHMMera.ApproximateHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(["ACGT-N", "ACGT-N"])), 0.01)
        @test approximatehmm.N == 2
        @test approximatehmm.L == 6
        @test approximatehmm.switch_probability == 0.01 / 6
        @test CHMMera.get_bs(approximatehmm, CHMMera.as_ints("ACGT-N"), [0.01, 0.02]) == [0.99 0.99 0.99 0.99 1.0 1.0;
                                                                                                0.98 0.98 0.98 0.98 1.0 1.0]
    end

    @testset "algorithms.jl" begin
        # Compare forward vs a naive implementation of forward
        n_tests = 100
        seq_length = 9
        num_refs = 4
        tests = []
        for i in 1:n_tests
            references = [join(rand(collect("ACGT"), seq_length)) for _ in 1:num_refs]
            query = join(rand(collect("ACGT"), seq_length))
            # Test O(N) vs O(N^2)
            full_hmm = CHMMera.FullHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(references)), [0.1, 0.2, 0.3], 1 / 300, 0.01)
            full_hmm_bs = CHMMera.get_bs(full_hmm, CHMMera.as_ints(query), [0.1, 0.2, 0.3])
            push!(tests, isapprox(CHMMera.forward(full_hmm, full_hmm_bs), naive_forward(full_hmm, full_hmm_bs), atol=1e-6))

            approx_hmm = CHMMera.ApproximateHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(references)), 1 / 300)
            approx_hmm_bs = CHMMera.get_bs(approx_hmm, CHMMera.as_ints(query), fill(0.05, length(query)))
            push!(tests, isapprox(CHMMera.forward(approx_hmm, approx_hmm_bs), naive_forward(approx_hmm, approx_hmm_bs), atol=1e-6))
        end
        @test all(tests)
    end

    @testset "CHMMera.jl" begin
        # Test chimera probability calculations
        # Non-chimeric sequence (AAAAAA) should have very low probability
        # Chimeric sequence (CCCAAA) should have very high probability

        references = ["AAAAAA", "CCCCCC"]
        queries = ["AAAAAA", "CCCAAA"]
        # Baum-Welch
        chimera_probs = CHMMera.get_chimera_probabilities(queries, references, bw = true)
        @test chimera_probs[1] < 0.1
        @test chimera_probs[2] > 0.9

        # Discrete Bayesian
        chimera_probs = CHMMera.get_chimera_probabilities(queries, references, bw = false)
        @test chimera_probs[1] < 0.1
        @test chimera_probs[2] > 0.9

        # Test recombination event calculations
        # Non-recombinant sequence (AAAAAA) should have no recombination events
        # Recombinant sequence (CCCAAA) should have recombination events
        # We should get the same results for Baum-Welch and Discrete Bayesian
        # Baum-Welch
        recombination_events = CHMMera.get_recombination_events(queries, references, bw = true, detailed = true)
        @test recombination_events[1].recombinations == []
        @test recombination_events[1].startingpoint == 1
        @test recombination_events[2].recombinations == [RecombinationEvent(4, 2, 1, 2, 1)]
        @test recombination_events[2].startingpoint == 2
        @test recombination_events[2].pathevaluation > 0.9

        # run without path evaluation/starting point
        recombination_events = CHMMera.get_recombination_events(queries, references, bw = true, detailed = false)
        @test recombination_events[1].recombinations == []
        @test recombination_events[1].startingpoint == 1
        @test recombination_events[2].recombinations == [RecombinationEvent(4, 2, 1, 2, 1)]

        # Discrete Bayesian
        recombination_events = CHMMera.get_recombination_events(queries, references, bw = false, mutation_probabilities = [0.01, 0.05, 0.1], detailed = true)
        @test recombination_events[1].recombinations == []
        @test recombination_events[1].startingpoint == 1
        @test recombination_events[2].recombinations == [RecombinationEvent(4, 2, 1, 4, 1)]
        @test recombination_events[2].startingpoint == 2
        @test recombination_events[2].pathevaluation > 0.9

        # k = 1
        recombination_events = CHMMera.get_recombination_events(queries, references, bw = false, mutation_probabilities = [0.005], detailed = true)
        @test recombination_events[1].recombinations == []
        @test recombination_events[1].startingpoint == 1
        @test recombination_events[2].recombinations == [RecombinationEvent(4, 2, 1, 2, 1)]
        @test recombination_events[2].startingpoint == 2
        @test recombination_events[2].pathevaluation > 0.9


        # Test O(N) vs O(N^2)
        full_hmm = CHMMera.FullHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(references)), [0.1, 0.2, 0.3], 1 / 300, 0.01)
        full_hmm_bs = CHMMera.get_bs(full_hmm, CHMMera.as_ints(queries[2]), [0.1, 0.2, 0.3])
        @test CHMMera.forward(full_hmm, full_hmm_bs) ≈ naive_forward(full_hmm, full_hmm_bs) atol=1e-5

        approx_hmm = CHMMera.ApproximateHMM(CHMMera.vovtomatrix(CHMMera.as_ints.(references)), 1 / 300)
        approx_hmm_bs = CHMMera.get_bs(approx_hmm, CHMMera.as_ints(queries[2]), fill(0.05, length(queries[2])))
        @test CHMMera.forward(approx_hmm, approx_hmm_bs) ≈ naive_forward(approx_hmm, approx_hmm_bs) atol=1e-5
    end
end
