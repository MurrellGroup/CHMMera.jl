# HMM structs

abstract type HMM end

struct ApproximateHMM <: HMM
    N::Int64 # Number of states (equal to number of reference sequences)
    L::Int64 # Length of reference sequences
    S::Matrix{Int64} # Reference sequences
    mutation_probabilities::Vector{Float64} # Mutation rate for each reference sequence - each ref has it's own mutation rate.
    switch_probability::Float64 # Probability of switching to a different reference sequence
end

function ApproximateHMM(references::Matrix{Int64}, base_mutation_probability::Float64, chimera_prior_probability::Float64)
    N = length(references[:, 1])
    L = length(references[1, :])
    S = references
    mutation_probabilities = [base_mutation_probability for i in 1:N]
    switch_probability = chimera_prior_probability / L
    return ApproximateHMM(N, L, S, mutation_probabilities, switch_probability)
end

struct FullHMM <: HMM
    N::Int64 # Number of states
    n::Int64 # Number of reference sequences
    L::Int64 # Length of reference sequences
    K::Int64 # Number of mutation rates
    S::Matrix{Float64} # Reference sequences
    mutation_probabilities::Vector{Float64} # Each reference has K copies, each with a different mutation rate.
    switch_probability::Float64 # Probability of switching to a different reference sequence at each site.
end

function FullHMM(references::Matrix{Int64}, mutation_probabilities::Vector{Float64}, chimera_prior_probability::Float64)
    K = length(mutation_probabilities)
    n = length(references[:, 1])
    N = n * K
    L = length(references[1, :])
    S = references
    switch_probability = chimera_prior_probability / L
    return FullHMM(N, n, L, K, S, mutation_probabilities, switch_probability)
end

initialstate(hmm::HMM) = 1 / hmm.N

a(samestate::Bool, hmm::HMM) = samestate ? 1 - hmm.switch_probability : hmm.switch_probability / (hmm.N - 1)

function b(i::Int64, t::Int64, O::Vector{Int64}, hmm::ApproximateHMM)
    if O[t] == 6
        return 1.0
    else
        #BM: Changed denom to 5 to avoid "cheating" with gaps.
        #We might want to have a separate "indel" rate though.
        return O[t] == hmm.S[i, t] ? 1-hmm.mutation_probabilities[i] : hmm.mutation_probabilities[i]/5 
    end
end

function b(i::Int64, m::Int64, t::Int64, O::Vector{Int64}, hmm::FullHMM)
    if O[t] == 6
        return 1.0
    else
        return O[t] == hmm.S[i, t] ? 1-hmm.mutation_probabilities[m] : hmm.mutation_probabilities[m]/5
    end
end
