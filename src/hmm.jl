# HMM structs

abstract type HMM end

struct ApproximateHMM <: HMM
    N::Int64 # Number of states (equal to number of reference sequences)
    L::Int64 # Length of reference sequences
    S::Matrix{UInt8} # Reference sequences
    switch_probability::Float64 # Probability of switching to a different reference sequence
end

function ApproximateHMM(references::Matrix{UInt8}, chimera_prior_probability::Float64)
    N = length(references[:, 1])
    L = length(references[1, :])
    S = references
    switch_probability = chimera_prior_probability / L
    return ApproximateHMM(N, L, S, switch_probability)
end

struct FullHMM <: HMM
    N::Int64 # Number of states
    n::Int64 # Number of reference sequences
    L::Int64 # Length of reference sequences
    K::Int64 # Number of mutation rates
    S::Matrix{UInt8} # Reference sequences
    switch_probability::Float64 # Probability of switching to a different reference sequence at each site.
end

function FullHMM(references::Matrix{UInt8}, mutation_probabilities::Vector{Float64}, chimera_prior_probability::Float64)
    K = length(mutation_probabilities)
    n = length(references[:, 1])
    N = n * K
    L = length(references[1, :])
    S = references
    switch_probability = chimera_prior_probability / L
    return FullHMM(N, n, L, K, S, switch_probability)
end

ref_index(state_index::Int64, hmm::FullHMM) = div(state_index - 1, hmm.K) + 1
ref_index(state_index::Int64, hmm::ApproximateHMM) = state_index

stateindicesofref(ref_idx::Int64, hmm::FullHMM) = (1 + (ref_idx-1) * hmm.K):(ref_idx * hmm.K)
stateindicesofref(ref_idx::Int64, hmm::ApproximateHMM) = ref_idx:ref_idx

initialstate(hmm::HMM) = 1 / hmm.N

# transition_probability
a(samestate::Bool, hmm::HMM) = samestate ? 1 - hmm.switch_probability : hmm.switch_probability / (hmm.N - 1)

function get_b(hmm_obs::UInt8, obs::UInt8, same_obs_prob::Float64, diff_obs_prob::Float64)
    if (obs == 0x05) | (obs == 0x06)
        return 1.0
    else
        if hmm_obs == obs
            return same_obs_prob
        else
            return diff_obs_prob
        end
    end
end

# calculate all observation probabilities for an observation vector
# uses more memory but a bit less time than a function
function get_bs(hmm::ApproximateHMM, O::Vector{UInt8}, mutation_probabilities::Vector{Float64})
    b = Matrix{Float64}(undef, hmm.N, hmm.L)
    @inbounds for i in 1:hmm.N # states (reference sequences)
        for j in 1:hmm.L # timepoints
            b[i, j] = get_b(hmm.S[i, j], O[j], 1 - mutation_probabilities[i], mutation_probabilities[i] / 3)
        end
    end
    return b
end

function get_bs(hmm::FullHMM, O::Vector{UInt8}, mutation_probabilities::Vector{Float64})
    b = Matrix{Float64}(undef, hmm.n * hmm.K, hmm.L)
    ref2stateindices = stateindicesofref.(1:hmm.n, Ref(hmm))
    @inbounds for i in 1:hmm.K # mutation rates
        same_obs_prob = 1 - mutation_probabilities[i]
        diff_obs_prob = mutation_probabilities[i] / 3
        for j in 1:hmm.n # references
            ind = ref2stateindices[j][i]
            for k in 1:hmm.L # timepoints
                b[ind, k] = get_b(hmm.S[j, k], O[k], same_obs_prob, diff_obs_prob)
            end
        end
    end
    return b
end