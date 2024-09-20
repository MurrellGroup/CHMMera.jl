# HMM structs

abstract type HMM end

struct ApproximateHMM <: HMM
    N::Int64 # Number of states (equal to number of reference sequences)
    L::Int64 # Length of reference sequences
    S::Matrix{UInt8} # Reference sequences
    switch_probability::Float64 # Probability of switching to a different reference sequence
end

function ApproximateHMM(references::Matrix{UInt8}, base_mutation_probability::Float64, chimera_prior_probability::Float64)
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
    S::Matrix{Float64} # Reference sequences
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

# calculate all observation probabilities for an observation vector
# uses more memory but a bit less time than a function
function get_bs(hmm::ApproximateHMM, O::Vector{UInt8}, mutation_probabilities::Vector{Float64})
    b = Matrix{Float64}(undef, hmm.N, hmm.L)
    @inbounds for i in 1:hmm.N
        same_obs_prob = 1 - mutation_probabilities[i]
        diff_obs_prob = mutation_probabilities[i] / 5
        for j in 1:hmm.L
            hmm_obs = hmm.S[i, j]
            if O[j] == 6
                prob = 1
            elseif hmm_obs == O[j]
                prob = same_obs_prob
            else
                prob = diff_obs_prob
            end
            b[i, j] = prob
        end
    end
    return b
end

function get_bs(hmm::FullHMM, O::Vector{UInt8}, mutation_probabilities::Vector{Float64})
    b = Matrix{Float64}(undef, hmm.n * hmm.K, hmm.L)
    @inbounds for i in 1:hmm.K # mutation rates
        same_obs_prob = 1 - mutation_probabilities[i]
        diff_obs_prob = mutation_probabilities[i] / 5
        for j in 1:hmm.n # references
            ind = (i - 1) * hmm.n + j
            for k in 1:hmm.L # timepoints
                hmm_obs = hmm.S[j, k]
                obs = O[k]
                b[ind, k] = if obs == 6
                    1.0
                elseif hmm_obs == obs
                    same_obs_prob
                else
                    diff_obs_prob
                end
            end
        end
    end
    return b
end