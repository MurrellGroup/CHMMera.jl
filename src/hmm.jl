# HMM structs

abstract type HMM end

struct ApproximateHMM <: HMM
    N::Int64 # Number of states (equal to number of reference sequences)
    L::Int64 # Length of reference sequences
    S::Matrix{Int64} # Reference sequences
    pmut::Vector{Float64} # Mutation rate for each reference sequence
    pswitch::Float64 # Probability of switching to a different reference sequence
end

function ApproximateHMM(refs::Matrix{Int64}, base_pmut::Float64, pchimera_prior::Float64)
    N = length(refs[:, 1])
    L = length(refs[1, :])
    S = refs
    pmut = [base_pmut for i in 1:N]
    pswitch = pchimera_prior / L
    return ApproximateHMM(N, L, S, pmut, pswitch)
end

struct FullHMM <: HMM
    N::Int64 # Number of states
    n::Int64 # Number of reference sequences
    L::Int64 # Length of reference sequences
    K::Int64 # Number of mutation categories
    S::Matrix{Float64} # Reference sequences
    mcat::Vector{Float64} # Mutation categories
    pswitch::Float64 # Probability of switching to a different reference sequence
end

function FullHMM(refs::Matrix{Int64}, mcat::Vector{Float64}, pchimera_prior::Float64)
    K = length(mcat)
    n = length(refs[:, 1])
    N = n * K
    L = length(refs[1, :])
    S = refs
    pswitch = pchimera_prior / L
    return FullHMM(N, n, L, K, S, mcat, pswitch)
end

initialstate(hmm::HMM) = 1 / hmm.N

a(samestate::Bool, hmm::HMM) = samestate ? 1 - hmm.pswitch : hmm.pswitch / (hmm.N - 1)

function b(i::Int64, t::Int64, O::Vector{Int64}, hmm::ApproximateHMM)
    if O[t] == 6
        return 1.0
    else
        #BM: Changed denom to 5 to avoid "cheating" with gaps.
        #We might want to have a separate "indel" rate though.
        return O[t] == hmm.S[i, t] ? 1-hmm.pmut[i] : hmm.pmut[i]/5 
    end
end

function b(i::Int64, m::Int64, t::Int64, O::Vector{Int64}, hmm::FullHMM)
    if O[t] == 6
        return 1.0
    else
        return O[t] == hmm.S[i, t] ? 1-hmm.mcat[m] : hmm.mcat[m]/5
    end
end
