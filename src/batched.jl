#=
TODO

- add forward!, backward!, viterbi
- add support for ApproximateHMM
- add unit tests

=#

# ==== Batched CHMMera.jl ==== #
get_chimera_probabilities(device::Function, queries::Vector{String}, references::Vector{String}, batchsize::Integer=100; bw::Bool = true, mutation_probabilities::Vector{Float64} = [0.0047, 0.01, 0.05, 0.1, 0.15, 0.2], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 0.02) =
    get_chimera_probabilities(device, as_ints.(queries), as_ints.(references), batchsize, bw, mutation_probabilities, base_mutation_probability, prior_probability)

function get_chimera_probabilities(device::Function, queries::Vector{Vector{UInt8}}, references::Vector{Vector{UInt8}}, batchsize::Integer, bw::Bool = true, mutation_probabilities::Vector{Float64} = [0.0047, 0.01, 0.05, 0.1, 0.15, 0.2], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 0.02)
    hmm = bw ? ApproximateHMM(vovtomatrix(references), prior_probability) : FullHMM(vovtomatrix(references), mutation_probabilities, prior_probability)
    mutation_probabilities = bw ? [base_mutation_probability for i in 1:length(references)] : mutation_probabilities
    chimera_probabiltiies = zeros(length(queries))
    # split query iteration among threads
    i = firstindex(queries)
    while i <= lastindex(queries)
        j = min(i + batchsize - 1, lastindex(queries))
        chimera_probabiltiies[i:j] .= chimeraprobability(device, queries[i:j], hmm, copy(mutation_probabilities))
        i = j + 1
    end
    return chimera_probabiltiies
end

# ==== Batched hmm.jl ==== # 
function batched_get_bs(hmm::FullHMM, O::AbstractMatrix, mutation_probabilities::AbstractVector)
    b = similar(O, Float32, hmm.n * hmm.K, hmm.L, size(O, 2))
    ref2stateindices = stateindicesofref.(1:hmm.n, (hmm,))
    @inbounds for i in 1:hmm.K # mutation rates
        same_obs_prob = 1 .- mutation_probabilities[i]
        diff_obs_prob = mutation_probabilities[i] ./ 3
        for j in 1:hmm.n # references
            ind = ref2stateindices[j][i]
            for k in 1:hmm.L # timepoints
                b[ind, k, :] .= get_b.(UInt32(hmm.S[j, k]), O[k, :], Float32(same_obs_prob), Float32(diff_obs_prob))
            end
        end
    end
    return b
end

function get_b(hmm_obs::UInt32, obs::UInt32, same_obs_prob::Float32, diff_obs_prob::Float32)
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

# ==== Batched algorithms.jl ==== #
function chimeraprobability(device::Function, O::Vector{Vector{UInt8}}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: FullHMM
    O = device(UInt32.(cat(O..., dims=2)))
    b = batched_get_bs(hmm, O, mutation_probabilities)
    #T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities, b)
    #b = T == ApproximateHMM ? get_bs(hmm, O, mutation_probabilities) : b
    return forward(hmm, b) |> Vector
end

function forward(hmm::HMM, b::AbstractArray{T}) where T <: Real
    a_self = T(1 - hmm.switch_probability - hmm.μ)
    a_diffref = T(hmm.switch_probability / ((hmm.n - 1) * hmm.K))
    a_diffmut = hmm isa ApproximateHMM || hmm.K == 1 ? zero(T) : T(hmm.μ / (hmm.K - 1))

    b = reshape(b, size(b, 1), size(b, 2), :)
    batchdim = size(b, 3)

    alpha = similar(b, 2, hmm.N, batchdim)

    alpha[1, :, :] .= T(initialstate(hmm)) .* b[:, 1, :]
    alpha[2, :, :] .= zero(T)

    postchimera_mask = similar(alpha, 2, 1, 1)
    postchimera_mask[1, :, :] .= zero(T)
    postchimera_mask[2, :, :] .= a_diffref

    for t in 1:hmm.L - 1
        sumalpha = sum(alpha, dims = (1, 2))

        for ref in 1:hmm.n
            states = stateindicesofref(ref, hmm)
            sumstates = sum(alpha[:, states, :], dims = 2)
            sumstates_tot = sum(alpha[:, states, :], dims = (1, 2))

            alpha[:, states, :] .= (alpha[:, states, :] .* a_self .+ (sumstates .- alpha[:, states, :]) .* a_diffmut .+ (sumalpha .- sumstates_tot) .* postchimera_mask) .* reshape(b[states, t+1, :], 1, :, batchdim)
        end

        alpha ./= sum(alpha, dims=(1, 2))
    end

    detec = sum(alpha[2, :, :], dims = 1)
    return reshape(detec ./ (detec .+ sum(alpha[1, :, :], dims = 1)), :)
end 

#stateindicesofref_bounds(ref_idx::Integer, hmm::FullHMM) = (1 + (ref_idx-1) * hmm.K, ref_idx * hmm.K)
#stateindicesofref_bounds(ref_idx::Integer, hmm::ApproximateHMM) = (ref_idx, ref_idx)