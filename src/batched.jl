#=
TODO

- add forward!, backward!, viterbi
- add support for ApproximateHMM
- add unit tests

=#

# ==== Batched CHMMera.jl ==== #
get_chimera_probabilities(device::Function, queries::Vector{String}, references::Vector{String}, batchsize::Integer=200; bw::Bool = true, mutation_probabilities::Vector{Float64} = [0.0047, 0.01, 0.05, 0.1, 0.15, 0.2], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 0.02) =
    get_chimera_probabilities(device, as_ints.(queries), as_ints.(references), batchsize, bw, mutation_probabilities, base_mutation_probability, prior_probability)

function get_chimera_probabilities(device::Function, queries::Vector{Vector{UInt8}}, references::Vector{Vector{UInt8}}, batchsize::Integer, bw::Bool = true, mutation_probabilities::Vector{Float64} = [0.0047, 0.01, 0.05, 0.1, 0.15, 0.2], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 0.02)
    hmm = bw ? ApproximateHMM(vovtomatrix(references), prior_probability) : FullHMM(vovtomatrix(references), mutation_probabilities, prior_probability)
    mutation_probabilities = bw ? [base_mutation_probability for i in 1:length(references)] : mutation_probabilities
    chimera_probabiltiies = zeros(length(queries))
    # split query iteration among threads
    Threads.@threads for i in 1:batchsize:lastindex(queries)
        j = min(i + batchsize - 1, lastindex(queries))
        chimera_probabiltiies[i:j] .= chimeraprobability(device, queries[i:j], hmm, copy(mutation_probabilities))
    end
    return chimera_probabiltiies
end

# ==== Batched hmm.jl ==== # 
function batched_get_bs(hmm::FullHMM, O::AbstractMatrix, mutation_probabilities::AbstractVector)
    b = similar(O, Float32, hmm.n, hmm.K, size(O, 1), hmm.L)
    S = similar(O, UInt32, size(hmm.S)...)
    copy!(S, UInt32.(hmm.S))
    @inbounds for mut in 1:hmm.K # mutation rates
        same_obs_prob = Float32(1 - mutation_probabilities[mut])
        diff_obs_prob = Float32(mutation_probabilities[mut] / 3)
        for ref in 1:hmm.n # references
            b[ref, mut, :, :] .= get_b.(reshape(S[ref, :], 1, :), O, same_obs_prob, diff_obs_prob)
        end
    end
    return b
end

function batched_get_bs(hmm::ApproximateHMM, O::AbstractMatrix, mutation_probabilities::AbstractMatrix)
    b = similar(O, Float32, hmm.N, size(O, 1), hmm.L)
    S = similar(O, UInt32, size(hmm.S, 1), size(hmm.S, 2))
    copy!(S, UInt32.(hmm.S))
    S = reshape(S, :, 1, size(hmm.S, 2))
    same_obs_prob = reshape(1 .- mutation_probabilities, size(mutation_probabilities)..., 1)
    diff_obs_prob = reshape(mutation_probabilities ./ 3, size(mutation_probabilities)..., 1)

    b .= get_b.(S, reshape(O, 1, size(O)...), same_obs_prob, diff_obs_prob)
    return b
end

function get_b(hmm_obs::UInt32, obs::UInt32, same_obs_prob::Real, diff_obs_prob::Real)
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
function chimeraprobability(device::Function, O::Vector{Vector{UInt8}}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    O = device(UInt32.(vovtomatrix(O)))
    if hmm isa ApproximateHMM
        mutation_probabilities_matrix = device(repeat(mutation_probabilities, 1, size(O, 1)))
        b = batched_get_bs(hmm, O, mutation_probabilities_matrix)
        parameterestimation!(hmm, O, mutation_probabilities_matrix, b)
        b = hmm isa ApproximateHMM ? batched_get_bs(hmm, O, mutation_probabilities_matrix) : b
    else
        b = batched_get_bs(hmm, O, mutation_probabilities)
    end
    x = forward(hmm, b) |> Vector
    return x
end

# == FullHMM == #
function forward(hmm::FullHMM, b::AbstractArray{T}) where T <: Real
    a_self = T(1 - hmm.switch_probability - hmm.μ)
    a_diffref = T(hmm.switch_probability / ((hmm.n - 1) * hmm.K))
    a_diffmut = hmm.K == 1 ? zero(T) : T(hmm.μ / (hmm.K - 1))

    #b = reshape(b, size(b)[1:3]..., :)
    batchdim = size(b, 3)

    alpha = similar(b, 2, hmm.n, hmm.K, batchdim)

    alpha[1, :, :, :] .= T(initialstate(hmm)) .* b[:, :, :, 1]
    alpha[2, :, :, :] .= zero(T)

    postchimera_mask = similar(alpha, 2, 1, 1, 1)
    postchimera_mask[1, :, :, :] .= zero(T)
    postchimera_mask[2, :, :, :] .= a_diffref

    for t in 1:hmm.L - 1
        sumref = sum(alpha, dims=3)
        sumref_tot = sum(sumref, dims=1)
        sumalpha = sum(sumref_tot, dims=2)

        alpha .= (alpha .* a_self .+ (sumref .- alpha) .* a_diffmut .+ (sumalpha .- sumref_tot) .* postchimera_mask) .* reshape(b[:, :, :, t+1], 1, hmm.n, hmm.K, :)

        alpha ./= sum(alpha, dims=(1, 2, 3))
    end

    detec = sum(alpha[2, :, :, :], dims = (1, 2))
    return vec(detec ./ (detec .+ sum(alpha[1, :, :, :], dims = (1, 2))))
end 


# == ApproximateHMM == #

function forward(hmm::ApproximateHMM, b::AbstractArray{T}) where T <: Real
    a_false = T(a(false, hmm))
    a_true = T(a(true, hmm))

    batchdim = size(b, 2)

    alpha = similar(b, 2, hmm.N, batchdim)

    alpha[1, :, :] .= T(initialstate(hmm)) .* b[:, :, 1]
    alpha[2, :, :] .= zero(T)

    postchimera_mask = similar(alpha, 2, 1)
    postchimera_mask[1, :, :] .= zero(T)
    postchimera_mask[2, :, :] .= a_false

    for t in 1:hmm.L-1
        sumalpha = sum(alpha, dims = (1, 2))
        sumref = sum(alpha, dims=1)

        alpha .= (alpha .* a_true .+ (sumalpha .- sumref) .* postchimera_mask) .* reshape(b[:, :, t+1], 1, hmm.N, :)
        
        alpha ./= sum(alpha, dims=(1, 2, 3))
    end

    detec = sum(alpha[2, :, :], dims = 1)
    return vec(detec ./ (detec .+ sum(alpha[1, :, :], dims = 1)))
end

function forward!(alpha::AbstractArray, c::AbstractMatrix, hmm::ApproximateHMM, b::AbstractArray{T}) where T <: Real
    a_false = T(a(false, hmm))
    a_true = T(a(true, hmm))

    c[:, 1] .= one(T)

    alpha[:, :, 1] .= T(initialstate(hmm)) .* b[:, :, 1]

    for t in 1:hmm.L-1
        sumalpha = sum(alpha[:, :, t], dims = 1)

        alpha[:, :, t+1] .= (alpha[:, :, t] .* a_true .+ (sumalpha .- alpha[:, :, t]) .* a_false) .* b[:, :, t+1]
        
        c[:, t+1] .= one(T) ./ vec(sum(alpha[:, :, t+1], dims=1))
        alpha[:, :, t+1] .*= reshape(c[:, t + 1], 1, :)
    end
end

function backward!(beta::AbstractArray, c::AbstractMatrix, hmm::ApproximateHMM, b::AbstractArray{T}) where T <: Real
    a_false = T(a(false, hmm))
    a_true = T(a(true, hmm))

    beta[:, :, hmm.L] .= reshape(c[:, hmm.L], 1, :)
    for t in hmm.L-1:-1:1
        sumbeta = sum(beta[:, :, t+1] .* b[:, :, t+1], dims=1)   #, dims = 1)
        beta[:, :, t] .= a_false .* (sumbeta .- beta[:, :, t+1] .* b[:, :, t+1]) .+ a_true .* beta[:, :, t+1] .* b[:, :, t+1]
        beta[:, :, t] .*= reshape(c[:, t], 1, :)
    end
end

function parameterestimation!(hmm::ApproximateHMM, O::AbstractMatrix{U}, mutation_probabilities::AbstractMatrix, b::AbstractArray{T}) where {U <: Unsigned, T <: Real}
    batchsize = size(O, 1)
    alpha = similar(b, hmm.N, batchsize, hmm.L)
    beta = similar(b, hmm.N, batchsize, hmm.L)
    c = similar(b, batchsize, hmm.L)
    forward!(alpha, c, hmm, b)
    backward!(beta, c, hmm, b)
	Nmut = similar(b, hmm.N, batchsize)
    Nsame = similar(b, hmm.N, batchsize)

    # pseudocount prior
    Nmut .= T(2)
    Nsame .= T(10)

    C = alpha .* beta
    sumC = sum(C, dims = 1)

    S = similar(O, UInt32, size(hmm.S)...)
    copy!(S, UInt32.(hmm.S))

    for t in 1:hmm.L
        Ot = reshape(O[:, t], 1, :)
        St = reshape(S[:, t], :, 1)

        # Ot.!= 6 to handle non-informative obs
        Nmut .+= (Ot .!= U(6)) .* (Ot .!= St) .* alpha[:, :, t] .* beta[:, :, t] ./ sumC[:, :, t]
        Nsame .+= (Ot .!= U(6)) .* (Ot .== St) .* alpha[:, :, t] .* beta[:, :, t] ./ sumC[:, :, t]
    end
    mutation_probabilities .= Nmut ./ (Nmut .+ Nsame)
end