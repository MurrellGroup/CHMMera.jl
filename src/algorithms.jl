
struct RecombinationEvent
    position::Int64
    # reference indices
    left::Int64
    right::Int64
    # state indices
    left_state::Int64
    right_state::Int64
end

struct RecombinationEvents
    recombinations::Vector{RecombinationEvent}
    startingpoint::Int64
end

struct DetailedRecombinationEvents
    recombinations::Vector{RecombinationEvent}
    startingpoint::Int64
    pathevaluation::Float64
    logsiteprobabilities::Vector{Float64}
end

Base.show(io::IO, recomb::RecombinationEvents) = print(io, "(recombinations = $(recomb.recombinations), startingpoint = $(recomb.startingpoint))")

Base.show(io::IO, recomb::DetailedRecombinationEvents) = print(io, "(recombinations = $(recomb.recombinations), startingpoint = $(recomb.startingpoint), pathevaluation = $(recomb.pathevaluation), logsiteprobabilities = $(recomb.logsiteprobabilities))")

Base.show(io::IO, recomb::RecombinationEvent) = print(io, "(position = $(recomb.position), left = $(recomb.left), right = $(recomb.right), left_state = $(recomb.left_state), right_state = $(recomb.right_state))")

function chimeraprobability(O::Vector{UInt8}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    b = get_bs(hmm, O, mutation_probabilities)
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities, b)
    b = T == ApproximateHMM ? get_bs(hmm, O, mutation_probabilities) : b
    return forward(hmm, b)
end

function findrecombinations(O::Vector{UInt8}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    b = get_bs(hmm, O, mutation_probabilities)
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities, b)
    b = T == ApproximateHMM ? get_bs(hmm, O, mutation_probabilities) : b
    return viterbi(hmm, b)
end

function findrecombinations_detailed(O::Vector{UInt8}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    b = get_bs(hmm, O, mutation_probabilities)
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities, b)
    b = T == ApproximateHMM ? get_bs(hmm, O, mutation_probabilities) : b
    recombs = viterbi(hmm, b)
    if length(recombs.recombinations) == 0
        return DetailedRecombinationEvents(recombs.recombinations, recombs.startingpoint, -Inf, [])
    end

    # scaling constants
    c = Vector{Float64}(undef, hmm.L)
    # forward
    alpha = Matrix{Float64}(undef, hmm.N, hmm.L)
    forward!(alpha, c, hmm, b)
    # backward
    beta = Matrix{Float64}(undef, hmm.N, hmm.L)
    backward!(beta, c, hmm, b)

    # Normalized log probability of being at each position, (state_index, site_index)
    logp_position = Matrix{Float64}(undef, hmm.N, hmm.L)
    for t in 1:hmm.L
        p_position = alpha[:, t] .* beta[:, t] # unnormalized
        logp_position[:, t] = log.(p_position) .- log(sum(p_position)) # log normalized
    end

    # p_ref[i] = the probability of being at ref[i] at the time t, for the t that maximizes this probability, where t ∈ {t : viterbi_path[t] == ref[i]}
    p_ref = Dict(union([recombs.startingpoint => 0.0], [recomb.right => 0.0 for recomb in recombs.recombinations]))

    cur = recombs.startingpoint
    recombindex = 1
    for t in 1:hmm.L
        # For the fullbayesian version, each reference has multiple states, hence why we use stateindicesofref
        # exp(logp_position[i, t]) should not underflow here, since we are iterating over the path with the highest (log)probability
        p_ref[cur] = max(sum( exp(logp_position[i, t]) for i in stateindicesofref(cur, hmm) ), p_ref[cur])
        if recombindex <= length(recombs.recombinations) && t == recombs.recombinations[recombindex].position
            cur = recombs.recombinations[recombindex].right
            recombindex += 1
        end
    end

    probability_of_2nd_most_probable_ref = sort(collect(values(p_ref)))[end - 1]
    logsiteprobabilities_ = logsiteprobabilities(recombs, O, hmm, b)
    return DetailedRecombinationEvents(recombs.recombinations, recombs.startingpoint, probability_of_2nd_most_probable_ref, logsiteprobabilities_)
end


function logsiteprobabilities(recombs::RecombinationEvents,
                            O::Vector{UInt8},
                            hmm::T,
                            b::Matrix{Float64}) where T <: HMM
    # length(recombs) > 0 || "No recombinations found, can only be run on chimeric sequences"
    if length(recombs.recombinations) == 0
        return zeros(Float64, length(O))
    end

    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, hmm, b)
    backward!(beta, c, hmm, b)
    sort!(recombs.recombinations, by = x -> x.position)
    log_probability = Array{Float64}(undef, hmm.L)
    i = 1
    cur = recombs.recombinations[i].left_state
    for t in 1:hmm.L
        log_probability[t] = log(alpha[cur, t]) + log(beta[cur, t]) - log(sum(alpha[:, t] .* beta[:, t]))
        if t < hmm.L && i <= length(recombs.recombinations) && recombs.recombinations[i].position == t
            cur = recombs.recombinations[i].right_state
            i += 1
        end
    end
    return log_probability
end

# parameter estimation with the mutation_probabilities factored out of the hmm object to allow multithreading
function parameterestimation!(hmm::ApproximateHMM, O::Vector{UInt8}, mutation_probabilities::Vector{Float64}, b::Matrix{Float64})
    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, hmm, b)
    backward!(beta, c, hmm, b)
	Nmut = fill(2.0, hmm.N) # pseudocount prior
    Nsame = fill(10.0, hmm.N) # pseudocount prior

    n = size(alpha)[2]
    C = similar(alpha)
    @inbounds @simd for j in 1:n
        @views C[:, j] .= alpha[:, j] .* beta[:, j]
    end
    Zs = sum(C, dims = 1)

    for t in 1:hmm.L
        Z = Zs[t]
        for i in 1:hmm.N
            if O[t] != 6 #BM change, to handle non-informative obs
                if O[t] != hmm.S[i, t]
                    Nmut[i] += alpha[i, t] * beta[i, t] / Z
                else
                    Nsame[i] += alpha[i, t] * beta[i, t] / Z
                end
            end
        end
    end
    mutation_probabilities .= Nmut ./ (Nmut .+ Nsame)
end

function forward(hmm::HMM, b::Matrix{Float64})
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    a_false = a(false, hmm)
    a_true = a(true, hmm)
    init_state = initialstate(hmm)

    @inbounds for i in 1:hmm.N
        alpha[1, i] = init_state * b[i, 1]
        alpha[2, i] = 0.0
    end

    @inbounds for t in 1:hmm.L-1
        sumalpha = 0.0
        @simd for i in 1:hmm.N
            sumalpha += alpha[1, i] + alpha[2, i]
        end

        scaling_constant = 0.0
        @simd for j in 1:hmm.N
            bval = b[j, t+1]
            alpha_sum = alpha[1, j] + alpha[2, j]
            alpha[2, j] = ((sumalpha - alpha_sum) * a_false + alpha[2, j] * a_true) * bval
            alpha[1, j] = alpha[1, j] * a_true * bval
            scaling_constant += alpha[1, j] + alpha[2, j]
        end

        scaling_constant = 1.0 / scaling_constant
        @simd for j in 1:hmm.N
            alpha[1, j] *= scaling_constant
            alpha[2, j] *= scaling_constant
        end
    end

    detec = sum(view(alpha, 2, :))
    return detec / (sum(view(alpha, 1, :)) + detec)
end

function forward!(alpha::Matrix{Float64}, c::Vector{Float64}, hmm::HMM, b::Matrix{Float64})
    a_false = a(false, hmm)
    a_true = a(true, hmm)

    c[1] = 1
    @inbounds for i in 1:hmm.N
        alpha[i, 1] = initialstate(hmm) * b[i, 1]
    end
    @inbounds for t in 1:hmm.L-1
        sumalpha = 0.0
        @simd for i in 1:hmm.N
            sumalpha += alpha[i,t]
        end

        @simd for j in 1:hmm.N
            alpha[j, t + 1] = ((sumalpha - alpha[j, t])* a_false + alpha[j, t] * a_true) * b[j, t + 1]
        end

        c[t+1] = 1 / sum(view(alpha, :, t+1))
        alpha[:,t+1] .*= c[t+1]
    end
end

function backward!(beta::Matrix{Float64}, c::Vector{Float64}, hmm::HMM, b::Matrix{Float64})
    a_false = a(false, hmm)
    a_true = a(true, hmm)

    beta[:, hmm.L] .= c[hmm.L]
    @inbounds for t in hmm.L-1:-1:1
        sumbeta = 0.0
        @simd for j in 1:hmm.N
            sumbeta += beta[j, t+1]*b[j, t+1]
        end
        @simd for i in 1:hmm.N
            beta[i, t] = a_false*(sumbeta - beta[i, t+1]*b[i, t+1]) + a_true*beta[i, t+1]*b[i, t+1]
            beta[i, t] *= c[t]
        end
    end
end

function viterbi(hmm::HMM, b::Matrix{Float64})
    log_b = log.(b)
    phi = Array{Float64}(undef, hmm.N)
    from = Array{Int64}(undef, hmm.N, hmm.L)
    for i in 1:hmm.N
        phi[i] = log(initialstate(hmm) * b[i, 1])
        from[i, 1] = i
    end

    log_a_true = log(a(true, hmm))
    log_a_false = log(a(false, hmm))
    @inbounds for t in 1:hmm.L-1
        maxstate = argmax(phi)
        @simd for j in 1:hmm.N
            if phi[j] + log_a_true > phi[maxstate] + log_a_false || j == maxstate
                from[j, t+1] = j
                phi[j] = phi[j] + log_a_true + log_b[j, t+1]
            else
                from[j, t+1] = maxstate
                phi[j] = phi[maxstate] + log_a_false + log_b[j, t+1]
            end
        end
    end
    cur = argmax(phi)
    recombinations = Vector{RecombinationEvent}()
    @inbounds @simd for t in hmm.L:-1:1
        if cur != from[cur, t]
            if ref_index(cur, hmm) != ref_index(from[cur, t], hmm)
                push!(recombinations, RecombinationEvent(t, ref_index(from[cur, t], hmm), ref_index(cur, hmm), from[cur, t], cur))
            end
            cur = from[cur, t]
        end
    end
    sort!(recombinations, by = x -> x.position)
    return RecombinationEvents(recombinations, ref_index(cur, hmm))
end