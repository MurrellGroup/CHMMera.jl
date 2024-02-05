#Approx Bayes version

# variable names are based on Rabiner, A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition

function forward(O::Vector{Int64}, hmm::HMM)
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    for i in 1:hmm.N
        alpha[1, i] = initialstate(hmm) * b(i, 1, O, hmm)
        alpha[2, i] = 0.0
    end
    for t in 1:hmm.L-1
        alpha_colsums = sum.(eachcol(alpha))
        sumalpha = sum(alpha)
        for j in 1:hmm.N
            bval = b(j, t+1, O, hmm)
            alpha[2, j] = ((sumalpha - alpha_colsums[j]) * a(false, hmm) + alpha[2, j] * a(true, hmm)) * bval
            alpha[1, j] = alpha[1, j] * a(true, hmm) * bval

        end
        scaling_constant = 1/sum(alpha)
        alpha .*= scaling_constant
    end
    
    detec = sum(alpha[2, :])
    return detec / (sum(alpha[1, :]) + detec)   #sum(alpha[2, :]) / sum(alpha)
end

function forward!(alpha::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::HMM)
    c[1] = 1
    for i in 1:hmm.N
        alpha[i, 1] = initialstate(hmm) * b(i, 1, O, hmm)
    end
    for t in 1:hmm.L-1
        sumalpha = 0.0
        for x in alpha[:, t]
            sumalpha += x
        end
        newsumalpha = 0.0
        for j in 1:hmm.N
            alpha[j, t+1] = ((sumalpha-alpha[j, t])*a(false, hmm) + alpha[j, t]*a(true, hmm)) * b(j, t+1, O, hmm)
            newsumalpha += alpha[j, t+1]
        end
        c[t+1] = 1/newsumalpha
        alpha[:,t+1] .*= c[t+1]
    end
end

function backward!(beta::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::HMM)
    beta[:, hmm.L] .= c[hmm.L]
    for t in hmm.L-1:-1:1
        sumbeta = 0.0
        for j in 1:hmm.N
            sumbeta += beta[j, t+1]*b(j, t+1, O, hmm)
        end
        for i in 1:hmm.N
            beta[i, t] = a(false, hmm)*(sumbeta - beta[i, t+1]*b(i, t+1, O, hmm)) + a(true, hmm)*beta[i, t+1]*b(i, t+1, O, hmm)
            beta[i, t] *= c[t]
        end
    end
end

function viterbi(O::Vector{Int64}, hmm::HMM)
    phi = Array{Float64}(undef, hmm.N)
    from = Array{Int64}(undef, hmm.N, hmm.L)
    for i in 1:hmm.N 
        phi[i] = log(initialstate(hmm) * b(i, 1, O, hmm))
        from[i, 1] = i
    end
    for t in 1:hmm.L-1
        maxstate = argmax(phi)
        for j in 1:hmm.N
            if phi[j] + log(a(true, hmm)) > phi[maxstate] + log(a(false, hmm)) || j == maxstate
                from[j, t+1] = j
                phi[j] = phi[j] + log(a(true, hmm)) + log(b(j, t+1, O, hmm))
            else
                from[j, t+1] = maxstate
                phi[j] = phi[maxstate] + log(a(false, hmm)) + log(b(j, t+1, O, hmm))
            end
        end
    end
    cur = argmax(phi)
    recombinations = NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}[]
    for t in hmm.L:-1:1
        if cur != from[cur, t]
            if ref_index(cur, hmm) != ref_index(from[cur, t], hmm)
                push!(recombinations, (position=t-1, at=ref_index(from[cur, t], hmm), to=ref_index(cur, hmm)))
            end
            cur = from[cur, t]
        end
    end
    sort!(recombinations, by = x -> x.at)
    return (recombinations = recombinations, startingpoint = ref_index(cur, hmm))
end

function parameterestimation!(hmm::ApproximateHMM, O::Vector{Int64})
    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, O, hmm)
    backward!(beta, c, O, hmm)
    for i in 1:hmm.N
        Nmut = 2.0 #pseudocount "prior"?
        Nsame = 10.0 #pseudocount "prior"?
        for t in 1:hmm.L
            if O[t] != 6 #BM change, to handle non-informative obs
                if O[t] != hmm.S[i, t]
                    Nmut += (alpha[i, t]*beta[i, t]/c[t])
                else
                    Nsame += (alpha[i, t]*beta[i, t]/c[t])
                end
            end
        end
        hmm.mutation_probabilities[i] = (Nmut) / ((Nmut + Nsame))
    end
end

function logsiteprobabilities(recombs::Vector{NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}}, O::Vector{Int}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O)

    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, O, hmm)
    backward!(beta, c, O, hmm)
    sort!(recombs, by = x -> x.position)
    log_probability = Array{Float64}(undef, hmm.L)
    i = 1
    cur = recombs[i].at
    for t in 1:hmm.L
        log_probability[t] = log(alpha[cur, t]) + log(beta[cur, t]) - log(c[t])
        if t < hmm.L && i <= length(recombs) && recombs[i].position == t
            cur = recombs[i].to
            i += 1
        end
    end
    return log_probability
end

function chimeraprobability(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O)
    return forward(O, hmm)
end

function findrecombinations(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O)
    return viterbi(O, hmm)[1]
end

function findrecombinations_and_startingpoint(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O)
    return viterbi(O, hmm)
end

function chimerapathevaluation(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O)

    recombs, startingpoint = viterbi(O, hmm)
    !isempty(recombs) || throw("Can't evaluate path without recombinations")

    # scaling constants
    c = Vector{Float64}(undef, hmm.L)
    # forward
    alpha = Matrix{Float64}(undef, hmm.N, hmm.L) 
    forward!(alpha, c, O, hmm)
    # backward
    beta = Matrix{Float64}(undef, hmm.N, hmm.L)
    backward!(beta, c, O, hmm)

    # Normalized log probability of being at each position, (state_index, site_index)
    logp_position = Matrix{Float64}(undef, hmm.N, hmm.L)
    for t in 1:hmm.L
        p_position = alpha[:, t] .* beta[:, t] # unnormalized
        logp_position[:, t] = log.(p_position) .- log(sum(p_position)) # log normalized
    end

    # p_ref[i] = the probability of being at ref[i] at the time t, for the t that maximizes this probability, where t ∈ {t : viterbi_path[t] == ref[i]}
    p_ref = Dict(union([startingpoint => 0.0], [recomb.to => 0.0 for recomb in recombs]))

    cur = startingpoint
    recombindex = 1
    for t in 1:hmm.L
        # For the fullbayesian version, each reference has multiple states, hence why we use stateindicesofref
        # exp(logp_position[i, t]) should not underflow here, since we are iterating over the path with the highest (log)probability
        p_ref[cur] = max(sum( exp(logp_position[i, t]) for i in stateindicesofref(cur, hmm) ), p_ref[cur])
        if recombindex <= length(recombs) && t == recombs[recombindex].position
            cur = recombs[recombindex].to
            recombindex += 1
        end
    end

    probability_of_2nd_most_probable_ref = sort(collect(values(p_ref)))[end - 1]
    return probability_of_2nd_most_probable_ref
end

##### Below are versions of the algorithms functions that have mutation_probabilities factored out into a Vector separate from the hmm to 
##### avoid having to allocate the hmm multiple times 


# parameter estimation with the mutation_probabilities factored out of the hmm object to allow multithreading
function parameterestimation!(hmm::ApproximateHMM, O::Vector{Int64}, mutation_probabilities::Vector{Float64})
    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, O, hmm, mutation_probabilities)
    backward!(beta, c, O, hmm, mutation_probabilities)
    for i in 1:hmm.N
        Nmut = 2.0 #pseudocount "prior"?
        Nsame = 10.0 #pseudocount "prior"?
        for t in 1:hmm.L
            if O[t] != 6 #BM change, to handle non-informative obs
                if O[t] != hmm.S[i, t]
                    Nmut += (alpha[i, t]*beta[i, t]/c[t])
                else
                    Nsame += (alpha[i, t]*beta[i, t]/c[t])
                end
            end
        end
        mutation_probabilities[i] = (Nmut) / ((Nmut + Nsame))
    end
end

using Base.Threads


function chimeraprobability(O::Vector{Int64}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM

    function unsafe_pointer_from_objectref(@nospecialize(x))
        #= Warning Danger=#
        ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x)
    end
#    @info "Thread ID: $(threadid()) Unique var address: $(unsafe_pointer_from_objectref(mutation_probabilities))"
#    @info "Thread ID: $(threadid()) Shared var address: $(unsafe_pointer_from_objectref(hmm))"
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities)
    return forward(O, hmm, mutation_probabilities::Vector{Float64})
end

function forward(O::Vector{Int64}, hmm::HMM, mutation_probabilities::Vector{Float64})
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    for i in 1:hmm.N
        alpha[1, i] = initialstate(hmm) * b(i, 1, O, hmm, mutation_probabilities)
        alpha[2, i] = 0.0
    end
    for t in 1:hmm.L-1
        alpha_colsums = sum.(eachcol(alpha))
        sumalpha = sum(alpha)
        for j in 1:hmm.N
            bval = b(j, t+1, O, hmm, mutation_probabilities)
            alpha[2, j] = ((sumalpha - alpha_colsums[j]) * a(false, hmm) + alpha[2, j] * a(true, hmm)) * bval
            alpha[1, j] = alpha[1, j] * a(true, hmm) * bval

        end
        scaling_constant = 1/sum(alpha)
        alpha .*= scaling_constant
    end
    
    detec = sum(alpha[2, :])
    return detec / (sum(alpha[1, :]) + detec)   #sum(alpha[2, :]) / sum(alpha)
end


function forward!(alpha::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::HMM, mutation_probabilities::Vector{Float64})
    c[1] = 1
    for i in 1:hmm.N
        alpha[i, 1] = initialstate(hmm) * b(i, 1, O, hmm, mutation_probabilities)
    end
    for t in 1:hmm.L-1
        sumalpha = 0.0
        for x in alpha[:, t]
            sumalpha += x
        end
        newsumalpha = 0.0
        for j in 1:hmm.N
            alpha[j, t+1] = ((sumalpha-alpha[j, t])*a(false, hmm) + alpha[j, t]*a(true, hmm)) * b(j, t+1, O, hmm, mutation_probabilities)
            newsumalpha += alpha[j, t+1]
        end
        c[t+1] = 1/newsumalpha
        alpha[:,t+1] .*= c[t+1]
    end
end

function backward!(beta::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::HMM, mutation_probabilities::Vector{Float64})
    beta[:, hmm.L] .= c[hmm.L]
    for t in hmm.L-1:-1:1
        sumbeta = 0.0
        for j in 1:hmm.N
            sumbeta += beta[j, t+1]*b(j, t+1, O, hmm, mutation_probabilities)
        end
        for i in 1:hmm.N
            beta[i, t] = a(false, hmm)*(sumbeta - beta[i, t+1]*b(i, t+1, O, hmm, mutation_probabilities)) + a(true, hmm)*beta[i, t+1]*b(i, t+1, O, hmm, mutation_probabilities)
            beta[i, t] *= c[t]
        end
    end
end

function findrecombinations_and_startingpoint(O::Vector{Int64}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities)
    return viterbi(O, hmm, mutation_probabilities)
end


function viterbi(O::Vector{Int64}, hmm::HMM, mutation_probabilities::Vector{Float64})
    phi = Array{Float64}(undef, hmm.N)
    from = Array{Int64}(undef, hmm.N, hmm.L)
    for i in 1:hmm.N 
        phi[i] = log(initialstate(hmm) * b(i, 1, O, hmm, mutation_probabilities))
        from[i, 1] = i
    end
    for t in 1:hmm.L-1
        maxstate = argmax(phi)
        for j in 1:hmm.N
            if phi[j] + log(a(true, hmm)) > phi[maxstate] + log(a(false, hmm)) || j == maxstate
                from[j, t+1] = j
                phi[j] = phi[j] + log(a(true, hmm)) + log(b(j, t+1, O, hmm, mutation_probabilities))
            else
                from[j, t+1] = maxstate
                phi[j] = phi[maxstate] + log(a(false, hmm)) + log(b(j, t+1, O, hmm, mutation_probabilities))
            end
        end
    end
    cur = argmax(phi)
    recombinations = NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}[]
    for t in hmm.L:-1:1
        if cur != from[cur, t]
            if ref_index(cur, hmm) != ref_index(from[cur, t], hmm)
                push!(recombinations, (position=t-1, at=ref_index(from[cur, t], hmm), to=ref_index(cur, hmm)))
            end
            cur = from[cur, t]
        end
    end
    sort!(recombinations, by = x -> x.at)
    return (recombinations = recombinations, startingpoint = ref_index(cur, hmm))
end


function findrecombinations(O::Vector{Int64}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities)
    return viterbi(O, hmm, mutation_probabilities)[1]
end


function chimerapathevaluation(O::Vector{Int64}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities)

    recombs, startingpoint = viterbi(O, hmm, mutation_probabilities)
    !isempty(recombs) || throw("Can't evaluate path without recombinations")

    # scaling constants
    c = Vector{Float64}(undef, hmm.L)
    # forward
    alpha = Matrix{Float64}(undef, hmm.N, hmm.L) 
    forward!(alpha, c, O, hmm, mutation_probabilities)
    # backward
    beta = Matrix{Float64}(undef, hmm.N, hmm.L)
    backward!(beta, c, O, hmm, mutation_probabilities)

    # Normalized log probability of being at each position, (state_index, site_index)
    logp_position = Matrix{Float64}(undef, hmm.N, hmm.L)
    for t in 1:hmm.L
        p_position = alpha[:, t] .* beta[:, t] # unnormalized
        logp_position[:, t] = log.(p_position) .- log(sum(p_position)) # log normalized
    end

    # p_ref[i] = the probability of being at ref[i] at the time t, for the t that maximizes this probability, where t ∈ {t : viterbi_path[t] == ref[i]}
    p_ref = Dict(union([startingpoint => 0.0], [recomb.to => 0.0 for recomb in recombs]))

    cur = startingpoint
    recombindex = 1
    for t in 1:hmm.L
        # For the fullbayesian version, each reference has multiple states, hence why we use stateindicesofref
        # exp(logp_position[i, t]) should not underflow here, since we are iterating over the path with the highest (log)probability
        p_ref[cur] = max(sum( exp(logp_position[i, t]) for i in stateindicesofref(cur, hmm) ), p_ref[cur])
        if recombindex <= length(recombs) && t == recombs[recombindex].position
            cur = recombs[recombindex].to
            recombindex += 1
        end
    end

    probability_of_2nd_most_probable_ref = sort(collect(values(p_ref)))[end - 1]
    return probability_of_2nd_most_probable_ref
end


function logsiteprobabilities(recombs::Vector{NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}}, O::Vector{Int}, hmm::T, mutation_probabilities::Vector{Float64}) where T <: HMM
    # length(recombs) > 0 || "No recombinations found, can only be run on chimeric sequences"
    if length(recombs) == 0
        return zeros(Float64, length(O))
    end
    T == ApproximateHMM && parameterestimation!(hmm, O, mutation_probabilities)

    alpha = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alpha, c, O, hmm, mutation_probabilities)
    backward!(beta, c, O, hmm, mutation_probabilities)
    sort!(recombs, by = x -> x.position)
    log_probability = Array{Float64}(undef, hmm.L)
    i = 1
    cur = recombs[i].at
    for t in 1:hmm.L
        log_probability[t] = log(alpha[cur, t]) + log(beta[cur, t]) - log(c[t])
        if t < hmm.L && i <= length(recombs) && recombs[i].position == t
            cur = recombs[i].to
            i += 1
        end
    end
    return log_probability
end
