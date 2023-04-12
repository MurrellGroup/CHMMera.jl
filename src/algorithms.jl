#Approx Bayes version

# alpha[t, i] = probability of ending up at Sᵢ after having observed observations up to t-1

function forward(O::Vector{Int64}, hmm::HMM)
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    for i in 1:hmm.N
        alpha[1, i] = initialstate(hmm) * b(i, 1, O, hmm)
        alpha[2, i] = 0.0
    end
    for t in 1:hmm.L-1
        sum_alpha = sum(alpha)
        for j in 1:hmm.N
            alpha[2, j] = ((sum_alpha - sum(alpha[:, j])) * a(false, hmm) + alpha[2, j] * a(true, hmm)) * b(j, t+1, O, hmm)
            alpha[1, j] = alpha[1, j] * a(true, hmm) * b(j, t+1, O, hmm)
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
    loga = Array{Float64}(undef, 2, hmm.N)
    logb = Array{Float64}(undef, hmm.N, hmm.L)
    loga[1, :] .= log(a(true, hmm))
    loga[2, :] .= log(a(false, hmm))
    for t in 1:hmm.L, i in 1:hmm.N
        logb[i, t] = log(b(i, t, O, hmm))
    end
    phi = Array{Float64}(undef, hmm.N)
    from = Array{Int64}(undef, hmm.N, hmm.L)
    for i in 1:hmm.N 
        phi[i] = log(initialstate(hmm)) + logb[i, 1]
        from[i, 1] = i
    end
    for t in 1:hmm.L-1
        max1, max2 = maxtwo(phi)
        for j in 1:hmm.N
            recomb_idx, recomb_val = max1[1] != j ? max1 : max2
            if phi[j] + loga[1, j] > recomb_val + loga[2, j]
                from[j, t+1] = j
                phi[j] = phi[j] + loga[1, j] + logb[j, t+1]
            else
                from[j, t+1] = recomb_idx
                phi[j] = recomb_val + loga[2, j] + logb[j, t+1]
            end
        end
    end
    cur = argmax(phi)
    recombinations = NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}[]
    for t in hmm.L:-1:1
        if ref_index(cur, hmm) != ref_index(from[cur, t], hmm)
            push!(recombinations, (position=t-1, at=ref_index(from[cur, t], hmm), to=ref_index(cur, hmm)))
            cur = from[cur, t]
        end
    end
    sort!(recombinations, by = x -> x.at)
    return (recombinations = recombinations, startingpoint = ref_index(cur, hmm))
end

function parameterestimation!(O::Vector{Int64}, hmm::ApproximateHMM)
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

function logsiteprobabilities(recombs::Vector{NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}}, O::Vector{Int}, hmm::HMM)
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
    T == ApproximateHMM && parameterestimation!(O, hmm)
    return forward(O, hmm)
end

function findrecombinations(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(O, hmm)
    return viterbi(O, hmm)[1]
end

function findrecombinations_and_startingpoint(O::Vector{Int64}, hmm::T) where T <: HMM
    T == ApproximateHMM && parameterestimation!(O, hmm)
    return viterbi(O, hmm)
end