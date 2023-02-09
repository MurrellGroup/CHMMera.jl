#Approx Bayes version

function forward(O::Vector{Int64}, hmm::ApproximateHMM)
    alfa = Array{Float64}(undef, 2, hmm.N)
    c = Vector{Float64}(undef, hmm.L)
    c[1] = 1
    for i in 1:hmm.N
        alfa[1, i] = initialstate(hmm) * b(i, 1, O, hmm)
        alfa[2, i] = 0.0
    end
    for t in 1:hmm.L-1
        sum_alfa = sum(alfa)
        for j in 1:hmm.N
            alfa[2, j] = ((sum_alfa - sum(alfa[:, j])) * a(false, hmm) + alfa[2, j] * a(true, hmm)) * b(j, t+1, O, hmm)
            alfa[1, j] = alfa[1, j] * a(true, hmm) * b(j, t+1, O, hmm)
        end
        c[t+1] = 1/sum(alfa)
        alfa .*= c[t+1]
    end
    
    detec = sum(alfa[2, :])
    return detec / (sum(alfa[1, :]) + detec)   #sum(alfa[2, :]) / sum(alfa)
end

function forward!(alfa::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::ApproximateHMM)
    c[1] = 1
    for i in 1:hmm.N
        alfa[i, 1] = initialstate(hmm) * b(i, 1, O, hmm)
    end
    for t in 1:hmm.L-1
        sumalfa = 0.0
        for x in alfa[:, t]
            sumalfa += x
        end
        newsumalfa = 0.0
        for j in 1:hmm.N
            alfa[j, t+1] = ((sumalfa-alfa[j, t])*a(false, hmm) + alfa[j, t]*a(true, hmm)) * b(j, t+1, O, hmm)
            newsumalfa += alfa[j, t+1]
        end
        c[t+1] = 1/newsumalfa
        alfa[:,t+1] .*= c[t+1]
    end
end

function backward!(beta::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, hmm::ApproximateHMM)
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

function parameterestimation!(O::Vector{Int64}, hmm::ApproximateHMM)
    alfa = Array{Float64}(undef, hmm.N, hmm.L)
    beta = Array{Float64}(undef, hmm.N, hmm.L)
    c = Array{Float64}(undef, hmm.L)
    forward!(alfa, c, O, hmm)
    backward!(beta, c, O, hmm)
    for i in 1:hmm.N
        Nmut = 2.0 #pseudocount "prior"?
        Nsame = 10.0 #pseudocount "prior"?
        for t in 1:hmm.L
            if O[t] != 6 #BM change, to handle non-informative obs
                if O[t] != hmm.S[i, t]
                    Nmut += (alfa[i, t]*beta[i, t]/c[t])
                else
                    Nsame += (alfa[i, t]*beta[i, t]/c[t])
                end
            end
        end
        hmm.pmut[i] = (Nmut) / ((Nmut + Nsame))
    end
end

function viterbi(O::Vector{Int64}, hmm::ApproximateHMM)
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
        if cur != from[cur, t]
            push!(recombinations, (position=t-1, at=from[cur, t], to=cur))
            cur = from[cur, t]
        end
    end
    return recombinations
end

function chimeraprobability(O::Vector{Int64}, hmm::ApproximateHMM)
    parameterestimation!(O, hmm)
    return forward(O, hmm)
end

function findrecombinations(O::Vector{Int64}, hmm::ApproximateHMM)
    parameterestimation!(O, hmm)
    return viterbi(O, hmm)
end

#Full Bayes version

function forward(O::Vector{Int64}, hmm::FullHMM)
    state(m, seq_idx) = (seq_idx-1)*hmm.K + m
    seq_idx(i) = div(i-1, hmm.K) + 1
    alfa = Matrix{Float64}(undef, 2, hmm.N)
    for m in 1:hmm.K, seq_idx in 1:hmm.n
        i = state(m, seq_idx)
        alfa[1, i] = initialstate(hmm) * b(seq_idx, m, 1, O, hmm)
        alfa[2, i] = 0.0
    end
    c = Vector{Float64}(undef, hmm.L)
    c[1] = 1
    for t in 1:hmm.L-1
        sumalfa = 0.0
        for i in 1:hmm.N, x in 1:2
            sumalfa += alfa[x, i]
        end
        newsumalfa = 0.0
        for m in 1:hmm.K, seq_idx in 1:hmm.n
            j = state(m, seq_idx)
            alfa[2, j] = ((sumalfa - sum(alfa[:, j]))*a(false, hmm) + alfa[2, j]*a(true, hmm)) * b(seq_idx, m, t+1, O, hmm)
            alfa[1, j] = alfa[1, j] * a(true, hmm) * b(seq_idx, m, t+1, O, hmm)
            newsumalfa += alfa[j]
        end
        c[t+1] = 1/newsumalfa
        alfa .*= c[t+1]
    end
    #This way prevents returns > 1 due to float nonsense
    detec = sum(alfa[2, :])
    return detec / (sum(alfa[1, :]) + detec)   #sum(alfa[2, :]) / sum(alfa)
end

function viterbi(O::Vector{Int64}, hmm::FullHMM)
    state(m, seq_idx) = (seq_idx-1)*hmm.K + m
    seq_idx(i) = div(i-1, hmm.K) + 1
    loga = Array{Float64}(undef, 2, hmm.N)
    logb = Array{Float64}(undef, hmm.N, hmm.L)
    loga[1, :] .= log(a(true, hmm))
    loga[2, :] .= log(a(false, hmm))
    for t in 1:hmm.L, m in 1:hmm.K, seq_idx in 1:hmm.n
        logb[state(m, seq_idx), t] = log(b(seq_idx, m, t, O, hmm))
    end
    phi = Array{Float64}(undef, hmm.n * hmm.K)
    from = Array{Int64}(undef, hmm.n * hmm.K, hmm.L)
    for m in 1:hmm.K, seq_idx in 1:hmm.n
        i = state(m, seq_idx)
        phi[i] = log(initialstate(hmm)) + logb[i, 1]
        from[i, 1] = i
    end
    for t in 1:hmm.L-1
        max1, max2 = maxtwo(phi)
        for m in 1:hmm.K, seq_idx in 1:hmm.n
            j = state(m, seq_idx)
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
        if cur != from[cur, t]
            push!(recombinations, (position=t-1, at=seq_idx(from[cur, t]), to=seq_idx(cur)))
            cur = from[cur, t]
        end
    end
    return recombinations
end

chimeraprobability(O::Vector{Int64}, hmm::FullHMM) = forward(O, hmm)

findrecombinations(O::Vector{Int64}, hmm::FullHMM) = viterbi(O, hmm)
