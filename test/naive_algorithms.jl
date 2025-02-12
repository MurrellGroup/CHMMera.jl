
function naive_forward(hmm::CHMMera.ApproximateHMM, b::Matrix)
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    init_state = CHMMera.initialstate(hmm)
    for i in 1:hmm.N
        alpha[1, i] = init_state * b[i, 1]
        alpha[2, i] = 0.0
    end
    for t in 1:hmm.L-1
        next_alpha = zero(alpha)
        for j in 1:hmm.N
            for i in 1:hmm.N
                if i == j
                    next_alpha[1, j] += alpha[1, i] * CHMMera.a(true, hmm) * b[j, t+1]
                    next_alpha[2, j] += alpha[2, i] * CHMMera.a(true, hmm) * b[j, t+1]
                else
                    next_alpha[2, j] += alpha[1, i] * CHMMera.a(false, hmm) * b[j, t+1]
                    next_alpha[2, j] += alpha[2, i] * CHMMera.a(false, hmm) * b[j, t+1]
                end
            end
        end
        scaling_constant = 1.0 / sum(next_alpha)
        next_alpha *= scaling_constant
        alpha = next_alpha
    end
    return sum(alpha[2, :]) / sum(alpha)
end

function naive_forward(hmm::CHMMera.FullHMM, b::Matrix)
    alpha = Matrix{Float64}(undef, 2, hmm.N)
    init_state = CHMMera.initialstate(hmm)

    a_self = 1 - hmm.switch_probability - hmm.μ
    a_diffref = hmm.switch_probability / ((hmm.n - 1) * hmm.K)
    a_diffmut = hmm.K == 1 ? 0.0 : hmm.μ / (hmm.K - 1)

    for i in 1:hmm.N
        alpha[1, i] = init_state * b[i, 1]
        alpha[2, i] = 0.0
    end

    for t in 1:hmm.L-1
        next_alpha = zero(alpha)
        for i in 1:hmm.N
            for j in 1:hmm.N
                if i == j
                    next_alpha[1, j] += alpha[1, i] * a_self * b[j, t+1]
                    next_alpha[2, j] += alpha[2, i] * a_self * b[j, t+1]
                elseif i in CHMMera.stateindicesofref(CHMMera.ref_index(j, hmm), hmm)
                    next_alpha[1, j] += alpha[1, i] * a_diffmut * b[j, t+1]
                    next_alpha[2, j] += alpha[2, i] * a_diffmut * b[j, t+1]
                else
                    next_alpha[2, j] += alpha[1, i] * a_diffref * b[j, t+1]
                    next_alpha[2, j] += alpha[2, i] * a_diffref * b[j, t+1]
                end
            end
        end
        scaling_constant = 1.0 / sum(next_alpha)
        next_alpha *= scaling_constant
        alpha = next_alpha
    end
    return sum(alpha[2, :]) / sum(alpha)
end