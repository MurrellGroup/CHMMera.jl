
function get_chimera_probabilities(queries::Vector{Vector{UInt8}}, references::Vector{Vector{UInt8}}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300)
    hmm = fast ? ApproximateHMM(vovtomatrix(references), base_mutation_probability, prior_probability) : FullHMM(vovtomatrix(references), mutation_probabilities, prior_probability)
    mutation_probabilities = fast ? [base_mutation_probability for i in 1:length(references)] : mutation_probabilities

    # split query iteration among threads
    chimera_probabiltiies = zeros(length(queries))
    Threads.@threads for i in eachindex(queries)
        chimera_probabiltiies[i] = chimeraprobability(queries[i], hmm, copy(mutation_probabilities))
    end
    return chimera_probabiltiies
end

"""
    get_chimera_probabilities(queries::Vector{String}, references::Vector{String}, fast::Bool = true, prior_probability::Float64 = 1/300)

Get the probability of a sequence being chimeric for each query sequence given a list of reference sequences. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_chimera_probabilities(queries::Vector{String}, references::Vector{String}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300) = 
    get_chimera_probabilities(as_ints.(queries), as_ints.(references), fast, mutation_probabilities, base_mutation_probability, prior_probability)

const Recombations = NamedTuple{(:recombinations, :startingpoint, :pathevaluation), Tuple{Vector{NamedTuple{(:position, :at, :to), Tuple{Int64, Int64, Int64}}}, Int64, Float64}}

function get_recombination_events(queries::Vector{Vector{UInt8}}, references::Vector{Vector{UInt8}}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300, startingpoint::Bool = false, pathevaluation::Bool = false)

    hmm = fast ? ApproximateHMM(vovtomatrix(references), base_mutation_probability, prior_probability) : FullHMM(vovtomatrix(references), mutation_probabilities, prior_probability)
    mutation_probabilities = fast ? [base_mutation_probability for i in eachindex(references)] : mutation_probabilities 

    # split query iteration among threads
    recombination_events = Vector{Recombations}(undef, length(queries))
    Threads.@threads for i in eachindex(queries)
            recombination_events[i] = get_recombination_events(queries[i], hmm, copy(mutation_probabilities), Val(startingpoint), Val(pathevaluation))
    end
    return recombination_events
end

# multiple dispatch to figure out if we want starting point or pathevaluation
function get_recombination_events(query::Vector{UInt8}, hmm::HMM, mutation_probabilities::Vector{Float64}, ::Val{false}, ::Val{false})
    findrecombinations(query, hmm, mutation_probabilities)
end

function get_recombination_events(query::Vector{UInt8}, hmm::HMM, mutation_probabilities::Vector{Float64}, ::Val{true}, ::Val{false})
    findrecombinations_and_startingpoint(query, hmm, mutation_probabilities)
end

function get_recombination_events(query::Vector{UInt8}, hmm::HMM, mutation_probabilities::Vector{Float64}, ::Val{true}, ::Val{true})
    findrecombinations_and_startingpoint_and_pathevaulation(query, hmm, mutation_probabilities)
end

"""
    get_recombination_events(query::String, references::Vector{String}, fast = true, prior_probability = 1/300)

Get the recombination events for a query sequence given a set of reference sequences. The return type is `Vector{NamedTuple{(:position, :at, :to), Int64, Int64, Int64}}`. Each tuple represents a recombination event and is of the form `(position, at, next)`, where `at` and `next` are indices of the references, whilst position is the site of the recombination event. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_recombination_events(queries::Vector{String}, references::Vector{String}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300, startingpoint::Bool = false, pathevaluation::Bool = false) = 
    get_recombination_events(as_ints.(queries), as_ints.(references), fast, mutation_probabilities, base_mutation_probability, prior_probability, startingpoint, pathevaluation)

"""
    get_log_site_probabilities(query::String, references::Vector{String}, fast = true, prior_probability = 1/300)
Find the probability for each site (i.e., forward[t] * backward[t] for each t) in the path given by the viterbi algorithm. Can only be run with chimeric sequences.
"""
function get_log_site_probabilities(queries::Vector{String}, references::Vector{String}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300)

    recombs = get_recombination_events(queries, references, fast, mutation_probabilities, base_mutation_probability, prior_probability, false, false)

    hmm = fast ? ApproximateHMM(vovtomatrix(as_ints.(references)), base_mutation_probability, prior_probability) : FullHMM(vovtomatrix(as_ints.(references)), mutation_probabilities, prior_probability)
    mutation_probabilities = fast ? [base_mutation_probability for i in eachindex(references)] : mutation_probabilities
    queries = as_ints.(queries)

    logsiteprobabilities_ = Vector{Vector{Float64}}(undef, length(recombs))
    Threads.@threads for i in eachindex(recombs)
            logsiteprobabilities_[i] = logsiteprobabilities(recombs[i], queries[i], hmm, copy(mutation_probabilities))
    end
    return logsiteprobabilities_
end

function get_chimerapathevaluations(queries::Vector{String}, references::Vector{String}, fast::Bool = true, mutation_probabilities::Vector{Float64} = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability::Float64 = 0.05, prior_probability::Float64 = 1/300)
    hmm = fast ? ApproximateHMM(vovtomatrix(as_ints.(references)), base_mutation_probability, prior_probability) : FullHMM(vovtomatrix(as_ints.(references)), mutation_probabilities, prior_probability)
    mutation_probabilities = fast ? [base_mutation_probability for i in eachindex(references)] : mutation_probabilities
    queries = as_ints.(queries)

    # split query iteration among threads
    chimerapathevaluations = Vector{Float64}(undef, length(queries))
    Threads.@threads for i in eachindex(queries)
            chimerapathevaluations[i] = chimerapathevaluation(queries[i], hmm, copy(mutation_probabilities))
    end
    return chimerapathevaluations
end