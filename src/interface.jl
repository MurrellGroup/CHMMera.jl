
function get_chimera_probabilities(queries::Vector{Vector{Int64}}, references::Vector{Vector{Int64}}; fast::Bool = true, mutation_probabilities = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability = 0.05, prior_probability::Float64 = 1/300)
    newhmm(refs::Vector{Vector{Int64}}) = fast ? ApproximateHMM(vovtomatrix(refs), base_mutation_probability, prior_probability) : FullHMM(vovtomatrix(refs), mutation_probabilities, prior_probability)
    return Float64[chimeraprobability(q, newhmm(references)) for q in queries]
end

"""
    get_chimera_probabilities(queries::Vector{String}, references::Vector{String}; fast::Bool = true, prior_probability::Float64 = 1/300)

Get the probability of a sequence being chimeric for each query sequence given a list of reference sequences. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_chimera_probabilities(queries::Vector{String}, references::Vector{String}; fast::Bool = true, mutation_probabilities = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability = 0.05, prior_probability::Float64 = 1/300) = 
    get_chimera_probabilities(as_ints.(queries), as_ints.(references); fast = fast, mutation_probabilities = mutation_probabilities, base_mutation_probability = base_mutation_probability, prior_probability = prior_probability)

function get_recombination_events(query::Vector{Int64}, references::Vector{Vector{Int64}}; fast::Bool = true, mutation_probabilities = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability = 0.05, prior_probability::Float64 = 1/300, startingpoint = false)
    if fast
        hmm = ApproximateHMM(vovtomatrix(references), base_mutation_probability, prior_probability)
    else
        hmm = FullHMM(vovtomatrix(references), mutation_probabilities, prior_probability)
    end
    return startingpoint ? findrecombinations_and_startingpoint(query, hmm) : findrecombinations(query, hmm)
end

"""
    get_recombination_events(query::String, references::Vector{String}; fast = true, prior_probability = 1/300)

Get the recombination events for a query sequence given a set of reference sequences. The return type is `Vector{NamedTuple{(:position, :at, :to), Int64, Int64, Int64}}`. Each tuple represents a recombination event and is of the form `(position, at, next)`, where `at` and `next` are indices of the references, whilst position is the site of the recombination event. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_recombination_events(query::String, references::Vector{String}; fast::Bool = true, mutation_probabilities = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability = 0.05, prior_probability::Float64 = 1/300, startingpoint = false) = 
    get_recombination_events(as_ints(query), as_ints.(references), fast = fast, mutation_probabilities = mutation_probabilities, base_mutation_probability = base_mutation_probability, prior_probability = prior_probability, startingpoint = startingpoint)

"""
    get_log_site_probabilities(query::String, references::Vector{String}; fast = true, prior_probability = 1/300)
Find the probability for each site (i.e., forward[t] * backward[t] for each t) in the path given by the viterbi algorithm. Can only be run with chimeric sequences.
"""
function get_log_site_probabilities(query::String, references::Vector{String}; fast::Bool = true, mutation_probabilities = [0.02, 0.04, 0.07, 0.11, 0.15], base_mutation_probability = 0.05, prior_probability::Float64 = 1/300)
    recombs = get_recombination_events(query, references, fast = fast, mutation_probabilities = mutation_probabilities, base_mutation_probability = base_mutation_probability, prior_probability = prior_probability)
    
    # length(recombs) > 0 || "No recombinations found, can only be run on chimeric sequences"
    if length(recombs) == 0
        return zeros(Float64, length(query))
    end

    O = as_ints(query)
    hmm = ApproximateHMM(vovtomatrix(as_ints.(references)), 0.05, prior_probability)
    parameterestimation!(O, hmm)
    return logsiteprobabilities(recombs, O, hmm)
end