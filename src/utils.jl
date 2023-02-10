# For algs

function maxtwo(arr::Vector{T}) where T<:Number
    indices = partialsortperm(arr, 1:2, rev=true)
    return collect(zip(indices, arr[indices]))
end

# Utils for API

function as_string(seq::Vector{Int64})
    mymap = ["A", "C", "G", "T", "-", "N"]
    return join((mymap[nt] for nt in seq))
end

function as_ints(seq::String)
    mymap = ['A', 'C', 'G', 'T', '-', 'N']
    return Int64[findfirst(mymap.==uppercase(nt)) for nt in seq]
end

function vovtomatrix(vov)
    n = length(vov)
    L = minimum(length.(vov))
    mat = Matrix{Int64}(undef, n, L)
    for j=1:L, i=1:n
        mat[i, j] = vov[i][j]
    end
    return mat
end

# API
function get_chimeraprobabilities(queries::Vector{Vector{Int64}}, references::Vector{Vector{Int64}}; fast::Bool = true, prior_probability::Float64 = 1/300)
    mcat = [0.02, 0.04, 0.07, 0.11, 0.15]
    basem = 0.05
    newhmm(refs::Vector{Vector{Int64}}) = fast ? ApproximateHMM(vovtomatrix(refs), basem, prior_probability) : FullHMM(vovtomatrix(refs), mcat, prior_probability)
    return Float64[chimeraprobability(q, newhmm(references)) for q in queries]
end


"""
    get_chimeraprobabilities(queries::Vector{String}, references::Vector{String}; fast::Bool = true, prior_probability::Float64 = 1/300)

Get the probability of a sequence being chimeric for each query sequence given a list of reference sequences. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_chimeraprobabilities(queries::Vector{String}, references::Vector{String}; fast::Bool = true, prior_probability::Float64 = 1/300) = get_chimeraprobabilities(as_ints.(queries), as_ints.(references); fast = fast, prior_probability = prior_probability)

function get_recombination_events(query::Vector{Int64}, references::Vector{Vector{Int64}}; fast::Bool = true, prior_probability::Float64 = 1/300)
    if fast
        hmm = ApproximateHMM(vovtomatrix(references), 0.05, prior_probability)
    else
        hmm = FullHMM(vovtomatrix(references), [0.02, 0.04, 0.07, 0.11, 0.15], prior_probability)
    end
    return findrecombinations(query, hmm)
end

"""
    get_recombination_events(query::String, references::Vector{String}; fast = true, prior_probability = 1/300)

Get the recombination events for a query sequence given a set of reference sequences. The return type is `Vector{NamedTuple{(:position, :at, :to), Int64, Int64, Int64}}`. Each tuple represents a recombination event and is of the form `(position, at, next)`, where `at` and `next` are indices of the references, whilst position is the site of the recombination event. `fast` is a boolean indicating whether to use the approximate HMM or the full HMM. `prior_probability` is the prior probability of a sequence being chimeric.
"""
get_recombination_events(query::String, references::Vector{String}; fast = true, prior_probability = 1/300) = get_recombination_events(as_ints(query), as_ints.(references), fast = fast, prior_probability = prior_probability)

"""
    path_scores(recombs::Vector{NamedTuple{(:position, :at, :to), Int64, Int64, Int64}}, query::Vector{Int64}, hmm::HMM)
Find the probability for each site (i.e., forward[t] * backward[t] for each t) in the path given by the viterbi algorithm.
"""
function get_site_log_probabilities(query::String, references::Vector{String}; prior_probability = 1/300)
    recombs = get_recombination_events(query, references, fast=true) # defaults to fast=true
    @assert length(recombs) > 0 "No recombinations found, can only be run on chimeric sequences"
    O = as_ints(query)
    hmm = ApproximateHMM(vovtomatrix(as_ints.(references)), 0.05, prior_probability)
    parameterestimation!(O, hmm)
    return sitelogprobabilities(recombs, O, hmm)
end