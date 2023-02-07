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

# API
function vovtomatrix(vov)
    n = length(vov)
    L = minimum(length.(vov))
    mat = Matrix{Int64}(undef, n, L)
    for j=1:L, i=1:n
        mat[i, j] = vov[i][j]
    end
    return mat
end

function get_chimeraprobabilities(queries::Vector{Vector{Int64}}, references::Vector{Vector{Int64}}; fast::Bool = true, prior_probability::Float64 = 1/300)
    mcat = [0.02, 0.04, 0.07, 0.11, 0.15]
    basem = 0.05
    newhmm(refs::Vector{Vector{Int64}}) = fast ? ApproximateHMM(vovtomatrix(refs), basem, prior_probability) : FullHMM(vovtomatrix(refs), mcat, prior_probability)
    return Float64[chimeraprobability(q, newhmm(references)) for q in queries]
end

get_chimeraprobabilities(queries::Vector{String}, references::Vector{String}; fast::Bool = true, prior_probability::Float64 = 1/300) = get_chimeraprobabilities(as_ints.(queries), as_ints.(references); fast = fast, prior_probability = prior_probability)

function get_path(query::Vector{Int64}, references::Vector{Vector{Int64}}; fast::Bool = true, prior_probability::Float64 = 1/300)
    if fast
        hmm = ApproximateHMM(vovtomatrix(references), 0.05, prior_probability)
    else
        hmm = FullHMM(vovtomatrix(references), [0.02, 0.04, 0.07, 0.11, 0.15], prior_probability)
    end
    return findpath(query, hmm)
end

get_path(query::String, references::Vector{String}; fast = true, prior_probability = 1/300) = get_path(as_ints(query), as_ints.(references), fast = fast, prior_probability = prior_probability)