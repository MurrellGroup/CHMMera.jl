# For algorithms

function maxtwo(arr::Vector{T}) where T<:Number
    indices = partialsortperm(arr, 1:2, rev=true)
    return collect(zip(indices, arr[indices]))
end

# For interface

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