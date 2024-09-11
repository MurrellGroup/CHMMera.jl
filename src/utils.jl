# For algorithms

function maxtwo(arr::Vector{T}) where T<:Number
    indices = partialsortperm(arr, 1:2, rev=true)
    return collect(zip(indices, arr[indices]))
end

# For interface

function as_string(seq::Vector{UInt8})
    mymap = ["A", "C", "G", "T", "-", "N"]
    return join((mymap[nt] for nt in seq))
end
NUC2INT = Dict('A' => 1, 'C' => 2, 'G' => 3, 'T' => 4, '-' => 5, 'N' => 6)

function as_ints(seq::String)
    ints = Vector{UInt8}(undef, length(seq))
    for i in eachindex(seq)
        ints[i] = NUC2INT[seq[i]]
    end
    return ints
end

function vovtomatrix(vov)
    n = length(vov)
    L = minimum(length.(vov))
    mat = Matrix{UInt8}(undef, n, L)
    for j=1:L, i=1:n
        mat[i, j] = vov[i][j]
    end
    return mat
end