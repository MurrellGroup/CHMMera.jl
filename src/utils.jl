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

const NUC2INT = (
    'A' => 0x01,
    'C' => 0x02,
    'G' => 0x03,
    'T' => 0x04,
    '-' => 0x05,
    'N' => 0x06
)

const NUC2INT_LOOKUP = let
    table = fill(0x06, 256) 
    for (k, v) in NUC2INT
        table[Int(k)] = v
    end
    tuple(table...) 
end

function as_ints(seq::String)
    ints = Vector{UInt8}(undef, length(seq))
    c_ints = Int.(codeunits(seq))
    @inbounds for i in eachindex(seq)
        ints[i] = NUC2INT_LOOKUP[c_ints[i]]
    end
    return ints
end

function vovtomatrix(vov::Vector{Vector{UInt8}})
    n = length(vov)
    L = minimum(length.(vov))
    mat = Matrix{UInt8}(undef, n, L)
    for j=1:L, i=1:n
        mat[i, j] = vov[i][j]
    end
    return mat
end