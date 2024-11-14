# CHMMera.jl

[![Coverage](https://codecov.io/gh/MurrellGroup/CHMMera.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/CHMMera.jl)

CHMMera.jl is a method for reference-based detection of chimeric DNA sequences.

Chimera detection is performed using a Hidden Markov Model (HMM) that models sequences as being generated from a single reference with mutations or from multiple references with mutations. We implemented two approaches to determine mutation rates:
- Discretizing mutation rates by creating a state for each mutation rate + template combination, called the Discretized Bayesian (DB) approach.
- Estimating a continuous per-reference mutation rate using the Baum-Welch (BW) algorithm.


## Example usage
```
julia> using CHMMera

julia> refs = ["ACGTACGTACGT", "ACCACCACCAAT"]

julia> queries = ["ACGTACACCAAT", "ACCACCACCAGT"]

julia> get_chimera_probabilities(queries, refs)
2-element Vector{Float64}:
 0.9956728308121604
 0.05863188730424695

julia> get_recombination_events(queries, refs; detailed = false)
2-element Vector{RecombinationEvents}:
 (recombinations = RecombinationEvent[(position = 7, left = 1, right = 2, left_state = 1, right_state = 2)], startingpoint = 1)
 (recombinations = RecombinationEvent[], startingpoint = 2)

julia> get_recombination_events(queries, refs; detailed = true)
2-element Vector{DetailedRecombinationEvents}:
 (recombinations = RecombinationEvent[(position = 7, left = 1, right = 2, left_state = 1, right_state = 2)], startingpoint = 1, pathevaluation = 0.995612088796054, logsiteprobabilities = [-0.007751384549707252, -0.006081763823303499, -0.00439756634058619, -0.00464963436088528, -0.01670468631375581, -0.7003603824992484, -4.360877076008885, -0.0003315532714940339, -1.1770957131840287e-5, -4.643262233303136e-6, -4.698595592046717e-5, -0.0017033680895343628])
 (recombinations = RecombinationEvent[], startingpoint = 2, pathevaluation = -Inf, logsiteprobabilities = Float64[])
```