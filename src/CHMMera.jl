module CHMMera

# Write your package code here.

export get_chimera_probabilities, get_recombination_events, get_site_log_probabilities

include("utils.jl")
include("hmm.jl")
include("algorithms.jl")

end