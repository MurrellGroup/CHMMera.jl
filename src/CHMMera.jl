module CHMMera

export get_chimera_probabilities, get_recombination_events, get_log_site_probabilities, get_chimerapathevaluation

include("utils.jl")
include("hmm.jl")
include("algorithms.jl")
include("interface.jl")

end