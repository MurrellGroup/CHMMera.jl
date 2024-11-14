using Documenter
using CHMMera

makedocs(
    sitename = "CHMMera.jl",
    format = Documenter.HTML(),
    modules = [CHMMera],
    pages = [
        "Overview" => "index.md",
        "API" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/MurrellGroup/CHMMera.jl.git",
    devbranch = "main",
    push_preview = true
)