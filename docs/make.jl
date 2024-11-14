using Documenter
using CHMMera

makedocs(
    sitename = "CHMMera",
    format = Documenter.HTML(),
    modules = [CHMMera],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ]
)