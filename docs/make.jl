using Documenter
using Hdbscan

makedocs(
    sitename = "Hdbscan.jl",
    modules = [Hdbscan],
    pages = ["Home" => "index.md", "API" => "api.md"],
)
