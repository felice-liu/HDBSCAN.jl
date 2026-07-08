using Documenter
using HDBSCAN

makedocs(
    sitename = "HDBSCAN.jl",
    modules = [HDBSCAN],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)