using CSV
using DataFrames
using Statistics
include(joinpath(SRC_DIR, "HDBSCAN.jl"))

const BENCHMARK_DIR = @__DIR__
const ROOT_DIR = normpath(joinpath(SRC_DIR, ".."))
const DATA_DIR = joinpath(ROOT_DIR, "data")
const SRC_DIR = joinpath(ROOT_DIR, "src")
const RESULT_DIR = joinpath(ROOT_DIR, "result")
const JULIA_RESULT_DIR = joinpath(RESULT_DIR, "julia")

mkpath(JULIA_RESULT_DIR)
using .HDBSCAN

# Dataset name -> has header?
const DATASETS = Dict{String,Bool}([
    "circles" => false,
    "moons" => false,
    "varied" => false,
    "aniso" => false,
    "blobs" => false,
    "no_structure" => false,
    "heartfailure" => true,
    "cardiacarrest" => true,
    "neuroblastoma" => true,
    "sepsis" => true,
    "type1diabetes" => true,
])

function vector_to_string(v)

    return join(v, " ")

end

function fill_missing_median!(df::DataFrame)

    for col in names(df)

        v = df[!, col]

        if any(ismissing, v)

            nonmiss = collect(skipmissing(v))

            if isempty(nonmiss)
                error("Column $col contains only missing values")
            end

            vals = Float64.(nonmiss)

            med = median(vals)

            df[!, col] = [ismissing(x) ? med : Float64(x) for x in v]

        else
            df[!, col] = Float64.(v)

        end

    end

    return df

end

function load_dataset(dataset_name::String, hasheader::Bool)

    path = joinpath(DATA_DIR, dataset_name * ".csv")

    if !isfile(path)
        error("Missing dataset CSV: $path")
    else
        df = CSV.read(path, DataFrame; header = hasheader)
        fill_missing_median!(df)
        return Matrix{Float64}(df)
    end
end



function generate_results(model, dataset_name::String, hasheader::Bool; n::Int = 1)

    println("Running Julia HDBSCAN on $dataset_name")
    X = load_dataset(dataset_name, hasheader)

    sum = 0.0
    average_fit_time = -1.0

    for i = 1:n
        t0 = time()
        fit!(model, X)
        fit_time = time() - t0
        sum += fit_time
    end

    if sum > 0
        average_fit_time = sum / n
    end

    cluster_labels = labels(model)
    cluster_probabilities = probabilities(model)

    out = DataFrame(
        dataset = [dataset_name],
        average_fit_time_sec = [average_fit_time],
        labels = [vector_to_string(cluster_labels)],
        probabilities = [vector_to_string(cluster_probabilities)],
    )

    out_path = joinpath(JULIA_RESULT_DIR, "$(dataset_name)_results.csv")
    CSV.write(out_path, out)

    println("Saved $out_path")

end


function main()

    # Example of Usage: min_cluster_size, min_sample_size are positional
    model = Hdbscan(
        15, # min_cluster_size
        5;  # min_sample_size
        cluster_selection_epsilon = 0.0,
        max_cluster_size = nothing,
        metric = "euclidean",
        metric_params = Number[],
        alpha = 1.0,
        algorithm = "auto",
        leaf_size = 40,
        cluster_selection_method = "leaf",
        allow_single_cluster = false,
        copy = false,
    )

    generate_all_examples(model)

end

function generate_all_examples(model::Hdbscan)
    for (dataset_name, hasheader) in DATASETS
        generate_results(model, dataset_name, hasheader; n = 20)
    end
end

main()
