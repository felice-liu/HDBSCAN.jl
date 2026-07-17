# This file can be run though REPL. It can accept hyperparameters in the command 
# line and generate files in hdbscan\result\julia by fitting said hypermarameters
# for the 11 datasets in hdbscan\data (as of now can't be modified).

using CSV
using DataFrames
using Statistics
using ArgParse
using BenchmarkTools


const BENCHMARK_DIR = @__DIR__
const ROOT_DIR = normpath(joinpath(BENCHMARK_DIR, ".."))
const DATA_DIR = joinpath(ROOT_DIR, "data")
const SRC_DIR = joinpath(ROOT_DIR, "src")
const RESULT_DIR = joinpath(ROOT_DIR, "result")
const JULIA_RESULT_DIR = joinpath(RESULT_DIR, "julia")

using HDBSCAN

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

    b = @benchmark begin
        fit!($model, $X)
    end

    cluster_labels = labels(model)
    cluster_probabilities = probabilities(model)

    out = DataFrame(
        dataset = [dataset_name],
        average_fit_time_sec = [mean(b).time / 1e9],
        labels = [vector_to_string(cluster_labels)],
        probabilities = [vector_to_string(cluster_probabilities)],
    )

    out_path = joinpath(JULIA_RESULT_DIR, "$(dataset_name)_results.csv")
    CSV.write(out_path, out)

    println("Saved $out_path")

end

function parse_commandline()

    settings = ArgParseSettings()

    @add_arg_table! settings begin

        "--min_cluster_size"
        help = "Minimum cluster size"
        arg_type = Int
        required = true

        "--min_samples"
        help = "Minimum samples"
        arg_type = Int
        required = true

        "--cluster_selection_epsilon"
        arg_type = Float64
        default = 0.0

        "--max_cluster_size"
        arg_type = Int
        default = -1

        "--metric"
        arg_type = String
        default = "euclidean"

        "--alpha"
        arg_type = Float64
        default = 1.0

        "--algorithm"
        arg_type = String
        default = "auto"

        "--leaf_size"
        arg_type = Int
        default = 40

        "--cluster_selection_method"
        arg_type = String
        default = "eom"

        "--allow_single_cluster"
        action = :store_true

        "--copy"
        action = :store_true

    end

    return parse_args(settings)

end

function main()

    args = parse_commandline()

    model = HDBSCAN(
        args["min_cluster_size"],
        args["min_samples"];
        cluster_selection_epsilon = args["cluster_selection_epsilon"],
        max_cluster_size = args["max_cluster_size"] == -1 ? nothing :
                           args["max_cluster_size"],
        metric = args["metric"],
        metric_params = Number[],
        alpha = args["alpha"],
        algorithm = args["algorithm"],
        leaf_size = args["leaf_size"],
        cluster_selection_method = args["cluster_selection_method"],
        allow_single_cluster = args["allow_single_cluster"],
        copy = args["copy"],
    )

    generate_all_examples(model)

end

function generate_all_examples(model::Hdbscan)
    for (dataset_name, hasheader) in DATASETS
        generate_results(model, dataset_name, hasheader; n = 1)
    end
end

main()
