using CSV
using DataFrames
include(joinpath(@__DIR__, "hdbscan.jl"))

# If hdbscan.jl defines a module, uncomment and adjust:
# using .HDBSCANModuleName


# ============================================================
# PATHS
# ============================================================

const SRC_DIR = @__DIR__
const ROOT_DIR = normpath(joinpath(SRC_DIR, ".."))
const DATA_DIR = joinpath(ROOT_DIR, "data")
const RESULT_DIR = joinpath(ROOT_DIR, "result")
const JULIA_RESULT_DIR = joinpath(RESULT_DIR, "julia")

mkpath(JULIA_RESULT_DIR)


# ============================================================
# DATASET CONFIG
# ============================================================

const DATASET_CONFIGS = Dict(
    "circles" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "circles_dataset.csv"),
    ),
    "moons" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "moons_dataset.csv"),
    ),
    "varied" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "varied_dataset.csv"),
    ),
    "aniso" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "aniso_dataset.csv"),
    ),
    "blobs" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "blobs_dataset.csv"),
    ),
    "no_structure" => Dict(
        :kind => :preprocessed,
        :path => joinpath(DATA_DIR, "no_structure_dataset.csv"),
    ),
    "heartfailure" => Dict(
    :kind => :unprocessed,
    :path => joinpath(DATA_DIR, "heartfailure.csv"),
    :feature_cols => [
        "Age (years)",
        "Male (1=Yes, 0=No)",
        "PHQ-9",
        "Systolic BP (mm Hg)",
        "Estimated glomerular filtration rate",
        "Ejection fraction (%)",
        "Serum sodium (mmol/l)",
        "Blood urea nitrogen (mg/dl)",
        "Etiology HF(1=Yes, 0=No)",
        "Prior diabetes mellitus",
        "Elevated level of BNP/NT-BNP (1=Yes, 0=No)",
    ],
),

"cardiacarrest" => Dict(
    :kind => :unprocessed,
    :path => joinpath(DATA_DIR, "cardiacarrest.csv"),
    :feature_cols => [
        "sex_woman",
        "Age_years",
        "Endotracheal_intubation",
        "Functional_status",
        "Asystole",
        "Bystander",
        "Time_min",
        "Cardiogenic",
        "Cardiac_arrest_at_home",
    ],
),

"neuroblastoma" => Dict(
    :kind => :unprocessed,
    :path => joinpath(DATA_DIR, "neuroblastoma.csv"),
    :feature_cols => [
        "age",
        "sex",
        "site",
        "stage",
        "time_months",
        "autologous_stem_cell_transplantation",
        "radiation",
        "degree_of_differentiation",
        "UH_or_FH",
        "MYCN_status ",
        "surgical_methods",
    ],
),

"sepsis" => Dict(
    :kind => :unprocessed,
    :path => joinpath(DATA_DIR, "sepsis.csv"),
    :feature_cols => [
        "Age",
        "sex_woman",
        "diagnosis_0EC_1M_2_AC",
        "APACHE II",
        "SOFA",
        "CRP",
        "WBCC",
        "NeuC",
        "LymC",
        "EOC",
        "NLCR",
        "PLTC",
        "MPV",
        "LOS-ICU",
    ],
),

"type1diabetes" => Dict(
    :kind => :unprocessed,
    :path => joinpath(DATA_DIR, "type1diabetes.csv"),
    :feature_cols => [
        "age",
        "duration.of.diabetes",
        "body_mass_index",
        "TDD",
        "basal",
        "bolus",
        "HbA1c",
        "eGFR",
        "perc.body.fat",
        "adiponectin",
        "free.testosterone",
        "SMI",
        "grip.strength",
        "knee.extension.strength",
        "gait.speed",
        "ucOC",
        "OC",
        "weight_kg",
        "sex_0man_1woman",
    ],
),
)

const DATASETS = [
    "circles",
    "moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
    "heartfailure",
    "cardiacarrest",
    "neuroblastoma",
    "sepsis",
    "type1diabetes",

]


# ============================================================
# HELPERS
# ============================================================

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

function standard_scaler(X::Matrix{Float64})
    for j in 1:size(X, 2)
        col = view(X, :, j)
        u = mean(col)
        o = std(col)

        # match sklearn behavior enough for this use:
        # if a column is constant, leave it centered at 0
        if iszero(o) || isnan(o)
            col .= col .- u
        else
            col .= (col .- u) ./ o
        end
    end
    return X
end



function robust_scaler(X::Matrix{Float64})
    Xf = Float64.(X)
    X_scaled = similar(Xf)

    n_features = size(Xf, 2)

    for j in 1:n_features
        col = Xf[:, j]
        med = median(col)
        q1 = quantile(col, 0.25)
        q3 = quantile(col, 0.75)
        iqr = q3 - q1

        if iqr == 0
            X_scaled[:, j] .= col .- med
        else
            X_scaled[:, j] .= (col .- med) ./ iqr
        end
    end

    return X_scaled
end


function load_dataset(dataset_name::String)
    cfg = DATASET_CONFIGS[dataset_name]
    path = cfg[:path]

    if !isfile(path)
        error("Missing dataset CSV: $path")
    end

    if cfg[:kind] == :preprocessed
        df = CSV.read(path, DataFrame; header=false)
        return Matrix{Float64}(df)

    elseif cfg[:kind] == :unprocessed
        df = CSV.read(path, DataFrame)

        # keep only the feature columns
        Xdf = select(df, cfg[:feature_cols])

        # median-impute missing values in place
        fill_missing_median!(Xdf)

        # convert to numeric matrix
        X = Matrix{Float64}(Xdf)

        # Scaler
        # X = standard_scaler(X) #for standard scaling
        X = robust_scaler(X)

        return X

    else
        error("Unsupported dataset kind $(cfg[:kind])")
    end
end


function run_one(dataset_name::String, algo)

    println("Running Julia HDBSCAN on $dataset_name")
    X = load_dataset(dataset_name)

    t0 = time()
    fit(algo, X)
    fit_time = time() - t0

    labels = Int.(algo.labels_)
    probabilities = Float64.(algo.probabilities_)

    out = DataFrame(
        dataset = [dataset_name],
        fit_time_sec = [fit_time],
        labels = [vector_to_string(labels)],
        probabilities = [vector_to_string(probabilities)],
    )

    out_path = joinpath(JULIA_RESULT_DIR, "$(dataset_name)_hdbscan_julia.csv")
    CSV.write(out_path, out)

    println("Saved $out_path")
end

# MAIN

function main()

    parameters = init_HDBSCAN(
        15,              # min_cluster_size
        5,              # min_samples
        0.0,            # cluster_selection_epsilon
        nothing,        # max_cluster_size
        "euclidean",    # metric
        Dict(),         # metric_params
        1.0,            # alpha
        "auto",         # algorithm
        40,             # leaf_size
        nothing,        # n_jobs
        "eom",          # cluster_selection_method
        false,           # allow_single_cluster
        nothing,        # store_centers
        true            # copy
    )

    algo_name = "hdbscan"

    for dataset_name in DATASETS
        run_one(dataset_name, parameters)
    end
end

main()
#=

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println("Usage: julia src/julia_generate_results.jl <min_cluster_size> <min_samples>")
        exit(1)
    end

    inp_min_cluster_size = parse(Int, ARGS[1])
    inp_min_samples = parse(Int, ARGS[2])

    main(inp_min_cluster_size, inp_min_samples)
end
=#