using Test
using Random
using Statistics
using Distances
using Hdbscan


const OUTLIER_SET = Set([
    -1,
    Hdbscan._OUTLIER_ENCODING["infinite"].label,
    Hdbscan._OUTLIER_ENCODING["missing"].label,
])

# Equivalent of StandardScaler
function standardize(X::Matrix{Float64})
    u = mean(X, dims = 1)
    o = std(X, dims = 1)

    o[o .== 0.0] .= 1.0

    return (X .- u) ./ o
end

# Generates an example dataset
function make_blobs(; n_samples = 200, centers = 3, cluster_std = 1.0, seed = 10)

    Random.seed!(seed)

    if centers isa Integer
        centers = randn(centers, 2) .* 8
    end

    n_clusters = size(centers, 1)

    X = Matrix{Float64}(undef, n_samples, 2)
    y = Vector{Int}(undef, n_samples)

    per_cluster = fill(div(n_samples, n_clusters), n_clusters)

    for i = 1:rem(n_samples, n_clusters)
        per_cluster[i] += 1
    end

    idx = 1

    for c = 1:n_clusters
        for _ = 1:per_cluster[c]
            X[idx, :] .= centers[c, :] .+ cluster_std .* randn(2)
            y[idx] = c - 1           # sklearn labels start from 0
            idx += 1
        end
    end

    return X, y
end

function shuffle_data(X, y; seed = 7)
    Random.seed!(seed)
    p = randperm(length(y))
    return X[p, :], y[p]
end

# Dataset

X, y = make_blobs(n_samples = 200, seed = 10)

X, y = shuffle_data(X, y, seed = 7)

X = standardize(X)


@testset "No Clusters" begin
    # Tests that HDBSCAN correctly does not generate a valid cluster when the
    # min_cluster_size is too large for the data.

    labels = fit_predict(Hdbscan((size(X, 1)-1), nothing; copy = false), X)

    @test all(label in OUTLIER_SET for label in labels)

end


@testset "Minimum Cluster Size" begin
    # Test that the smallest non-noise cluster has at least "min_cluster_size"
    # many points
    for min_cluster_size in (2, 3, 5, 10, 20, 50, 100, 150,)

        labels = fit_predict(Hdbscan(min_cluster_size, nothing, copy = false), X)

        cluster_labels = filter(!=(-1), labels)

        if !isempty(cluster_labels)

            counts = Dict{Int,Int}()

            for c in cluster_labels
                counts[c] = get(counts, c, 0)+1
            end

            @test minimum(values(counts)) >= min_cluster_size

        end
    end

end

@testset "Precomputed Tree Algorithms" begin
    # Tests that HDBSCAN correctly raises an error when passing precomputed data
    # while requesting a tree-based algorithm.

    for algo in ("kd_tree", "ball_tree")

        model = Hdbscan(5, nothing, metric = "precomputed", algorithm = algo, copy = false)

        @test_throws ArgumentError fit!(model, X)

    end

end


@testset "Too Many min_samples" begin
    # Tests that HDBSCAN correctly raises an error when setting "min_samples"
    # larger than the number of samples.

    model = Hdbscan(5, size(X, 1)+1, copy = false)

    @test_throws ArgumentError fit!(model, X)

end

@testset "Dense Precomputed NaN" begin
    # Tests that HDBSCAN correctly raises an error when providing precomputed
    # distances with "np.nan" values.
    D = pairwise(Euclidean(), X; dims = 1)

    D[1, 1] = NaN

    model = Hdbscan(5, nothing, metric = "precomputed", copy = false)

    @test_throws ArgumentError fit!(model, D)

end


@testset "Cluster Centers" begin
    # Tests that HDBSCAN centers are calculated and stored properly, and are
    # accurate to the data.

    centers = [(0.0, 0.0), (3.0, 3.0)]

    H, _ = make_blobs(
        n_samples = 2000,
        centers = [
            [0.0 0.0];
            [3.0 3.0]
        ],
        cluster_std = 0.5,
        seed = 0,
    )

    model = Hdbscan(5, nothing, store_centers = "both", copy = false)

    fit!(model, H)

    @test size(model.centroids_, 1) == 2
    @test size(model.medoids_, 1) == 2

    for i = 1:2

        @test isapprox(model.centroids_[i, :], collect(centers[i]), atol = 0.05, rtol = 1)

        @test isapprox(model.medoids_[i, :], collect(centers[i]), atol = 0.05, rtol = 1)

    end

end

@testset "_do_labelling Distinct Clusters" begin
    # Tests that the "_do_labelling" helper function correctly assigns labels.

    Xtest, ytest = make_blobs(
        n_samples = 48,
        centers = [
            [0.0 0.0];
            [10.0 0.0];
            [0.0 10.0]
        ],
        cluster_std = 0.5,
        seed = 1234,
    )

    est = Hdbscan(copy = false)

    fit!(est, Xtest)

    condensed = Hdbscan._condense_tree(est._single_linkage_tree, est.min_cluster_size)

    n_samples = size(Xtest, 1)

    clusters = Set([n_samples+2, n_samples+3, n_samples+4])

    cluster_label_map = Dict(n_samples+2 => 0, n_samples+3 => 1, n_samples+4 => 2)

    labels = Hdbscan._do_labelling(condensed, clusters, cluster_label_map, false, 0.0)

    first_index = Dict()

    for c in unique(ytest)
        first_index[c] = findfirst(==(c), ytest)
    end

    mapping = Dict()

    for c in unique(ytest)
        mapping[c] = labels[first_index[c]]
    end

    aligned = [mapping[c] for c in ytest]

    @test labels == aligned

end

@testset "_do_labelling Thresholding" begin
    # Tests that the _do_labelling helper function correctly thresholds the
    # incoming lambda values given various "cluster_selection_epsilon" values.

    max_lambda = 1.5
    n_samples = 5

    condensed = Hdbscan.CondensedTree[
        Hdbscan.CondensedTree(6, 3, max_lambda, 1),
        Hdbscan.CondensedTree(6, 2, 0.1, 1),
        Hdbscan.CondensedTree(6, 1, max_lambda, 1),
        Hdbscan.CondensedTree(6, 4, 0.2, 1),
        Hdbscan.CondensedTree(6, 5, 0.3, 1),
    ]

    labels = Hdbscan._do_labelling(condensed, Set([6]), Dict(6 => 0), true, 1.0)

    expected_noise = count(c -> c.value < 1.0, condensed)

    @test count(labels .== -1) == expected_noise

    labels = Hdbscan._do_labelling(condensed, Set([6]), Dict(6 => 0), true, 0.0)

    expected_noise = count(c -> c.value < max_lambda, condensed)

    @test count(labels .== -1) == expected_noise

end
