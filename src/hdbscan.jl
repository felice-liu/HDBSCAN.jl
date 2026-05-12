module hdbscan

    import Pkg;
    Pkg.add("Missings")
    using Missing
    using Distances

    import _linkage:
        MST_edge_dtype,
        make_single_linkage,
        mst_from_data_matrix,
        mst_from_mutual_reachability
    import _reachability:
        mutual_reachability_graph
    import _tree:
        HIERARCHY_dtype,
        labelling_at_cut,
        tree_to_labels
    import sparse:
        csgraph,
        issparse
    
    import base:
        BaseEstimator,
        ClusterMixin,
        _fit_context
    import metrics:
        pairwise_distances,
        DistanceMetric,
        _VALID_METRICS
    import neighbours:
        BallTree,
        KDTree,
        NearestNeighbors
    import _param_validation:
        Hidden,
        Interval,
        StrOptions
    import validation:
        _allclose_dense_sparse,
        _assert_all_finite,
        validate_data

    export remap_single_linkage_tree,
        HDBSCAN,
        fit,
        fit_predict,
        dbscan_clustering

    FAST_METRICS = set(KDTree.valid_metrics + BallTree.valid_metrics)

    struct encoding
        label::Int
        prob::Int
    end

    _OUTLIER_ENCODING = dict(
        "infinite" => encoding(-2, 0),
        "missing" => encoding(-3, nothing))

    

    function _brute_mst(mutual_reachability, min_samples)
    end

    function _process_mst(min_spanning_tree)
    end

    function _hdbscan_brute(X, min_samples=5, alpha=nothing,
    metric="euclidean", n_jobs=nothing, copy=False, metric_params...)
    end

    function _hdbscan_prims(X, algo, min_samples=5, alpha=1.0,
    metric="euclidean", leaf_size=40, n_jobs=None, metric_params...)
    end

    function remap_single_linkage_tree(tree, internal_to_raw, non_finite)
    end

    function _get_finite_row_indices(matrix)
    end

    struct base_estimator
    end

    struct cluster_mixin
    end

    struct hdbscan_state
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        max_cluster_size,
        metric::String,
        metric_params=None,
        alpha=1.0,
        algorithm::String,
        leaf_size=40,
        n_jobs=None,
        cluster_selection_method::String,
        allow_single_cluster:Bool,
        store_centers,
        copy::String,
        estimator_state::base_estimator,
        mixin_state::cluster_mixin
    end
#=
    function HDBSCAN()
        hdbscan_state(5, nothing, 0.0, nothing, "euclidean", nothing, 1.0,
        "auto", 40, nothing, "eom", false, nothing, "warn", nothing, nothing)
    end

    #inheritance of ClusterMixin e BaseEstimator

    function HDBSCAN(ClusterMixin, BaseEstimator)
    end

    function fit(self, X, y=nothing)
    end

    function fit_predict(self, X, y=nothing)
    end

    function _weighted_cluster_center(self, X)
    end

    function dbscan_clustering(self, cut_distance, min_cluster_size=5)
    end

    =#
end