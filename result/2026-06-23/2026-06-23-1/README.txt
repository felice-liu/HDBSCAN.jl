Validation comparison vs HDBSCAN Python vs HDBSCAN Julia
parameters used With scalers and imputer on real data
HDBSCAN(
    5,             # min_cluster_size
    1,              # min_samples
    0.0,            # cluster_selection_epsilon
    nothing,        # max_cluster_size
    "euclidean",    # metric
    Dict(),         # metric_params
    1.0,            # alpha
    "auto",         # algorithm
    40,             # leaf_size
    nothing,        # n_jobs
    "eom",          # cluster_selection_method
    true,           # allow_single_cluster
    nothing,        # store_centers
    true            # copy)