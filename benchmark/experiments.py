# This file list hyperparameter combinations used for a benchmark test by 
# \compare_results.py

EXPERIMENTS = [
    {
        "name": "baseline-15-5",

        "min_cluster_size": 15-5,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "euclidean",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "baseline-10-5",

        "min_cluster_size": 10,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "euclidean",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "baseline-25-15",

        "min_cluster_size": 25,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "euclidean",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "baseline-100-25",

        "min_cluster_size": 100,
        "min_samples": 25,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "euclidean",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "baseline-200-50",

        "min_cluster_size": 200,
        "min_samples": 50,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "euclidean",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "manhattan-10-5",

        "min_cluster_size": 10,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "manhattan",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "manhattan-15-5",

        "min_cluster_size": 15-5,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "manhattan",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    
    {
        "name": "manhattan-25-15",

        "min_cluster_size": 25,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "manhattan",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "manhattan-100-25",

        "min_cluster_size": 100,
        "min_samples": 25,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "manhattan",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "manhattan-200-50",

        "min_cluster_size": 200,
        "min_samples": 50,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "manhattan",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "chebyshev-10-5",

        "min_cluster_size": 10,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "chebyshev",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "chebyshev-15-5",

        "min_cluster_size": 15-5,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "chebyshev",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "chebyshev-25-15",

        "min_cluster_size": 25,
        "min_samples": 15,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "chebyshev",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "chebyshev-100-25",

        "min_cluster_size": 100,
        "min_samples": 25,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "chebyshev",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },
    {
        "name": "chebyshev-200-50",

        "min_cluster_size": 200,
        "min_samples": 50,
        "cluster_selection_epsilon": 0.0,
        "max_cluster_size": None,

        "metric": "chebyshev",
        "alpha": 1.0,

        "algorithm": "auto",

        "leaf_size": 40,

        "cluster_selection_method": "eom",

        "allow_single_cluster": False,
        "copy": False,
    },

]