#=
   HDBSCAN:
   Hierarchical Density-Based Spatial Clustering of Applications with Noise

   HDBSCAN is a clustering algorithm that Performs DBSCAN over varying epsilon
   values and integrates the result to find a clustering that gives the best
   stability over epsilon

   It returns a good clustering straight away with little or no parameter tuning.
   The primary parameter, minimum cluster size, is intuitive and easy to select.

   Based on the papers:

    McInnes L, Healy J. Accelerated Hierarchical Density Based Clustering In:
    2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE,
    pp 33-42. 2017

    R. Campello, D. Moulavi, and J. Sander, Density-Based Clustering Based on
    Hierarchical Density Estimates In: Advances in Knowledge Discovery and Data
    Mining, Springer, pp 160-172. 2013
=#

module Hdbscan

export Hdbscan, fit!, fit_predict

export labels, probabilities, centroids, medoids, single_linkage_tree, nclusters

using Statistics
using SparseArrays

using Missings
using Distances
using NearestNeighbors
using Graphs
using SimpleWeightedGraphs
using SortingAlgorithms

# Edit here to add your pythonpath to use python argsort (remember to remove it
# once you're done). It will use SIMD implementation.
# It will will give the same tie-breaker/edge-case order as scikitlearn.

pythonpath = ""

ENV["JULIA_PYTHONCALL_EXE"] = pythonpath

using PythonCall

const np = pyimport("numpy")


#=
    This library is based on scikit-learn python implementation version 1.8.0
    and it translates in Julia the 4 main code files found in the github page:

    scikit-learn / sklearn / cluster / _hdbscan /
        _tree.pyx
        _reachability.pyx
        _linkage.pyx
        hdbscan.py

    ref: https://github.com/scikit-learn/scikit-learn/tree/1.8.X/sklearn/cluster/_hdbscan
=#

# Trees

# 1-based indexing

mutable struct HierarchyTree
    left_node::Int
    right_node::Int
    value::Float64
    cluster_size::Int
end

# Effectively an edgelist encoding a parent/child pair, along with a value and
# the corresponding cluster_size in each row providing a tree structure.

mutable struct CondensedTree
    parent::Int
    child::Int
    value::Float64
    cluster_size::Int
end

function tree_to_labels(
    single_linkage_tree::Array{HierarchyTree},
    min_cluster_size::Int = 10,
    cluster_selection_method::String = "eom",
    allow_single_cluster::Bool = false,
    cluster_selection_epsilon::Float64 = 0.0,
    max_cluster_size::Union{Nothing,Int} = nothing,
)

    condensed_tree = _condense_tree(single_linkage_tree, min_cluster_size)

    labels, probabilities = _get_clusters(
        condensed_tree,
        _compute_stability(condensed_tree),
        cluster_selection_method,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,
    )

    return (labels, probabilities)
end

function bfs_from_hierarchy(hierarchy::Array{HierarchyTree}, bfs_root::Int)
    """
    Perform a breadth first search on a tree.
    """
    n_samples = length(hierarchy) + 1
    process_queue = Int[bfs_root]
    result = Int[]

    while !isempty(process_queue)

        append!(result, process_queue)
        process_queue = [x - n_samples for x in process_queue if x > n_samples]

        if !isempty(process_queue)
            next_queue = Int[]
            for node in process_queue
                push!(next_queue, hierarchy[node].left_node)
                push!(next_queue, hierarchy[node].right_node)
            end
            process_queue = next_queue
        end
    end

    return result
end

"""
    _condense_tree(hierarchy, min_cluster_size=10)

Condense a single linkage hierarchy by pruning clusters smaller than the
specified minimum cluster size.

This procedure is analogous to the *runt pruning* method described by
Stuetzle and produces a simplified hierarchy that is easier to analyze.
The condensed tree also records the lambda value at which individual points
leave a cluster, which is later used for stability analysis and cluster
selection.

# Arguments
- `hierarchy::Vector{HierarchyTree}`: Single linkage hierarchy.
- `min_cluster_size::Int=10`: Minimum number of samples required for a
  cluster to be retained in the condensed tree.

# Returns
- `Vector{CondensedTree}`: Condensed cluster tree represented as a list of
  parent-child relationships, where each entry stores the parent cluster,
  child cluster or point, the corresponding lambda value, and the size of the
  child cluster.
"""
function _condense_tree(hierarchy::Array{HierarchyTree}, min_cluster_size::Int = 10)

    root = 2 * length(hierarchy) + 1
    n_samples = length(hierarchy) + 1
    next_label = n_samples + 2

    node_list = bfs_from_hierarchy(hierarchy, root)

    relabel = zeros(Int, root)
    relabel[root] = n_samples + 1

    result_list = CondensedTree[]

    ignore = falses(root)

    for node in node_list
        if ignore[node] || node <= n_samples
            continue
        end

        children = hierarchy[node-n_samples]
        left = children.left_node
        right = children.right_node
        distance = children.value

        if distance > 0.0
            lambda_value::Float64 = 1 / distance
        else
            lambda_value = INFTY
        end

        if left > n_samples
            left_count = hierarchy[left-n_samples].cluster_size
        else
            left_count = 1
        end

        if right > n_samples
            right_count = hierarchy[right-n_samples].cluster_size
        else
            right_count = 1
        end

        if left_count >= min_cluster_size && right_count >= min_cluster_size
            relabel[left] = next_label

            next_label += 1
            push!(
                result_list,
                CondensedTree(relabel[node], relabel[left], lambda_value, left_count),
            )

            relabel[right] = next_label
            next_label += 1
            push!(
                result_list,
                CondensedTree(relabel[node], relabel[right], lambda_value, right_count),
            )

        elseif left_count < min_cluster_size && right_count < min_cluster_size
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node <= n_samples
                    push!(
                        result_list,
                        CondensedTree(relabel[node], sub_node, lambda_value, 1),
                    )
                end
                ignore[sub_node] = true
            end

            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node <= n_samples
                    push!(
                        result_list,
                        CondensedTree(relabel[node], sub_node, lambda_value, 1),
                    )
                end
                ignore[sub_node] = true
            end

        elseif left_count < min_cluster_size
            relabel[right] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node <= n_samples
                    push!(
                        result_list,
                        CondensedTree(relabel[node], sub_node, lambda_value, 1),
                    )
                end
                ignore[sub_node] = true
            end

        else
            relabel[left] = relabel[node]
            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node <= n_samples
                    push!(
                        result_list,
                        CondensedTree(relabel[node], sub_node, lambda_value, 1),
                    )
                end
                ignore[sub_node] = true
            end
        end

    end

    return result_list
end

function _compute_stability(condensed_tree::Array{CondensedTree})

    parents = [c.parent for c in condensed_tree]

    largest_child = maximum([c.child for c in condensed_tree])
    smallest_cluster = minimum(parents)
    num_clusters = maximum(parents) - smallest_cluster + 1

    largest_child = max(largest_child, smallest_cluster)
    births = fill(NaN, largest_child)

    for idx in eachindex(condensed_tree)
        condensed_node = condensed_tree[idx]
        births[condensed_node.child] = condensed_node.value
    end

    births[smallest_cluster] = 0.0

    result = zeros(Float64, num_clusters)

    for idx in eachindex(condensed_tree)
        condensed_node = condensed_tree[idx]
        parent = condensed_node.parent
        lambda_val = condensed_node.value
        cluster_size = condensed_node.cluster_size

        result_index = parent - smallest_cluster + 1
        result[result_index] =
            result[result_index] + ((lambda_val - births[parent]) * cluster_size)
    end

    stability_dict = Dict()

    for idx = 1:num_clusters
        stability_dict[idx+smallest_cluster-1] = result[idx]
    end

    return stability_dict
end

function bfs_from_cluster_tree(condensed_tree::Array{CondensedTree}, bfs_root::Int)

    result = Int[]
    process_queue = [bfs_root]

    children = [c.child for c in condensed_tree]
    parents = [c.parent for c in condensed_tree]

    while !isempty(process_queue)
        append!(result, process_queue)
        process_queue =
            [children[i] for i in eachindex(children) if parents[i] in process_queue]
    end

    return result
end

function max_lambdas(condensed_tree::Array{CondensedTree})

    largest_parent = maximum([c.parent for c in condensed_tree])
    deaths = zeros(Float64, largest_parent)

    current_parent = condensed_tree[1].parent
    max_lambda = condensed_tree[1].value

    for i = 2:length(condensed_tree)
        parent = condensed_tree[i].parent
        lambda_val = condensed_tree[i].value

        if parent == current_parent
            max_lambda = max(max_lambda, lambda_val)
        else
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_val
        end
    end

    deaths[current_parent] = max_lambda # value for last parent

    return deaths
end

mutable struct TreeUnionFind
    data::Array{Int,2}
    is_component::Vector{Bool}
end

function TreeUnionFind(size::Int)

    data = zeros(Int, size, 2)

    for i = 1:size
        data[i, 1] = i
    end

    is_component = trues(size)
    tuf = TreeUnionFind(data, is_component)

    return tuf
end

function tuf_union!(tuf::TreeUnionFind, x::Int, y::Int)

    x_root = tuf_find!(tuf, x)
    y_root = tuf_find!(tuf, y)

    if tuf.data[x_root, 2] < tuf.data[y_root, 2]
        tuf.data[x_root, 1] = y_root
    elseif tuf.data[x_root, 2] > tuf.data[y_root, 2]
        tuf.data[y_root, 1] = x_root
    else
        tuf.data[y_root, 1] = x_root
        tuf.data[x_root, 2] += 1
    end

end

function tuf_find!(tuf::TreeUnionFind, x::Int)

    if tuf.data[x, 1] != x
        tuf.data[x, 1] = tuf_find!(tuf, tuf.data[x, 1])
        tuf.is_component[x] = false
    end

    return tuf.data[x, 1]
end

"""
    labelling_at_cut(linkage, cut, min_cluster_size)

Return the cluster labels obtained by cutting a single linkage tree at the
specified distance threshold.

# Arguments
- `linkage::Vector{HierarchyTree}`: Single linkage hierarchy.
- `cut::Real`: Distance threshold.
- `min_cluster_size::Int`: Minimum cluster size. Smaller clusters are
  labelled as noise.

# Returns
- `Vector{Int}`: Cluster labels, where `-1` denotes noise.
"""
function labelling_at_cut(
    linkage::Array{HierarchyTree},
    cut::Float64,
    min_cluster_size::Int,
)

    # Given a single linkage tree and a cut value, return the
    # vector of cluster labels at that cut value. This is useful
    # for Robust Single Linkage, and extracting DBSCAN results
    # from a single HDBSCAN run.

    root = 2 * length(linkage) + 1
    n_samples = length(linkage) + 1

    result = zeros(Int, n_samples)
    tuf = TreeUnionFind(root + 1)

    cluster = n_samples + 1

    for node in linkage

        if node.value < cut
            tuf_union!(tuf, node.left_node, cluster)
            tuf_union!(tuf, node.right_node, cluster)
        end

        cluster += 1
    end

    cluster_size = zeros(Int, cluster)

    for n = 1:n_samples
        cluster = tuf_find!(tuf, n)
        cluster_size[cluster] += 1
        result[n] = cluster
    end

    cluster_label_map = Dict(-1 => NOISE)
    cluster_label = 0

    unique_labels = unique(result)

    for cluster in unique_labels

        if cluster_size[cluster] < min_cluster_size
            cluster_label_map[cluster] = NOISE
        else
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1
        end

    end

    for n = 1:n_samples
        result[n] = cluster_label_map[result[n]]
    end

    return result
end

"""
    _do_labelling(condensed_tree, clusters, cluster_label_map,
                  allow_single_cluster, cluster_selection_epsilon)

Assign a cluster label to each sample based on the selected clusters in the
condensed tree.

Samples that do not belong to any selected cluster are labelled as noise.
For datasets containing a single large cluster, the assignment of border
points to noise is influenced by the `allow_single_cluster` and
`cluster_selection_epsilon` parameters.

# Arguments
- `condensed_tree::Vector{CondensedTree}`: Condensed cluster hierarchy.
- `clusters`: Set of cluster nodes selected during cluster selection.
- `cluster_label_map::Dict`: Mapping from cluster node identifiers to the
  output cluster labels.
- `allow_single_cluster::Bool`: Whether the root cluster may be selected as
  the only cluster.
- `cluster_selection_epsilon::Real`: Distance threshold used during cluster
  selection.

# Returns
- `Vector{Int}`: Cluster label for each sample. A label of `-1` indicates
  that the sample is classified as noise.
"""
function _do_labelling(
    condensed_tree::Array{CondensedTree},
    clusters,
    cluster_label_map::Dict,
    allow_single_cluster::Bool,
    cluster_selection_epsilon::Float64,
)

    child_array = [c.child for c in condensed_tree]
    parent_array = [c.parent for c in condensed_tree]
    lambda_array = [c.value for c in condensed_tree]

    root_cluster = minimum(parent_array)
    n_samples = root_cluster - 1
    result = fill(NOISE, n_samples)

    max_label = max(maximum(parent_array), maximum(child_array))
    tuf = TreeUnionFind(max_label)

    for i in eachindex(condensed_tree)
        child = child_array[i]
        parent = parent_array[i]

        if !(child in clusters)
            child_root = tuf_find!(tuf, child)
            parent_root = tuf_find!(tuf, parent)
            tuf.data[child_root, 1] = parent_root
        end

    end

    parent_map = Dict{Int,Int}()

    for i in eachindex(condensed_tree)
        parent_map[child_array[i]] = parent_array[i]
    end

    for n = 1:n_samples
        cluster = tuf_find!(tuf, n)
        label = NOISE

        while cluster != root_cluster &&
                  !haskey(cluster_label_map, cluster) &&
                  haskey(parent_map, cluster)

            cluster = parent_map[cluster]
        end

        if cluster != root_cluster && haskey(cluster_label_map, cluster)
            label = cluster_label_map[cluster]

        elseif length(clusters) == 1 && allow_single_cluster
            # There can only be one edge with this particular child hence this
            # expression extracts a unique, scalar lambda value.
            parent_lambda =
                [lambda_array[i] for i in eachindex(child_array) if child_array[i] == n]

            threshold = if cluster_selection_epsilon != 0.0
                1 / cluster_selection_epsilon
            else
                # The threshold should be calculated per-sample based on the
                # largest lambda of any simbling node.
                maximum([
                    lambda_array[i] for
                    i in eachindex(parent_array) if parent_array[i] == root_cluster
                ])
            end

            if !isempty(parent_lambda) && maximum(parent_lambda) >= threshold

                if haskey(cluster_label_map, root_cluster)
                    label = cluster_label_map[root_cluster]
                end
            end
        end

        result[n] = label
    end

    return result
end

function get_probabilities(
    condensed_tree::Array{CondensedTree},
    cluster_map::Dict,
    labels::Array{Int},
)

    child_array = [c.child for c in condensed_tree]
    parent_array = [c.parent for c in condensed_tree]
    lambda_array = [c.value for c in condensed_tree]

    result = zeros(Float64, length(labels))
    deaths = max_lambdas(condensed_tree)
    root_cluster = minimum(parent_array)

    for i in eachindex(condensed_tree)

        point = child_array[i]

        if point >= root_cluster
            continue
        end

        cluster_num = labels[point]

        if cluster_num == NOISE
            continue
        end

        cluster = cluster_map[cluster_num]
        max_lambda = deaths[cluster]

        if max_lambda == 0.0 || isinf(lambda_array[i])
            result[point] = 1.0
        else
            lambda_val = min(lambda_array[i], max_lambda)
            result[point] = lambda_val / max_lambda
        end
    end

    return result
end

function recurse_leaf_dfs(cluster_tree::Array{CondensedTree}, current_node::Int)

    children = [c.child for c in cluster_tree if c.parent == current_node]

    if isempty(children)

        return [current_node]

    else

        result = []

        for child in children
            append!(result, recurse_leaf_dfs(cluster_tree, child))
        end

        return result
    end
end

function get_cluster_tree_leaves(cluster_tree::Array{CondensedTree})

    if isempty(cluster_tree)
        return []
    end

    root = minimum([c.parent for c in cluster_tree])

    return recurse_leaf_dfs(cluster_tree, root)
end

function traverse_upwards(
    cluster_tree::Array{CondensedTree},
    cluster_selection_epsilon::Float64,
    leaf::Int,
    allow_single_cluster::Bool,
)

    root = minimum([c.parent for c in cluster_tree])
    parent = only([c.parent for c in cluster_tree if c.child == leaf])

    if parent == root
        if allow_single_cluster
            return parent
        else
            return leaf # return node closest to root
        end
    end

    parent_eps = 1 / only([c.value for c in cluster_tree if c.child == parent])

    if parent_eps > cluster_selection_epsilon
        return parent
    else
        return traverse_upwards(
            cluster_tree,
            cluster_selection_epsilon,
            parent,
            allow_single_cluster,
        )
    end
end

function epsilon_search(
    leaves::Set,
    cluster_tree::Array{CondensedTree},
    cluster_selection_epsilon::Float64,
    allow_single_cluster::Bool,
)

    selected_clusters = []
    processed = []

    children = [c.child for c in cluster_tree]
    distances = [c.value for c in cluster_tree]

    for leaf in leaves
        first = findfirst(==(leaf), children)
        eps = 1 / distances[first]

        if eps < cluster_selection_epsilon

            if !(leaf in processed)
                epsilon_child = traverse_upwards(
                    cluster_tree,
                    cluster_selection_epsilon,
                    leaf,
                    allow_single_cluster,
                )
                push!(selected_clusters, epsilon_child)

                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child)

                    if sub_node != epsilon_child
                        push!(processed, sub_node)
                    end

                end
            end
        else
            push!(selected_clusters, leaf)
        end
    end

    return Set(selected_clusters)
end

"""
    _get_clusters(condensed_tree, stability, cluster_selection_method,
                  allow_single_cluster, cluster_selection_epsilon,
                  max_cluster_size)

Select the final clusters from a condensed cluster tree and compute the
corresponding cluster labels and membership probabilities.

Clusters are selected using either the Excess of Mass (`"eom"`) or
leaf-based (`"leaf"`) cluster selection method.

# Arguments
- `condensed_tree::Vector{CondensedTree}`: Condensed cluster hierarchy.
- `stability::Dict{Int, Float64}`: Mapping from cluster identifiers to
  their stability values.
- `cluster_selection_method::String="eom"`: Cluster selection method.
  Supported values are `"eom"` and `"leaf"`.
- `allow_single_cluster::Bool=false`: Whether the root cluster may be
  selected as the only cluster.
- `cluster_selection_epsilon::Real=0.0`: Distance threshold used during
  cluster selection.
- `max_cluster_size::Union{Nothing, Int}=nothing`: Maximum size of a
  cluster selected by the EOM algorithm.

# Returns
A tuple `(labels, probabilities)` where:
- `labels::Vector{Int}` contains the cluster label assigned to each
  sample, with `-1` denoting noise.
- `probabilities::Vector{Float64}` contains the membership strength of
  each sample in its assigned cluster.
"""
function _get_clusters(
    condensed_tree::Array{CondensedTree},
    stability::Dict,
    cluster_selection_method = "eom",
    allow_single_cluster = false,
    cluster_selection_epsilon = 0.0,
    max_cluster_size::Union{Nothing,Int} = nothing,
)
    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    if allow_single_cluster
        node_list = sort(collect(keys(stability)), rev = true)
    else
        node_list = sort(collect(keys(stability)), rev = true)[1:(end-1)]
    end

    cluster_tree = [c for c in condensed_tree if c.cluster_size > 1]
    is_cluster = Dict(c => true for c in node_list)

    n_samples = maximum([c.child for c in condensed_tree if c.cluster_size == 1])

    if isnothing(max_cluster_size)
        max_cluster_size = n_samples + 1 # Set to a value that will never be triggered
    end

    cluster_sizes = Dict(c.child => c.cluster_size for c in cluster_tree)

    if allow_single_cluster
        root = node_list[end]
        cluster_sizes[root] =
            sum([c.cluster_size for c in cluster_tree if c.parent == root])
    end

    if cluster_selection_method == "eom"

        for node in node_list

            children = [c.child for c in cluster_tree if c.parent == node]

            if isempty(children)
                subtree_stability = 0.0
            else
                subtree_stability = sum(stability[c] for c in children)
            end

            if subtree_stability > stability[node] || cluster_sizes[node] > max_cluster_size

                is_cluster[node] = false
                stability[node] = subtree_stability

            else

                for sub_node in bfs_from_cluster_tree(cluster_tree, node)
                    if sub_node != node
                        is_cluster[sub_node] = false
                    end
                end
            end
        end

        if cluster_selection_epsilon != 0.0 && !isempty(cluster_tree)

            eom_clusters = [c for c in keys(is_cluster) if is_cluster[c]]
            # first check if eom_clusters only has root node, which skips epsilon check.
            if length(eom_clusters) == 1 &&
               eom_clusters[1] == minimum([c.parent for c in cluster_tree])

                selected_clusters = allow_single_cluster ? eom_clusters : Int[]

            else

                selected_clusters = epsilon_search(
                    Set(eom_clusters),
                    cluster_tree,
                    cluster_selection_epsilon,
                    allow_single_cluster,
                )

            end

            for c in keys(is_cluster)
                is_cluster[c] = c in selected_clusters
            end

        end

    elseif cluster_selection_method == "leaf"

        leaves = Set(get_cluster_tree_leaves(cluster_tree))

        if isempty(leaves)

            for c in keys(is_cluster)
                is_cluster[c] = false
            end

            is_cluster[minimum([c.parent for c in condensed_tree])] = true

        end

        if cluster_selection_epsilon != 0.0

            selected_clusters = epsilon_search(
                leaves,
                cluster_tree,
                cluster_selection_epsilon,
                allow_single_cluster,
            )

        else
            selected_clusters = leaves
        end

        for c in keys(is_cluster)
            is_cluster[c] = c in selected_clusters
        end
    end

    clusters = Set([c for c in keys(is_cluster) if is_cluster[c]])

    cluster_map = Dict(c => i - 1 for (i, c) in enumerate(sort(collect(clusters))))

    reverse_cluster_map = Dict(v => k for (k, v) in cluster_map)

    labels = _do_labelling(
        condensed_tree,
        clusters,
        cluster_map,
        allow_single_cluster,
        cluster_selection_epsilon,
    )

    probs = get_probabilities(condensed_tree, reverse_cluster_map, labels)

    return (labels, probs)

end

# linkage

struct MSTEdge
    current_node::Int64
    next_node::Int64
    distance::Float64
end

mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
    next_label::Int
end

function UnionFind(n::Int)

    parent = zeros(Int, 2n - 1)
    size = vcat(ones(Int, n), zeros(Int, n - 1))
    next_label = n + 1

    return UnionFind(parent, size, next_label)

end

function uf_find!(uf::UnionFind, x::Int)

    p = x

    while uf.parent[p] != 0
        p = uf.parent[p]
    end

    y = x

    while uf.parent[y] != 0 && uf.parent[y] != p
        old = y
        y = uf.parent[y]
        uf.parent[old] = p
    end

    return p

end

function uf_union!(uf::UnionFind, m::Int, n::Int)

    new_label = uf.next_label

    uf.parent[m] = new_label
    uf.parent[n] = new_label
    uf.size[new_label] = uf.size[m] + uf.size[n]
    uf.next_label += 1

end

"""
    mst_from_mutual_reachability(mutual_reachability)

Compute the Minimum Spanning Tree (MST) of the mutual reachability graph
using Prim's algorithm.

# Arguments
- `mutual_reachability::Matrix{Float64}`: Matrix containing the mutual
  reachability distances between all pairs of samples.

# Returns
- `Vector{MST_edge}`: Minimum spanning tree represented as a collection of
  edges connecting all samples with minimum total weight.
"""
function mst_from_mutual_reachability(mutual_reachability::Matrix{Float64})

    n_samples = size(mutual_reachability, 1)
    mst = Vector{MSTEdge}(undef, n_samples - 1)

    current_labels = collect(1:n_samples)
    current_node = 1

    min_reachability = fill(Inf, n_samples)

    for i = 1:(n_samples-1)
        label_filter = current_labels .!= current_node
        current_labels = current_labels[label_filter]

        left = min_reachability[label_filter]
        right = mutual_reachability[current_node, current_labels]

        min_reachability = min.(left, right)

        new_node_index = argmin(min_reachability)
        new_node = current_labels[new_node_index]

        mst[i] = MSTEdge(current_node, new_node, min_reachability[new_node_index])

        current_node = new_node
    end

    return mst

end

"""
    mst_from_data_matrix(raw_data, core_distances, dist_metric; alpha=1.0)

Compute the Minimum Spanning Tree (MST) of the mutual reachability graph
constructed from the input data using Prim's algorithm.

The mutual reachability graph is computed implicitly from the input data,
the corresponding core distances, and the selected distance metric, without
explicitly constructing the full graph.

# Arguments
- `raw_data::Matrix{Float64}`: Matrix whose rows correspond to data samples.
- `core_distances::Vector{Float64}`: Core distance associated with each
  sample.
- `dist_metric`: Distance metric used to compute pairwise distances between
  samples.
- `alpha::Real=1.0`: Scaling factor applied to pairwise distances before
  computing the mutual reachability distance.

# Returns
- `Vector{MST_edge}`: Minimum spanning tree represented as a collection of
  weighted edges.
"""
function mst_from_data_matrix(
    raw_data::Matrix{Float64},
    core_distances::Vector{Float64},
    dist_metric,
    alpha::Float64 = 1.0,
)

    n_samples = size(raw_data, 1)
    mst = Vector{MSTEdge}(undef, n_samples - 1)

    in_tree = falses(n_samples)
    min_reachability = fill(Inf, n_samples)
    current_sources = ones(Int, n_samples)

    current_node = 1

    for i = 1:(n_samples-1)

        in_tree[current_node] = true
        current_node_core_dist = core_distances[current_node]
        new_reachability = Inf
        source_node = 1
        new_node = 1

        for j = 1:n_samples

            if in_tree[j]
                continue
            end

            next_node_min_reach = min_reachability[j]
            next_node_source = current_sources[j]

            pair_distance =
                Distances.evaluate(
                    dist_metric,
                    view(raw_data, current_node, :),
                    view(raw_data, j, :),
                ) / alpha


            next_node_core_dist = core_distances[j]

            mutual_reachability_distance =
                max(current_node_core_dist, next_node_core_dist, pair_distance)

            # If MRD(i, j) is smaller than node j's min_reachability, we update
            # node j's min_reachability for future reference.
            if mutual_reachability_distance < next_node_min_reach

                min_reachability[j] = mutual_reachability_distance
                current_sources[j] = current_node

                # If MRD(i, j) is also smaller than node i's current
                # min_reachability, we update and set their edge as the current
                # MST edge candidate.
                if mutual_reachability_distance < new_reachability

                    new_reachability = mutual_reachability_distance
                    source_node = current_node
                    new_node = j

                end
                # If the node j is closer to another node already in the tree, we
                # make their edge the current MST candidate edge.
            elseif next_node_min_reach < new_reachability

                new_reachability = next_node_min_reach
                source_node = next_node_source
                new_node = j

            end

        end

        mst[i] = MSTEdge(source_node, new_node, new_reachability)
        current_node = new_node

    end

    return mst

end

"""
    make_single_linkage(mst)

Construct a single linkage hierarchy from a Minimum Spanning Tree (MST).

The hierarchy is represented as a dendrogram in which each merge records
the two merged nodes or clusters, the distance at which the merge occurs,
and the size of the newly formed cluster.

# Arguments
- `mst::Vector{MST_edge}`: Minimum spanning tree represented as a collection
  of weighted edges.

# Returns
- `Vector{HierarchyTree}`: Single linkage hierarchy. Each element stores:
  - the left child node or cluster,
  - the right child node or cluster,
  - the merge distance,
  - the size of the newly formed cluster.
"""
function make_single_linkage(mst::Vector{MSTEdge})
    # Note length(mst) is one fewer than the number of samples
    n_samples = length(mst) + 1

    single_linkage = Vector{HierarchyTree}(undef, n_samples - 1)

    uf = UnionFind(n_samples)

    for i = 1:(n_samples-1)

        current_node = mst[i].current_node
        next_node = mst[i].next_node
        distance = mst[i].distance

        current_node_cluster = uf_find!(uf, current_node)
        next_node_cluster = uf_find!(uf, next_node)

        single_linkage[i] = HierarchyTree(
            current_node_cluster,
            next_node_cluster,
            distance,
            uf.size[current_node_cluster] + uf.size[next_node_cluster],
        )

        uf_union!(uf, current_node_cluster, next_node_cluster)

    end

    return single_linkage

end

# reachability

"""
    mutual_reachability_graph(distance_matrix, min_samples=5;
                              max_distance=0.0)

Compute the weighted adjacency matrix of the mutual reachability graph.

The mutual reachability distance between two samples `xp` and `xq` is defined
as

`max(d_core(x_p), d_core(x_q), d(x_p, x_q))`

where `d_core` is the distance from a sample to its `min_samples`-th nearest
neighbor.

The computation is performed in-place whenever possible.

# Arguments
- `distance_matrix`: Pairwise distance matrix. Sparse matrices must be in
  CSR format.
- `min_samples::Int=5`: Number of nearest neighbors used to compute the
  core distance.
- `max_distance::Real=0.0`: Value used to replace infinite mutual
  reachability distances when `distance_matrix` is sparse.

# Returns
- A dense or sparse weighted adjacency matrix representing the mutual
  reachability graph.

# References
- Campello, R. J., Moulavi, D., & Sander, J. (2013). *Density-based
  clustering based on hierarchical density estimates*. Pacific-Asia
  Conference on Knowledge Discovery and Data Mining, 160-172.
"""
function mutual_reachability_graph(
    distance_matrix;
    min_samples::Int = 5,
    max_distance::Float64 = 0.0,
)

    further_neighbor_idx = min_samples - 1

    if issparse(distance_matrix)

        if !(distance_matrix isa SparseMatrixCSC)

            throw(ArgumentError("Only sparse CSC matrices
                   are supported for `distance_matrix`."))

        end

        _sparse_mutual_reachability_graph!(
            distance_matrix.nzval,
            distance_matrix.rowval,
            distance_matrix.colptr,
            size(distance_matrix, 1),
            further_neighbor_idx,
            max_distance,
        )

    else

        _dense_mutual_reachability_graph!(distance_matrix, further_neighbor_idx)

    end

    return distance_matrix

end

"""
    _dense_mutual_reachability_graph!(distance_matrix, further_neighbor_idx)

Compute the mutual reachability graph for a dense distance matrix.

This is the dense implementation of the mutual reachability graph
construction. The computation is performed in-place by modifying
`distance_matrix` directly.

# Arguments
- `distance_matrix`: Pairwise distance matrix between
  samples.
- `further_neighbor_idx::Int`: Index of the furthest nearest neighbor
  used to compute the core distance of each sample.

# Returns
- The modified distance matrix containing the mutual reachability
  distances.
"""
function _dense_mutual_reachability_graph!(distance_matrix, further_neighbor_idx::Int)

    n_samples = size(distance_matrix, 1)

    core_distances = Vector{eltype(distance_matrix)}(undef, n_samples)

    for i = 1:n_samples
        row = copy(view(distance_matrix, i, :))
        partialsort!(row, further_neighbor_idx + 1)
        core_distances[i] = row[further_neighbor_idx+1]
    end

    for i = 1:n_samples

        for j = 1:n_samples
            mutual_reachability_distance =
                max(core_distances[i], core_distances[j], distance_matrix[i, j])

            distance_matrix[i, j] = mutual_reachability_distance
        end

    end

    return nothing

end

"""
    _sparse_mutual_reachability_graph!(
        data, rowval, colptr,
        n_samples, further_neighbor_idx, max_distance)

Compute the mutual reachability graph for a sparse distance matrix stored
in Compressed Sparse Column (CSC) format.

This is the sparse implementation of the mutual reachability graph
construction. The computation is performed in-place by modifying the
nonzero values stored in `data`.

# Arguments
- `data::Vector{<:Real}`: Nonzero values of the sparse distance matrix.
- `rowval::Vector{Int}`: Row indices corresponding to the entries in
  `data`.
- `colptr::Vector{Int}`: Column pointer array defining the CSC structure.
- `n_samples::Int`: Number of samples represented by the distance matrix.
- `further_neighbor_idx::Int`: Index of the furthest nearest neighbor
  used to compute the core distance of each sample.
- `max_distance::Real`: Value used to replace infinite mutual
  reachability distances.

# Returns
- The modified sparse distance matrix represented by `data`, `rowval`,
  and `colptr`.
"""
function _sparse_mutual_reachability_graph!(
    data,
    rowval,
    colptr,
    n_samples::Int,
    further_neighbor_idx::Int,
    max_distance,
)

    core_distances = Vector{eltype(data)}(undef, n_samples)

    for col = 1:n_samples
        start_idx = colptr[col]
        end_idx = colptr[col+1] - 1

        if start_idx <= end_idx

            col_data = data[start_idx:end_idx]

            if further_neighbor_idx < length(col_data)

                tmp = copy(col_data)
                partialsort!(tmp, 1:(further_neighbor_idx+1))
                core_distances[col] = tmp[further_neighbor_idx+1]

            else
                core_distances[col] = Inf
            end
        else
            core_distances[col] = Inf
        end
    end

    for col = 1:n_samples

        for k = colptr[col]:(colptr[col+1]-1)

            row = rowval[k]

            mr = max(core_distances[row], core_distances[col], data[k])

            if isfinite(mr)

                data[k] = mr

            elseif max_distance > 0

                data[k] = max_distance

            end
        end
    end

    return nothing

end

# HDBSCAN



const INFTY = Inf
const NOISE = -1

const KD_TREE_VALID_METRICS = Set([
    "euclidean",
    "manhattan",
    "taxicab",
    "cityblock",
    "chebyshev",
    "chessboard",
    "minkowski",
])

# Missing distances from BALL_TREE_VALID_METRICS: Dice, Russel Rao, SokalMickner
# SokalSneath
const BALL_TREE_VALID_METRICS = Set([
    "braycurtis",
    "canberra",
    "weightedcityblock",
    "chebyshev",
    "chessboard",
    "l2",
    "euclidean",
    "hamming",
    "haversine",
    "jaccard",
    "mahalanobis",
    "l1",
    "manhattan",
    "taxicab",
    "cityblock",
    "minkowski",
    "rogerstanimoto",
    "sqeuclidean",
    "seuclidean",
    "weightedminkowsky",
    "wminkowsky",
])

# Returns the function from the string

function _get_metric_object(metric::String)
    if metric == "euclidean" || metric == "l2"
        return Euclidean()
    elseif metric == "cosine"
        return CosineDist()
    elseif metric == "manhattan" ||
           metric == "taxicab" ||
           metric == "cityblock" ||
           metric == "l1"
        return Cityblock()
    elseif metric == "chebyshev" || metric == "chessboard"
        return Chebyshev()
    elseif metric == "braycurtis"
        return BrayCurtis()
    elseif metric == "hamming"
        return Hamming()
    elseif metric == "haversine"
        return Haversine()
    elseif metric == "jaccard"
        return Jaccard()
    elseif metric == "rogerstanimoto"
        return RogersTanimoto()
    elseif metric == "sqeuclidean" || metric == "seuclidean"
        return SqEuclidean()
    else
        throw(ArgumentError("No valid metric found for the input metric params"))
    end
end

function _get_metric_object(metric::String, param)
    if metric == "minkowski"
        return Minkowski(params)
    elseif metric == "canberra" || metric == "weightedcityblock"
        return WeightedCityblock(params)
    elseif metric == "mahalanobis"
        return Mahalanobis(params)
    else
        throw(ArgumentError("No valid metric found for the input metric params"))
    end
end

function _get_metric_object(metric::String, param1, param2)
    if metric == "weightedminkowsky" || metric == "wminkowsky"
        return WeightedMinkowski(param1, param2)
    else
        throw(ArgumentError("No valid metric found for the input metric params"))
    end
end

const FAST_METRICS = union(KD_TREE_VALID_METRICS, BALL_TREE_VALID_METRICS)

# Encodings are arbitrary but must be strictly negative.
# The current encodings are chosen as extensions to the -1 noise label.
# Avoided enums so that the end user only deals with simple labels.
struct Encoding
    label::Int
    prob::Union{Nothing,Float64}
end

_OUTLIER_ENCODING = Dict("infinite" => Encoding(-2, 0.0), "missing" => Encoding(-3, NaN))

"""
    Hdbscan(min_cluster_size, min_samples=nothing; kwargs...)

Hierarchical Density-Based Spatial Clustering of Applications with Noise
(HDBSCAN).

`Hdbscan` performs hierarchical density-based clustering by constructing a
hierarchy of density-connected components and selecting the most stable
clusters. Unlike DBSCAN, HDBSCAN can identify clusters with varying
densities and is generally less sensitive to parameter selection.

# Arguments
- `min_cluster_size::Int=5`: Minimum number of samples required for a
  group to be considered a cluster.
- `min_samples::Union{Nothing,Int}=nothing`: Number of neighbors used to
  compute the core distance. If `nothing`, it defaults to
  `min_cluster_size`.
- `cluster_selection_epsilon::Real=0.0`: Distance threshold used when
  merging clusters during cluster selection.
- `max_cluster_size::Union{Nothing,Int}=nothing`: Maximum size of clusters
  selected by the `"eom"` cluster selection method.
- `metric::String="euclidean"`: Distance metric used to compute pairwise
  distances. Use `"precomputed"` if the input is a distance matrix.
- `metric_params`: Additional arguments passed to the selected distance
  metric.
- `alpha::Real=1.0`: Distance scaling parameter used in Robust Single
  Linkage.
- `algorithm::String="auto"`: Algorithm used to compute core distances.
  Supported values are `"auto"`, `"brute"`, `"kd_tree"` and
  `"ball_tree"`.
- `leaf_size::Int=40`: Leaf size used by tree-based nearest-neighbor
  algorithms.
- `n_jobs::Union{Nothing,Int}=nothing`: Number of parallel jobs used
  during distance computations, if supported.
- `cluster_selection_method::String="eom"`: Cluster selection method.
  Supported values are `"eom"` and `"leaf"`.
- `allow_single_cluster::Bool=false`: Whether a single cluster may be
  returned.
- `store_centers::Union{Nothing,String}=nothing`: Cluster centers to
  compute and store. Supported values are `"centroid"`, `"medoid"` and
  `"both"`.
- `copy::Bool=false`: Whether to copy the input before performing
  in-place operations.

# Fields
After calling `fit!`, the following fields are populated:

- `labels_::Vector{Int}`: Cluster label assigned to each sample.
- `probabilities_::Vector{Float64}`: Membership strength of each sample.
- `n_features_in_`: Number of input features.
- `feature_names_in_`: Names of the input features, when available.
- `centroids_`: Cluster centroids, if requested.
- `medoids_`: Cluster medoids, if requested.
"""
mutable struct Hdbscan
    # Hyperparameters
    min_cluster_size::Int
    min_samples::Union{Nothing,Int}
    cluster_selection_epsilon::Float64
    max_cluster_size::Union{Nothing,Int}
    metric::String
    metric_params::Array{Number}
    alpha::Float64
    algorithm::String
    leaf_size::Int
    n_jobs::Union{Int,Nothing}
    cluster_selection_method::String
    allow_single_cluster::Bool
    store_centers::Union{Nothing,String}
    copy::Union{Bool,String}

    # Internal state
    _metric_params::Array{Number}
    _raw_data::Any
    _min_samples::Union{Int,Nothing}
    _single_linkage_tree::Union{Vector{HierarchyTree},Nothing}

    # Outputs
    labels_::Union{Nothing,Vector{Int}}
    probabilities_::Union{Nothing,Vector{Float64}}
    n_features_in_::Any
    feature_names_in_::Any
    centroids_::Any
    medoids_::Any
end

function Hdbscan(
    min_cluster_size::Int = 5,
    min_samples::Union{Nothing,Int} = nothing;
    cluster_selection_epsilon::Float64 = 0.0,
    max_cluster_size::Union{Nothing,Int} = nothing,
    metric::String = "euclidean",
    metric_params::Array{Number} = Number[],
    alpha::Float64 = 1.0,
    algorithm::String = "auto",
    leaf_size::Int = 40,
    n_jobs::Union{Int,Nothing} = nothing,
    cluster_selection_method::String = "eom",
    allow_single_cluster::Bool = false,
    store_centers::Union{Nothing,String} = nothing,
    copy::Union{Bool,String} = "warn",
)

    return Hdbscan(
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        max_cluster_size,
        metric,
        metric_params,
        alpha,
        algorithm,
        leaf_size,
        n_jobs,
        cluster_selection_method,
        allow_single_cluster,
        store_centers,
        copy,
        Number[],
        nothing,
        nothing,
        nothing,
        Int[],
        Float64[],
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

"""
    _brute_mst(mutual_reachability, min_samples)

Construct the Minimum Spanning Tree (MST) of a mutual reachability graph.

This function computes the MST from the provided mutual reachability graph,
using an implementation specialized for either dense or sparse input.

# Arguments
- `mutual_reachability`: Dense or sparse weighted adjacency matrix
  representing the mutual reachability graph.
- `min_samples::Union{Nothing, Int}=nothing`: Number of neighbors used to
  define core points. This parameter is only required when processing
  sparse graphs.

# Returns
- `Vector{MST_edge}`: Minimum spanning tree represented as a collection of
  weighted edges.
"""
function _brute_mst(mutual_reachability, min_samples::Int)

    if !issparse(mutual_reachability)
        return mst_from_mutual_reachability(mutual_reachability)
    end

    if !(mutual_reachability isa SparseMatrixCSC)

        throw(
            ArgumentError(
                "Only sparse CSC matrices are supported for mutual_reachability.",
            ),
        )

    end

    n_samples = size(mutual_reachability, 1)
    colptr = mutual_reachability.colptr

    for j = 1:n_samples

        nnz_in_col = colptr[j+1] - colptr[j]

        if nnz_in_col < min_samples

            throw(
                ArgumentError(
                    "There exist points with fewer than $min_samples neighbors. " *
                    "Ensure your sparse distance matrix has non-zero values for at least " *
                    "min_samples=$min_samples neighbors for each point, or specify " *
                    "a max_distance to use when distances are missing.",
                ),
            )

        end

    end

    G = SimpleWeightedGraph(mutual_reachability)

    # Check connected component on mutual reachability.
    # If more than one connected component is present,
    # it means that the graph is disconnected.
    comps = connected_components(G)

    if length(comps) > 1

        throw(
            ArgumentError(
                "Sparse mutual reachability matrix has $(length(comps)) connected " *
                "components. HDBSCAN cannot be performed on a disconnected graph.",
            ),
        )

    end

    # Compute the minimum spanning tree for the sparse graph. The algorithm
    # used is Kruskal
    mst_graph = kruskal_mst(G)

    mst = Vector{MSTEdge}(undef, length(mst_graph))

    for (k, e) in enumerate(mst_graph)
        u = src(e)
        v = dst(e)
        w = weight(G, u, v)

        mst[k] = MSTEdge(u, v, w)
    end

    return mst

end

"""
    _hdbscan_brute(X; min_samples=5, alpha=1.0,
                   metric="euclidean", copy=false, metric_params...)

Construct a single linkage hierarchy from the input data using the brute-force
HDBSCAN algorithm.

If `metric == "precomputed"`, `X` is interpreted as a symmetric distance
matrix. Otherwise, pairwise distances are computed from the input data and
used to construct the mutual reachability graph.

# Arguments
- `X`: Input data matrix or precomputed distance matrix.
- `min_samples::Int=5`: Number of neighbors used to determine the core
  distance of each sample.
- `alpha::Float64=1.0`: Distance scaling parameter used in Robust Single
  Linkage.
- `metric="euclidean"`: Distance metric used to compute pairwise distances.
  If `"precomputed"`, `X` is assumed to be a square distance matrix.
- `copy::Bool=false`: Whether to copy the input before performing any
  in-place modifications.
- `metric_params::Array{Number}`: Additional arguments passed to the distance metric.

# Returns
- `Vector{HierarchyTree}`: Single linkage hierarchy represented as a
  dendrogram.
"""
function _hdbscan_brute(
    X;
    min_samples::Int = 5,
    alpha::Float64 = 1.0,
    metric::String = "euclidean",
    metric_params = Number[],
)

    dist_metric = nothing

    if length(metric_params) == 0
        dist_metric = _get_metric_object(metric)
    elseif length(metric_params) == 1
        dist_metric = _get_metric_object(metric, metric_params[1])
    elseif length(metric_params) == 2
        dist_metric = _get_metric_object(metric, metric_params[1], metric_params[2])
    end

    distance_matrix = metric == "precomputed" ? copy(X) : pairwise(dist_metric, X; dims = 1)

    distance_matrix ./= alpha

    mutual_reachability_ = mutual_reachability_graph(distance_matrix; min_samples)

    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples)

    return _process_mst(min_spanning_tree)
end

"""
    _hdbscan_prims(X, algo; min_samples=5, alpha=1.0,
                   metric="euclidean", leaf_size=40,
                   n_jobs=nothing, metric_params...)

Construct a single linkage hierarchy from the input data using Prim's
algorithm.

Unlike `_hdbscan_brute`, this implementation computes the Minimum
Spanning Tree (MST) directly from the input data without explicitly
constructing the full mutual reachability graph.

# Arguments
- `X::Matrix{<:Real}`: Input data matrix whose rows correspond to samples.
- `algo`::String: Nearest-neighbor search structure used during MST construction.
- `min_samples::Int=5`: Number of neighbors used to compute the core
  distance of each sample.
- `alpha:::Float64=1.0`: Distance scaling parameter used in Robust Single
  Linkage.
- `metric=::String"euclidean"`: Distance metric used to compute pairwise
  distances.
- `leaf_size::Int=40`: Leaf size used by the nearest-neighbor search
  structure, when applicable.
- `n_jobs::Union{Nothing,Int}=nothing`: Number of parallel jobs used for
  distance computations, if supported.
- `metric_params::Array{Number}`: Additional arguments passed to the distance metric.

# Returns
- `Vector{HierarchyTree}`: Single linkage hierarchy represented as a
  dendrogram.
"""
function _hdbscan_prims(
    X;
    algo::String,
    min_samples::Int = 5,
    alpha::Float64 = 1.0,
    metric::String = "euclidean",
    leaf_size::Int = 40,
    n_jobs = nothing,
    metric_params = Number[],
)

    dist_metric = nothing

    if length(metric_params) == 0
        dist_metric = _get_metric_object(metric)
    elseif length(metric_params) == 1
        dist_metric = _get_metric_object(metric, metric_params[1])
    elseif length(metric_params) == 2
        dist_metric = _get_metric_object(metric, metric_params[1], metric_params[2])
    end

    columndata = permutedims(X)

    tree = if algo == "kd_tree"
        KDTree(columndata, dist_metric; leafsize = leaf_size)
    elseif algo == "ball_tree"
        BallTree(columndata, dist_metric; leafsize = leaf_size)
    else
        error("Unsupported algorithm: $algo")
    end

    _, neighbors_distances = knn(tree, columndata, min_samples, true)


    core_distances = [d[end] for d in neighbors_distances]

    # Mutual reachability distance is implicit in mst_from_data_matrix
    min_spanning_tree = mst_from_data_matrix(X, core_distances, dist_metric, alpha)

    return _process_mst(min_spanning_tree)

end


"""
    _process_mst(min_spanning_tree)

Construct a single linkage hierarchy from a Minimum Spanning Tree (MST).

The edges of the MST are sorted by weight before being processed to build
the single linkage hierarchy.

# Arguments
- `min_spanning_tree::Vector{MST_edge}`: Minimum spanning tree represented
  as a collection of weighted edges.

# Returns
- `Vector{HierarchyTree}`: Single linkage hierarchy represented as a
  dendrogram.
"""

function _process_mst(min_spanning_tree::Vector{MSTEdge})

    # Note: scikit-learn uses Numpy's argsort (SIMD implementation)
    # It is not stable.

    if pythonpath != ""
        order =
            pyconvert(Vector{Int}, np.argsort([e.distance for e in min_spanning_tree])) .+= 1
    else
        # Sort edges of the min_spanning_tree by weight
        order = sortperm([e.distance for e in min_spanning_tree], alg = QuickSort)
    end
    min_spanning_tree = min_spanning_tree[order]
    # Convert edge list into standard hierarchical clustering format
    return make_single_linkage(min_spanning_tree)

end

"""
    fit!(model, X; y=nothing)

Fit an HDBSCAN model to the input data.

If `model.metric == "precomputed"`, `X` is interpreted as a square
distance matrix. Otherwise, each row of `X` is treated as a data sample
and pairwise distances are computed according to the selected metric.

The fitted model stores the resulting cluster labels, membership
probabilities, and any requested cluster centers.

# Arguments
- `model::Hdbscan`: HDBSCAN model to fit.
- `X`: Feature matrix or precomputed distance matrix.
- `y=nothing`: Ignored. Present for compatibility with the MLJ and
  scikit-learn APIs.

# Returns
- `Hdbscan`: The fitted model.
"""
function fit!(model::Hdbscan, X; y = nothing)


    if model.copy == "warn"
        @warn "The default value of `copy` will change from false to true in
         a future version. Explicitly set `copy` to silence this warning."
        _copy = false
    else
        _copy = model.copy
    end

    if model.metric == "precomputed" && model.store_centers !== nothing
        throw(
            ArgumentError("Cannot store centers when using a precomputed distance matrix."),
        )
    end

    model._metric_params =
        isnothing(model.metric_params) ? Number[] : copy(model.metric_params)

    all_finite = true
    finite_index = nothing
    infinite_index = Int[]
    missing_index = Int[]
    internal_to_raw = Dict{Int,Int}()

    if model.metric != "precomputed"

        X = Matrix{Float64}(X)   # TODO: replace with validate_data equivalent and add support for precomputed data
        model._raw_data = X

        all_finite = all(isfinite, X)

        if !all_finite
            # Pass only the purely finite indices into hdbscan
            # We will later assign all non-finite points their
            # corresponding labels, as specified in `_OUTLIER_ENCODING`

            # Reduce X to make the checks for missing/outlier samples more
            # convenient.
            reduced_X = vec(any(!isfinite, X; dims = 2))

            # Samples with missing data are denoted by the presence of NaN
            missing_index = findall(i -> any(isnan, @view X[i, :]), axes(reduced_X, 1))

            infinite_index = findall(i -> any(isinf, @view X[i, :]), axes(reduced_X, 1))

            # Continue with only finite samples
            finite_index = _get_finite_row_indices(X)

            internal_to_raw = Dict(i => finite_index[i] for i in eachindex(finite_index))

            X = X[finite_index, :]

        end

    elseif issparse(X)

        throw(ArgumentError("Sparse precomputed matrices not yet supported in
                  this Julia port"))

    else

        X = Matrix{Float64}(X)

        if any(isnan, X)

            throw(ArgumentError("NaN values found in precomputed dense
                  distance matrix"))

        end
    end

    if size(X, 1) == 1

        throw(ArgumentError("n_samples = 1 while HDBSCAN requires more than 
            one sample"))

    end

    model.n_features_in_ = ndims(X) == 2 ? size(X, 2) : size(X, 1)

    model._min_samples =
        isnothing(model.min_samples) ? model.min_cluster_size : model.min_samples

    if model._min_samples > size(X, 1)

        throw(
            ArgumentError("min_samples ($(model._min_samples)) must be at most the number
                              of samples in X ($(size(X,1)))"),
        )

    end

    mst_func = nothing
    algo = nothing

    if model.algorithm == "kd_tree" && !(model.metric in KD_TREE_VALID_METRICS)

        throw(ArgumentError("$(model.metric) is not a valid metric for a KDTree-based 
                            algorithm."))

    elseif model.algorithm == "ball_tree" && !(model.metric in BALL_TREE_VALID_METRICS)

        throw(ArgumentError("$(model.metric) is not a valid metric for a BallTree-based
                            algorithm."))

    end

    if model.algorithm != "auto"

        if model.metric != "precomputed" && issparse(X) && model.algorithm != "brute"

            throw(ArgumentError("Sparse data matrices only support algorithm = \"brute\""))
        end

        if model.algorithm == "brute"
            mst_func = _hdbscan_brute

        elseif model.algorithm == "kd_tree"

            mst_func = _hdbscan_prims
            algo = "kd_tree"

        else

            mst_func = _hdbscan_prims
            algo = "ball_tree"

        end

    else

        if issparse(X) || !(model.metric in FAST_METRICS)
            mst_func = _hdbscan_brute

        elseif model.metric in KD_TREE_VALID_METRICS

            mst_func = _hdbscan_prims
            algo = "kd_tree"

        else

            mst_func = _hdbscan_prims
            algo = "ball_tree"

        end

    end

    if mst_func === _hdbscan_brute

        model._single_linkage_tree = _hdbscan_brute(
            X;
            min_samples = model._min_samples,
            alpha = model.alpha,
            metric = model.metric,
            metric_params = model.metric_params,
        )

    elseif mst_func === _hdbscan_prims

        model._single_linkage_tree = _hdbscan_prims(
            X;
            min_samples = model._min_samples,
            alpha = model.alpha,
            metric = model.metric,
            leaf_size = model.leaf_size,
            algo = algo,
            metric_params = model.metric_params,
        )

    else

        error("No MST backend selected")

    end

    model.labels_, model.probabilities_ = tree_to_labels(
        model._single_linkage_tree,
        model.min_cluster_size,
        model.cluster_selection_method,
        model.allow_single_cluster,
        model.cluster_selection_epsilon,
        model.max_cluster_size,
    )
    if model.metric != "precomputed" && !all_finite
        # Remap indices to align with original data in the case of
        # non-finite entries. Samples with inf are mapped to -1 and
        # those with NaN are mapped to -2.
        non_finite = unique(vcat(infinite_index, missing_index))

        model._single_linkage_tree = remap_single_linkage_tree(
            model._single_linkage_tree,
            internal_to_raw,
            non_finite,
        )

        new_labels = Vector{Int}(undef, size(model._raw_data, 1))
        new_labels[finite_index] = model.labels_
        new_labels[infinite_index] .= _OUTLIER_ENCODING["infinite"].label
        new_labels[missing_index] .= _OUTLIER_ENCODING["missing"].label
        model.labels_ = new_labels

        new_probabilities = zeros(Float64, size(model._raw_data, 1))
        new_probabilities[finite_index] = model.probabilities_
        # Infinite outliers have probability 0 by convention, though this
        # is arbitrary.
        new_probabilities[infinite_index] .= _OUTLIER_ENCODING["infinite"].prob
        new_probabilities[missing_index] .= _OUTLIER_ENCODING["missing"].prob
        model.probabilities_ = new_probabilities
    end

    if model.store_centers !== nothing
        _weighted_cluster_center!(model, X)
    end

    return model

end

"""
    fit_predict(model, X; y=nothing)

Fit an HDBSCAN model to the input data and return the resulting cluster
labels.

If `model.metric == "precomputed"`, `X` is interpreted as a square
distance matrix. Otherwise, each row of `X` is treated as a data sample
and pairwise distances are computed according to the selected metric.

# Arguments
- `model::Hdbscan`: HDBSCAN model to fit.
- `X`: Feature matrix or precomputed distance matrix.
- `y=nothing`: Ignored. Present for compatibility with the MLJ and
  scikit-learn APIs.

# Returns
- `Vector{Int}`: Cluster label assigned to each sample. A label of `-1`
  denotes noise, while `-2` and `-3` denote samples containing infinite
  and missing values, respectively.
"""
function fit_predict(hdb::Hdbscan, X; y = nothing)

    fit!(hdb, X)

    return hdb.labels_

end

"""
    _get_finite_row_indices(X::SparseMatrixCSC)

Return the indices of the rows in `X` whose nonzero entries are all finite.

# Arguments
- `X::SparseMatrixCSC`: Sparse matrix to inspect.

# Returns
- `Vector{Int}`: Indices of the rows that do not contain `NaN` or `Inf`
  values.
"""
function _get_finite_row_indices(X::SparseMatrixCSC)

    row_mask = trues(size(X, 1))

    for col = 1:size(X, 2)

        for ptr = X.colptr[col]:(X.colptr[col+1]-1)

            row = X.rowval[ptr]

            if !isfinite(X.nzval[ptr])
                row_mask[row] = false
            end

        end

    end

    return findall(row_mask)

end

"""
    _get_finite_row_indices(X::AbstractMatrix)

Return the indices of the rows in `X` whose entries are all finite.

# Arguments
- `X::AbstractMatrix`: Dense matrix to inspect.

# Returns
- `Vector{Int}`: Indices of the rows that do not contain `NaN` or `Inf`
  values.
"""
function _get_finite_row_indices(X::AbstractMatrix)

    row_mask = vec(all(isfinite.(X), dims = 2))

    return findall(row_mask)

end

"""
    remap_single_linkage_tree(tree, internal_to_raw, non_finite)

Reconstruct a single linkage hierarchy by reintroducing samples that were
removed because they contained non-finite values.

The reintroduced samples are merged into the root of the hierarchy at an
infinite distance and are therefore treated as noise during cluster
extraction.

# Arguments
- `tree::Vector{HierarchyTree}`: Single linkage hierarchy built from the
  finite samples.
- `internal_to_raw::Dict{Int, Int}`: Mapping from the indices used in the
  filtered dataset to the corresponding indices in the original dataset.
- `non_finite::Vector{Int}`: Boolean vector indicating which
  samples in the original dataset contain non-finite values.

# Returns
- `Vector{HierarchyTree}`: Single linkage hierarchy with the non-finite
  samples reinserted.
"""
function remap_single_linkage_tree(
    tree::Vector{HierarchyTree},
    internal_to_raw::Dict{Int,Int},
    non_finite::Vector{Int},
)

    finite_count = length(internal_to_raw)

    outlier_count = length(non_finite)

    for i in eachindex(tree)

        left = tree[i].left_node
        right = tree[i].right_node

        left_remapped = left <= finite_count ? internal_to_raw[left] : left + outlier_count

        right_remapped =
            right <= finite_count ? internal_to_raw[right] : right + outlier_count

        tree[i] = HierarchyTree(
            left_remapped,
            right_remapped,
            tree[i].value,
            tree[i].cluster_size,
        )
    end

    outlier_tree = Vector{HierarchyTree}(undef, length(non_finite))

    last_cluster_id = max(tree[end].left_node, tree[end].right_node)

    last_cluster_size = tree[end].cluster_size

    for i in eachindex(non_finite)

        outlier = non_finite[i]

        outlier_node = outlier + 1

        outlier_tree[i] =
            HierarchyTree(outlier_node, last_cluster_id + 1, Inf, last_cluster_size + 1)

        last_cluster_id += 1
        last_cluster_size += 1

    end

    return vcat(tree, outlier_tree)

end

"""
    _weighted_cluster_center!(model, X)

Compute and store the centroids and/or medoids of the clusters identified
by the fitted HDBSCAN model.

This function requires `X` to contain the original feature vectors rather
than a precomputed distance matrix. The computed centers are stored in the
`centroids_` and/or `medoids_` fields of `model`, depending on the value
of `model.store_centers`.

# Arguments
- `model::Hdbscan`: Fitted HDBSCAN model.
- `X::Matrix{Float64}`: Feature matrix used to fit the model.

# Returns
- The updated `model`, with the requested cluster centers stored in
  `centroids_` and/or `medoids_`.
"""
function _weighted_cluster_center!(model::Hdbscan, X::Matrix{Float64})

    cluster_ids = sort(collect(setdiff(Set(model.labels_), Set([-1, -2]))))
    # Number of non-noise clusters
    n_clusters = length(cluster_ids)

    make_centroids = model.store_centers in ("centroid", "both")
    make_medoids = model.store_centers in ("medoid", "both")

    n_features = size(X, 2)

    if make_centroids
        model.centroids_ = Matrix{Float64}(undef, n_clusters, n_features)
    end

    if make_medoids
        model.medoids_ = Matrix{Float64}(undef, n_clusters, n_features)
    end

    # Need to handle iteratively seen each cluster may have a different
    # number of samples, hence we can't create a homogeneous 3D array.
    for (idx, cluster_label) in enumerate(cluster_ids)

        mask = model.labels_ .== cluster_label
        data = X[mask, :]
        strength = Float64.(model.probabilities_[mask])

        if make_centroids

            total_weight = sum(strength)

            if total_weight == 0.0

                model.centroids_[idx, :] = vec(mean(data, dims = 1))
            else
                centroid = vec(sum(data .* strength, dims = 1) ./ total_weight)
                model.centroids_[idx, :] = centroid
            end

        end

        if make_medoids

            n_points = size(data, 1)

            dist_mat = zeros(Float64, n_points, n_points)

            dist_metric = _get_metric_object(model.metric)

            for i = 1:n_points

                for j = 1:n_points

                    dist_mat[i, j] =
                        evaluate(dist_metric, view(data, i, :), view(data, j, :))

                end
            end

            weighted_dist = dist_mat .* reshape(strength, 1, :)

            medoid_index = argmin(vec(sum(weighted_dist, dims = 2)))
            model.medoids_[idx, :] = data[medoid_index, :]

        end

    end

    return nothing

end

"""
    labels(model)

Return the cluster labels assigned to each sample after fitting the model.

Samples labelled `-1` are considered noise. Labels `-2` and `-3`
correspond to samples containing infinite and missing values,
respectively.
"""
labels(model::Hdbscan) = model.labels_

"""
    probabilities(model)

Return the membership probability of each sample in its assigned cluster.

Values range from 0 to 1, where larger values indicate stronger cluster
membership.
"""
probabilities(model::Hdbscan) = model.probabilities_

"""
    centroids(model)

Return the cluster centroids computed during fitting, if available.

Centroids are only available when `store_centers` is set to `"centroid"`
or `"both"`.
"""
centroids(model::Hdbscan) = model.centroids_

"""
    medoids(model)

Return the cluster medoids computed during fitting, if available.

Medoids are only available when `store_centers` is set to `"medoid"` or
`"both"`.
"""
medoids(model::Hdbscan) = model.medoids_

"""
    single_linkage_tree(model)

Return the single linkage hierarchy constructed during model fitting.
"""
single_linkage_tree(model::Hdbscan) = model._single_linkage_tree

const OUTLIER_SET = (-1, -2, -3)

"""
    nclusters(model)

Return the number of clusters identified by the fitted model.

Noise points are not included in the count.
"""
function nclusters(model::Hdbscan)
    length(setdiff(unique(model.labels_), OUTLIER_SET))
end

end  # module
