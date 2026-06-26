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



#=
    Required libraries
    Quick mass installation -> uncomment

    import Pkg;
    Pkg.add("Statistics")
    Pkg.add("SparseArrays")
    Pkg.add("Missings")
    Pkg.add("Distances")
    Pkg.add("NearestNeighbors")
    Pkg.add("Graphs")
    Pkg.add("SimpleWeightedGraphs")
=#

using Statistics
using SparseArrays
using Missings
using Distances
using NearestNeighbors
using Graphs
using SimpleWeightedGraphs

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

################################################################################
#                                _tree.pyx                                     #
################################################################################

#= 
    HIERARCHY_t

    The HDBSCAN single-linkage hierarchy tree is represented by an 
    Array{HIERARCHY_t}, where each entry is a merge event defined by:

    left_node::Int
    right_node::Int
        Identifiers of child in the merge.

    value::Float64
        Distance in the mutual reachability tree at which the two components
        are joined.

    cluster_size::Int
        Number of points contained in the newly formed cluster.
=#

mutable struct HIERARCHY_t
    left_node::Int
    right_node::Int
    value::Float64
    cluster_size::Int
end

#=
    CONDENSED_t

    The condensed tree is the result of the "pruning" clusters that falls below
    minimum cluster size.

    It's represented by an Array{CONDENSED_t} where each entry is defined by:
    
    parent::Int
    Identifier of the parent cluster in the condensed tree.

    child::Int
    Identifier of the child cluster in the condensed tree.

    value::Float64
    The persistence/split level where the child cluster survives. It's the alpha
    value where alpha = 1 / distance

    cluster_size::Int
    Number of points contained in child at this level of the condensed tree
=#

mutable struct CONDENSED_t
    parent::Int
    child::Int
    value::Float64
    cluster_size::Int
end

#=
    function HIERARCHY_t_shift_index_python_to_julia
    function HIERARCHY_t_shift_index_julia_to_python

    Not included in the scikit-learn code. It defines the relation between
    Python indexing (0-based) and Julia indexing (1-based) of hierarchy trees
    for testing purposes.
=#

function HIERARCHY_t_shift_index_python_to_julia(
    tree::Array{HIERARCHY_t})

    for n in tree
        n.left_node += 1
        n.right_node += 1
    end
    return tree
end

function HIERARCHY_t_shift_index_julia_to_python(
    tree::Array{HIERARCHY_t})

    for n in tree
        n.left_node -= 1
        n.right_node -= 1
    end
    return tree
end

#=
    
=#

function tree_to_labels(
    single_linkage_tree::Array{HIERARCHY_t},
    min_cluster_size::Int=10,
    cluster_selection_method="eom",
    allow_single_cluster::Bool=false,
    cluster_selection_epsilon::Float64=0.0,
    max_cluster_size=nothing)

    condensed_tree = _condense_tree(single_linkage_tree, min_cluster_size)
    labels, probabilities = _get_clusters(condensed_tree,
        _compute_stability(condensed_tree),
        cluster_selection_method,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,)

    return (labels, probabilities)
end


    function bfs_from_hierarchy(
        hierarchy::Array{HIERARCHY_t}, bfs_root::Int)

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

    #=
    bfs_from_hierarchy(hierarchy_test_tree_julia, 19)
    19-element Vector{Int64}:
    19
    18
    17
    16
    15
    13
    14
    11
    12
    9
    10
    5
    6
    7
    8
    1
    2
    3
    4

    -> Correct
    =#


    function _condense_tree(
        hierarchy::Array{HIERARCHY_t}, min_cluster_size::Int=10)

        root = 2 * length(hierarchy) + 1
        n_samples = length(hierarchy) + 1
        next_label = n_samples + 2

        node_list = bfs_from_hierarchy(hierarchy, root)

        relabel = zeros(Int, root + 1)
        relabel[root] = n_samples + 1

        result_list = CONDENSED_t[]
        ignore = falses(root)

        for node in node_list
            if ignore[node] || node <= n_samples
                continue
            end

            children = hierarchy[node - n_samples]
            left = children.left_node
            right = children.right_node
            distance = children.value

            if distance > 0.0
                lambda_value = 1.0 / distance
            else
                lambda_value = INFTY
            end

            if left > n_samples
                left_count = hierarchy[left - n_samples].cluster_size
            else
                left_count = 1
            end
            
            if right > n_samples
                right_count = hierarchy[right - n_samples].cluster_size
            else
                right_count = 1
            end

            if left_count >= min_cluster_size && right_count >= min_cluster_size
                relabel[left] = next_label
                next_label += 1
                push!(result_list, CONDENSED_t(relabel[node], relabel[left],
                    lambda_value, left_count))

                relabel[right] = next_label
                next_label += 1
                push!(result_list, CONDENSED_t(relabel[node],
                    relabel[right], lambda_value, right_count))

            elseif left_count < min_cluster_size && right_count < min_cluster_size
                for sub_node in bfs_from_hierarchy(hierarchy, left)
                    if sub_node <= n_samples
                        push!(result_list, CONDENSED_t(relabel[node],
                            sub_node, lambda_value, 1))
                    end
                    ignore[sub_node] = true
                end

                for sub_node in bfs_from_hierarchy(hierarchy, right)
                    if sub_node <= n_samples
                        push!(result_list, CONDENSED_t(relabel[node],
                            sub_node, lambda_value, 1))
                    end
                    ignore[sub_node] = true
                end

            elseif left_count < min_cluster_size
                relabel[right] = relabel[node]
                for sub_node in bfs_from_hierarchy(hierarchy, left)
                    if sub_node < n_samples
                        push!(result_list, CONDENSED_t(relabel[node],
                        sub_node, lambda_value, 1))
                    end
                    ignore[sub_node] = true
                end

            else
                relabel[left] = relabel[node]
                for sub_node in bfs_from_hierarchy(hierarchy, right)
                    if sub_node <= n_samples
                        push!(result_list, CONDENSED_t(relabel[node],
                        sub_node,lambda_value, 1))
                    end
                    ignore[sub_node] = true
                end
            end
        end

        return result_list
    end

    #=
    _condense_tree(hierarchy_test_tree_julia, 10)
    10-element Vector{CONDENSED_t}:
    CONDENSED_t(11, 9, 1.0, 1)
    CONDENSED_t(11, 10, 1.0, 1)
    CONDENSED_t(11, 1, 1.0, 1)
    CONDENSED_t(11, 2, 1.0, 1)
    CONDENSED_t(11, 3, 1.0, 1)
    CONDENSED_t(11, 4, 1.0, 1)
    CONDENSED_t(11, 5, 1.0, 1)
    CONDENSED_t(11, 6, 1.0, 1)
    CONDENSED_t(11, 7, 1.0, 1)
    CONDENSED_t(11, 8, 1.0, 1)
    -> Correct

    =#

    function _compute_stability(condensed_tree::Array{CONDENSED_t})
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

        result = zeros(Float64, maximum(parents))

        for idx in eachindex(condensed_tree)
            condensed_node = condensed_tree[idx]
            parent = condensed_node.parent
            lambda_val = condensed_node.value
            cluster_size = condensed_node.cluster_size
            
            result_index = parent - smallest_cluster + 1
            result[result_index] = result[result_index] +
                ((lambda_val - births[parent]) * cluster_size)
        end

        stability_dict = Dict{Int, Float64}()

        for idx in 1:num_clusters
            stability_dict[idx + smallest_cluster - 1] = result[idx]
        end

        return stability_dict
    end

    #= _compute_stability(_condense_tree(hierarchy_test_tree_julia, 10))

    Dict{Int64, Float64} with 1 entry:
    11 => 10.0 

    -> Correct
    =#

    function bfs_from_cluster_tree(
        condensed_tree::Array{CONDENSED_t}, bfs_root::Int)

        result = Int[]
        process_queue = [bfs_root]

        children = [c.child for c in condensed_tree]
        parents = [c.parent for c in condensed_tree]

        while !isempty(process_queue)
            append!(result, process_queue)
            process_queue = [children[i]
                for i in eachindex(children)
                    if parents[i] in process_queue]
        end

        return result
    end

    #= bfs_from_cluster_tree(_condense_tree(hierarchy_test_tree_julia, 10), 11)
    11-element Vector{Int64}:
    11
    9
    10
    1
    2
    3
    4
    5
    6
    7
    8

    -> Correct
    =#

    function max_lambdas(
        condensed_tree::Array{CONDENSED_t})

        largest_parent = maximum([c.parent for c in condensed_tree])
        deaths = zeros(Float64, largest_parent)

        current_parent = condensed_tree[1].parent
        max_lambda = condensed_tree[1].value

        for i in 2:length(condensed_tree)
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
        deaths[current_parent] = max_lambda
        return deaths
    end

    #= max_lambdas(_condense_tree(hierarchy_test_tree_julia, 10))

    11-element Vector{Float64}:
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    1.0

    -> Correct
    =#

    mutable struct TreeUnionFind
        data::Array{Int,2}
        is_component::Vector{Bool}
    end

    function init_TreeUnionFind(size::Int)
        data = zeros(Int, size, 2)
        for i in 1:size
            data[i , 1] = i
        end
        is_component = trues(size)
        tuf = TreeUnionFind(data, is_component)
        return tuf
    end
    
    #= index shift

    init_TreeUnionFind(5)
    TreeUnionFind([1 0; 2 0;...; 4 0; 5 0], [1, 1, 1, 1, 1])

    -> Correct
    =#


    function union(tuf::TreeUnionFind, x::Int, y::Int)

        x_root = find(tuf, x)
        y_root = find(tuf, y)

        if tuf.data[x_root, 2] < tuf.data[y_root, 2]
            tuf.data[x_root, 1] = y_root
        elseif tuf.data[x_root, 2] > tuf.data[y_root, 2]
            tuf.data[y_root, 1] = x_root
        else
            tuf.data[y_root, 1] = x_root
            tuf.data[x_root, 2] += 1
        end
        return tuf
    end

    #= union(init_TreeUnionFind(5), 3, 5)
    TreeUnionFind([1 0; 2 0; ... ; 4 0; 3 0], Bool[1, 1, 1, 1, 1])

    -> Correct
    =#

    function find(tuf::TreeUnionFind, x::Int)
        if tuf.data[x, 1] != x
            tuf.data[x, 1] = find(tuf, tuf.data[x, 1])
            tuf.is_component[x] = false
        end
        return tuf.data[x, 1]
    end

    #= find(init_TreeUnionFind(5), 3)
    3

    -> Correct
    =#


    function labelling_at_cut(linkage::Array{HIERARCHY_t},
        cut::Float64,
        min_cluster_size::Int)

        root = 2 * length(linkage) + 1 #1 index
        n_samples = length(linkage) + 1

        result = zeros(Int, n_samples)
        union_find = init_TreeUnionFind(root + 1)

        cluster = n_samples + 1
        for node in linkage
            if node.value < cut
                union_find = union(union_find, node.left_node, cluster)
                union_find = union(union_find, node.right_node, cluster)
            end
            cluster += 1
        end

        cluster_size = zeros(Int, cluster)

        for n in 1:n_samples
            cluster = find(union_find, n)
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

        for n in 1:n_samples
            result[n] = cluster_label_map[result[n]]
        end

        return result
    end

    #= labelling_at_cut(hierarchy_test_tree_julia, 0.5, 1)
    10-element Vector{Int64}:
    0
    0
    1
    1
    2
    2
    3
    3
    4
    5

    -> Correct
    =#


    function _do_labelling(
    condensed_tree::Array{CONDENSED_t},
    clusters,
    cluster_label_map::Dict,
    allow_single_cluster::Bool,
    cluster_selection_epsilon::Float64
)
    child_array = [c.child for c in condensed_tree]
    parent_array = [c.parent for c in condensed_tree]
    lambda_array = [c.value for c in condensed_tree]

    root_cluster = minimum(parent_array)
    n_samples = root_cluster - 1
    result = fill(NOISE, n_samples)

    max_label = max(maximum(parent_array), maximum(child_array))
    union_find = init_TreeUnionFind(max_label)

    for i in eachindex(condensed_tree)
        child = child_array[i]
        parent = parent_array[i]

        if !(child in clusters)
            child_root = find(union_find, child)
            parent_root = find(union_find, parent)
            union_find.data[child_root, 1] = parent_root
        end
    end

    parent_map = Dict{Int,Int}()
    for i in eachindex(condensed_tree)
        parent_map[child_array[i]] = parent_array[i]
    end

    for n in 1:n_samples
        cluster = find(union_find, n)
        label = NOISE

        while cluster != root_cluster &&
              !haskey(cluster_label_map, cluster) &&
              haskey(parent_map, cluster)
            cluster = parent_map[cluster]
        end

        if cluster != root_cluster && haskey(cluster_label_map, cluster)
            label = cluster_label_map[cluster]

        elseif length(clusters) == 1 && allow_single_cluster
            parent_lambda = [
                lambda_array[i]
                for i in eachindex(child_array)
                if child_array[i] == n
            ]

            threshold =
                if cluster_selection_epsilon != 0.0
                    1 / cluster_selection_epsilon
                else
                    maximum([
                        lambda_array[i]
                        for i in eachindex(parent_array)
                        if parent_array[i] == root_cluster
                    ])
                end

            if !isempty(parent_lambda) && maximum(parent_lambda) >= threshold
                # only assign if the root cluster is actually selected
                if haskey(cluster_label_map, root_cluster)
                    label = cluster_label_map[root_cluster]
                end
            end
        end

        result[n] = label
    end

    return result
end
    
#=
    _do_labelling(CONDENSED_t[
        CONDENSED_t(11, 12, 0.5, 2),
        CONDENSED_t(11, 13, 0.5, 2),
        CONDENSED_t(12, 1, 1.0, 1),
        CONDENSED_t(12, 2, 1.0, 1),
        CONDENSED_t(13, 3, 1.0, 1),
        CONDENSED_t(13, 4, 1.0, 1),],
        
        Set([12, 13]),

        Dict(12 => 0, 13 => 1),

        false,

        0.0)
=#

    function get_probabilities(condensed_tree::Array{CONDENSED_t},
        cluster_map::Dict,
        labels::Array{Int})

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

    #= get_probabilities([CONDENSED_t(11, 1, 0.4, 1),
        CONDENSED_t(11, 2, 0.7, 1),
        CONDENSED_t(12, 3, 1.2, 1),
        CONDENSED_t(12, 4, 0.9, 1),
        CONDENSED_t(13, 5, 2.0, 1),],
        Dict(0 => 11,
            1 => 12,
            2 => 13),
        [0, 0, 1, 1, 2])

        5-element Vector{Float64}:
        0.5714285714285715
        1.0
        1.0
        0.75
        1.0

    -> Correct
    =#


    function recurse_leaf_dfs(cluster_tree::Array{CONDENSED_t},
        current_node::Int)

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

    #=  recurse_leaf_dfs(_condense_tree(hierarchy_test_tree_julia), 11)
    10-element Vector{Any}:
    9
    10
    1
    2
    3
    4
    5
    6
    7
    8

    -> Correct
    =#

    function get_cluster_tree_leaves(cluster_tree::Array{CONDENSED_t}
        )

        if isempty(cluster_tree)
            return []
        end
        root = minimum([c.parent for c in cluster_tree])
        return recurse_leaf_dfs(cluster_tree, root)
    end

    #= get_cluster_tree_leaves(_condense_tree(hierarchy_test_tree_julia))
    10-element Vector{Any}:
    9
    10
    1
    2
    3
    4
    5
    6
    7
    8

    -> Correct
    =#

    function traverse_upwards(cluster_tree::Array{CONDENSED_t},
        cluster_selection_epsilon::Float64,
        leaf::Int,
        allow_single_cluster::Bool)

        root = minimum([c.parent for c in cluster_tree])
        parent = only([c.parent for c in cluster_tree if c.child == leaf])

        if parent == root
            if allow_single_cluster
                return parent
            else
                return leaf
            end
        end

        parent_eps = 1 / only([c.value for c in cluster_tree if c.child == parent])

        if parent_eps > cluster_selection_epsilon
            return parent
        else
            return traverse_upwards(cluster_tree, cluster_selection_epsilon,
            parent, allow_single_cluster)
        end
    end

    #=
    traverse_upwards(_condense_tree(hierarchy_test_tree_julia), 0.5, 5, true)
    11

    -> Correct
    =#

    function epsilon_search(leaves::Set,
        cluster_tree::Array{CONDENSED_t},
        cluster_selection_epsilon::Float64,
        allow_single_cluster::Bool)

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
                        allow_single_cluster
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
#=
    epsilon_search(Set([9, 7, 8, 5]), _condense_tree(hierarchy_test_tree_julia), 0.5, true)

    -> Correct
=#
    

    function _get_clusters(condensed_tree::Array{CONDENSED_t},
        stability::Dict,
        cluster_selection_method="eom",
        allow_single_cluster=false,
        cluster_selection_epsilon=0.0,
        max_cluster_size=nothing)

        if allow_single_cluster
            node_list = sort(collect(keys(stability)), rev=true)
        else
            node_list = sort(collect(keys(stability)), rev=true)[1:end-1]
        end

        cluster_tree = [c for c in condensed_tree if c.cluster_size > 1]
        is_cluster = Dict(c => true for c in node_list)

        n_samples = maximum([c.child
            for c in condensed_tree if c.cluster_size == 1])

        if max_cluster_size === nothing
            max_cluster_size = n_samples + 1
        end

        cluster_sizes = Dict(c.child => c.cluster_size for c in cluster_tree)

        if allow_single_cluster
            root = node_list[end]
            cluster_sizes[root] = sum([c.cluster_size
            for c in cluster_tree if c.parent == root])
        end

        if cluster_selection_method == "eom"
            for node in node_list
                
                children = [c.child for c in cluster_tree if c.parent == node]

                if isempty(children)
                    continue
                end

                subtree_stability = sum(stability[c] for c in children)

                if subtree_stability > stability[node] ||
                    get(cluster_sizes, node, 0) > max_cluster_size

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

                if length(eom_clusters) == 1 &&
                    eom_clusters[1] == minimum([c.parent for c in cluster_tree])

                    selected_clusters = allow_single_cluster ? eom_clusters : Int[]
                else
                    selected_clusters = epsilon_search(Set(eom_clusters),
                        cluster_tree,
                        cluster_selection_epsilon,
                        allow_single_cluster)
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
                selected_clusters = epsilon_search(leaves,
                    cluster_tree, cluster_selection_epsilon, allow_single_cluster)
            else
                selected_clusters = leaves
            end

            for c in keys(is_cluster)
                is_cluster[c] = c in selected_clusters
            end
        end

        clusters = Set([c for c in keys(is_cluster) if is_cluster[c]])
        cluster_map = Dict(c => i-1
            for (i,c) in enumerate(sort(collect(clusters))))
        reverse_cluster_map = Dict(v => k for (k,v) in cluster_map)

        labels = _do_labelling(condensed_tree,
            clusters,
            cluster_map,
            allow_single_cluster,
            cluster_selection_epsilon)

        probs = get_probabilities(condensed_tree, reverse_cluster_map, labels)

        return (labels, probs)
    end

#=
    _get_clusters(_condense_tree(hierarchy_test_tree_julia),
        _compute_stability(_condense_tree(hierarchy_test_tree_julia)),
        "eom", false, 0.0, nothing)

    ([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
=#

################################# _linkage ###################################

using Distances

struct MST_edge_t
    current_node::Int64
    next_node::Int64
    distance::Float64
end

#Union find optimized for MST

mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
    next_label::Int
end

function init_UnionFind(n::Int)
    parent = zeros(Int, 2n - 1)
    size = vcat(ones(Int, n), zeros(Int, n - 1))
    next_label = n + 1
    return UnionFind(parent, size, next_label)
end

function link_find(uf::UnionFind, x::Int)
    p = x
    while uf.parent[p] != 0
        p = uf.parent[p]
    end

    # path compression
    y = x
    while uf.parent[y] != 0 && uf.parent[y] != p
        old = y
        y = uf.parent[y]
        uf.parent[old] = p
    end

    return p
end

function link_union(uf::UnionFind, m::Int, n::Int)
    new_label = uf.next_label

    uf.parent[m] = new_label
    uf.parent[n] = new_label
    uf.size[new_label] = uf.size[m] + uf.size[n]

    uf.next_label += 1
    return new_label
end

#=
    fast_union(init_UnionFind(5), 3, 5)
    UnionFind(7, [-1, -1, 6, -1, 6, -1, -1, -1, -1], [1, 1, 1, 1, 1, 2, 0, 0, 0])

    -> Correct

=#

function mst_from_mutual_reachability(mutual_reachability::Matrix{Float64})
    
    n_samples = size(mutual_reachability, 1)
    mst = Vector{MST_edge_t}(undef, n_samples - 1)
    
    current_labels = collect(1:n_samples)
    current_node = 1

    min_reachability = fill(Inf, n_samples)

    for i in 1:n_samples -1
        label_filter = current_labels .!= current_node
        current_labels = current_labels[label_filter]

        left = min_reachability[label_filter]
        right = mutual_reachability[current_node, current_labels]

        min_reachability = min.(left, right)

        new_node_index = argmin(min_reachability)
        new_node = current_labels[new_node_index]

        mst[i] = MST_edge_t(
            current_node,
            new_node,
            min_reachability[new_node_index])

        current_node = new_node
    end

    return mst
end



#=
mst_from_mutual_reachability([
    0.0  1.0  4.0  3.0;
    1.0  0.0  2.0  5.0;
    4.0  2.0  0.0  1.5;
    3.0  5.0  1.5  0.0])

3-element Vector{MST_edge_t}:
 MST_edge_t(1, 2, 1.0)
 MST_edge_t(2, 3, 2.0)
 MST_edge_t(3, 4, 1.5)

-> Correct
=#


function mst_from_data_matrix(
    raw_data::Matrix{Float64},
    core_distances::Vector{Float64},
    dist_metric,
    alpha::Float64=1.0,
)

    n_samples = size(raw_data, 1)
    mst = Vector{MST_edge_t}(undef, n_samples - 1)

    in_tree = falses(n_samples)
    min_reachability = fill(Inf, n_samples)
    current_sources = ones(Int, n_samples)

    current_node = 1

    for i in 1:(n_samples - 1)

        in_tree[current_node] = true
        current_node_core_dist = core_distances[current_node]
        new_reachability = Inf
        source_node = 1
        new_node = 1

        # Update frontier
        for j in 1:n_samples
            if in_tree[j]
                continue
            end

            next_node_min_reach = min_reachability[j]
            next_node_source = current_sources[j]

            pair_distance = Distances.evaluate(
                dist_metric,
                view(raw_data, current_node, :),
                view(raw_data, j, :)

            ) / alpha

            next_node_core_dist = core_distances[j]

            mutual_reachability_distance = max(
                current_node_core_dist,
                next_node_core_dist,
                pair_distance
            )

            if mutual_reachability_distance < next_node_min_reach
                min_reachability[j] = mutual_reachability_distance
                current_sources[j] = current_node

                if mutual_reachability_distance < new_reachability
                    new_reachability = mutual_reachability_distance
                    source_node = current_node
                    new_node = j
                end

            elseif next_node_min_reach < new_reachability
                new_reachability = next_node_min_reach
                source_node = next_node_source
                new_node = j
            end
        
        end

        mst[i] = MST_edge_t(source_node, new_node, new_reachability)
        current_node = new_node
    end
    return mst
end

mst_test = mst_from_data_matrix(
    [0.0 0.0;
    1.0 0.0;
    0.0 1.0;
    1.0 1.0;
    3.0 3.0],

    [1.0,
    1.0,
    1.0,
    1.0,
    2.5],

    Euclidean(),

    1.0)
#=
    4-element Vector{MST_edge_t}:
    MST_edge_t(1, 2, 1.0)
    MST_edge_t(1, 3, 1.0)
    MST_edge_t(2, 4, 1.0)
    MST_edge_t(4, 5, 2.8284271247461903)

-> Correct
=#

function make_single_linkage(mst::Vector{MST_edge_t})
    n_samples = length(mst) + 1
    single_linkage = Vector{HIERARCHY_t}(undef, n_samples - 1)

    parent = collect(1:n_samples)
    comp_size = ones(Int, n_samples)
    cluster_label = collect(1:n_samples)

    function find_root(x::Int)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    next_cluster = n_samples + 1

    for i in eachindex(mst)
        u = mst[i].current_node
        v = mst[i].next_node
        d = mst[i].distance

        ru = find_root(u)
        rv = find_root(v)

        left = cluster_label[ru]
        right = cluster_label[rv]
        merged_size = comp_size[ru] + comp_size[rv]

        single_linkage[i] = HIERARCHY_t(left, right, d, merged_size)

        if ru < rv
            parent[rv] = ru
            comp_size[ru] += comp_size[rv]
            cluster_label[ru] = next_cluster
        else
            parent[ru] = rv
            comp_size[rv] += comp_size[ru]
            cluster_label[rv] = next_cluster
        end

        next_cluster += 1
    end

    return single_linkage
end

################################ _reachability ###############################



dense_distance_matrix = [
    0.0  1.0  4.0  3.0  6.0;
    1.0  0.0  2.0  5.0  7.0;
    4.0  2.0  0.0  1.5  8.0;
    3.0  5.0  1.5  0.0  2.5;
    6.0  7.0  8.0  2.5  0.0
]

I = [1, 2, 2, 3, 3, 4, 4, 5, 1, 4]
J = [2, 1, 3, 2, 4, 3, 5, 4, 4, 1]
V = [1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5, 3.0, 3.0]

sparse_distance_matrix = sparse(I, J, V, 5, 5)
Matrix(sparse_distance_matrix)

function mutual_reachability_graph(
    distance_matrix;
    min_samples::Int = 5,
    max_distance::Float64 = 0.0)

    further_neighbor_idx = min_samples - 1

    if issparse(distance_matrix)
        if !(distance_matrix isa SparseMatrixCSC)
            throw(ArgumentError(
                "Only sparse CSC matrices are supported for `distance_matrix`."
            ))
        end

        _sparse_mutual_reachability_graph(
            distance_matrix.nzval,
            distance_matrix.rowval,
            distance_matrix.colptr,
            size(distance_matrix, 1),
            further_neighbor_idx,
            max_distance,
        )
    else
        _dense_mutual_reachability_graph(
            distance_matrix,
            further_neighbor_idx,
        )
    end

    return distance_matrix
end

#=
    mutual_reachability_graph(dense_distance_matrix, 2, 0.0)
    -> Correct
=#

function _dense_mutual_reachability_graph(distance_matrix,
    further_neighbor_idx::Int)

    n_samples = size(distance_matrix, 1)

    core_distances = Vector{eltype(distance_matrix)}(undef, n_samples)

    for i in 1:n_samples
        row = copy(view(distance_matrix, i, :))
        partialsort!(row, further_neighbor_idx + 1)
        core_distances[i] = row[further_neighbor_idx + 1]
    end

    for i in 1:n_samples
        for j in 1:n_samples
            mutual_reachability_distance = max(
                core_distances[i],
                core_distances[j],
                distance_matrix[i, j])

            distance_matrix[i, j] = mutual_reachability_distance
        end
    end

    return nothing
end

function _sparse_mutual_reachability_graph(
    data,
    rowval,
    colptr,
    n_samples::Int,
    further_neighbor_idx::Int,
    max_distance)

    core_distances = Vector{eltype(data)}(undef, n_samples)

    for col in 1:n_samples
        start_idx = colptr[col]
        end_idx = colptr[col + 1] - 1

        if start_idx <= end_idx
            col_data = data[start_idx:end_idx]

            if further_neighbor_idx < length(col_data)
                tmp = copy(col_data)
                partialsort!(tmp, 1:(further_neighbor_idx + 1))
                core_distances[col] = tmp[further_neighbor_idx + 1]
            else
                core_distances[col] = Inf
            end
        else
            core_distances[col] = Inf
        end
    end

    for col in 1:n_samples
        for k in colptr[col]:(colptr[col + 1] - 1)
            row = rowval[k]   # already 1-based

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

############################### HDBSCAN ######################################

const INFTY = Inf
const NOISE = -1
    

    
    const KD_TREE_VALID_METRICS = Set(["euclidean", "manhattan", "chebyshev", "minkowski"])
    const BALL_TREE_VALID_METRICS = Set(["euclidean", "manhattan", "chebyshev", "minkowski"])
    const FAST_METRICS = union!(KD_TREE_VALID_METRICS, BALL_TREE_VALID_METRICS)

    struct encoding
        label::Int
        prob::Union{Nothing, Int}
    end

    _OUTLIER_ENCODING = Dict(
        "infinite" => encoding(-2, 0),
        "missing" => encoding(-3, nothing))

    mutable struct HDBSCAN
        #Hyperparameters
        min_cluster_size::Int
        min_samples::Union{Nothing,Int}
        cluster_selection_epsilon::Float64
        max_cluster_size::Union{Nothing,Int}
        metric::String
        metric_params::Dict
        alpha::Float64
        algorithm::String
        leaf_size::Int
        n_jobs::Union{Int, Nothing}
        cluster_selection_method::String
        allow_single_cluster::Bool
        store_centers::Union{Nothing,String}
        copy

        #Internal state
        _metric_params::Dict{Any,Any}
        _raw_data
        _min_samples::Union{Int,Nothing}
        _single_linkage_tree::Union{Vector{HIERARCHY_t},Nothing}

        #Outputs
        labels_::Union{Nothing,Vector{Int}}
        probabilities_::Union{Nothing,Vector{Float64}}
        n_features_in_
        feature_names_in_
        centroids_
        medoids_
    end

    function init_HDBSCAN(min_cluster_size::Int=5,
        min_samples::Union{Nothing,Int}=nothing,
        cluster_selection_epsilon::Float64=0.0,
        max_cluster_size::Union{Nothing,Int}=nothing,
        metric::String="euclidean",
        metric_params::Dict=Dict(),
        alpha::Float64=1.0,
        algorithm::String="auto",
        leaf_size::Int=40,
        n_jobs::Union{Int, Nothing}=nothing,
        cluster_selection_method::String="eom",
        allow_single_cluster::Bool=false,
        store_centers::Union{Nothing,String}=nothing,
        copy="warn")
        
        return HDBSCAN(
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
            
            Dict{Any,Any}(),
            nothing,
            nothing,
            nothing,

            Int[],
            Float64[],
            nothing,
            nothing,
            nothing,
            nothing,)
    end

    function _brute_mst(mutual_reachability,
        min_samples::Int)

        if !issparse(mutual_reachability)
            return mst_from_mutual_reachability(mutual_reachability)
        end

        if !(mutual_reachability isa SparseMatrixCSC)
        throw(ArgumentError(
            "Only sparse CSC matrices are supported for mutual_reachability."))
        end

        n_samples = size(mutual_reachability, 1)
        colptr = mutual_reachability.colptr

        for j in 1:n_samples
        nnz_in_col = colptr[j + 1] - colptr[j]
            if nnz_in_col < min_samples
                throw(ArgumentError(
                    "There exist points with fewer than $min_samples neighbors. " *
                    "Ensure your sparse distance matrix has non-zero values for at least " *
                    "min_samples=$min_samples neighbors for each point, or specify " *
                    "a max_distance to use when distances are missing."
                ))
            end
        end

        G = SimpleWeightedGraph(mutual_reachability)
        comps = connected_components(G)

        if length(comps) > 1
            throw(ArgumentError(
                "Sparse mutual reachability matrix has $(length(comps)) connected " *
                "components. HDBSCAN cannot be performed on a disconnected graph."
            ))
        end

        mst_graph = kruskal_mst(G)

        mst = Vector{MST_edge_t}(undef, length(mst_graph))

        for (k, e) in enumerate(mst_graph)
            u = src(e)
            v = dst(e)
            w = weight(G, u, v)

            mst[k] = MST_edge_t(u, v, w)
        end
        return mst
    end

    #=
        _brute_mst(mutual_reachability_graph(dense_distance_matrix, 2, 0.0), 2)
        4-element Vector{MST_edge_t}:
        MST_edge_t(1, 2, 1.0)
        MST_edge_t(2, 3, 2.0)
        MST_edge_t(3, 4, 1.5)
        MST_edge_t(4, 5, 2.5)
        -> Correct
    =#

    function _hdbscan_brute(
    X;
    min_samples=5,
    alpha=1.0,
    metric="euclidean")

    distance_matrix =
        metric == "precomputed" ?
        copy(X) :
        pairwise(_metric_object(metric), X; dims=1)

    distance_matrix ./= alpha

    mutual_reachability_ =
        mutual_reachability_graph(
            distance_matrix;
            min_samples
        )

    min_spanning_tree = _brute_mst(mutual_reachability_, min_samples)

    return _process_mst(min_spanning_tree)
    end

    #=
        tree = _hdbscan_brute(
           X;
           min_samples=2,
           alpha=1.0,
           metric="euclidean"
       )
        4-element Vector{HIERARCHY_t}:
        HIERARCHY_t(1, 2, 1.0, 2)
        HIERARCHY_t(6, 3, 1.0, 3)
        HIERARCHY_t(7, 4, 1.0, 4)
        HIERARCHY_t(8, 5, 2.8284271247461903, 5)
    =#

function _hdbscan_prims(
    X;
    algo::String,
    min_samples::Int=5,
    alpha::Float64=1.0,
    metric::String="euclidean",
    leaf_size::Int=40,
    n_jobs=nothing,
    metric_params...)
    
    tree =
        if algo == "kd_tree"
            KDTree(permutedims(X))
        elseif algo == "ball_tree"
            BallTree(permutedims(X))
        else
            error("Unsupported algorithm: $algo")
        end

    _ , neighbors_distances = knn(
        tree,
        permutedims(X),
        min_samples,
        true
    )

    core_distances = [d[end] for d in neighbors_distances]

    dist_metric =
        if metric == "euclidean"
            Euclidean()
        else
            error("Metric $metric not yet implemented in _hdbscan_prims")
        end

    min_spanning_tree = mst_from_data_matrix(
        X,
        core_distances,
        dist_metric,
        alpha
    )

    return _process_mst(min_spanning_tree)
end

    #=
    X = [
    0.0   0.0;
    0.1   0.0;
    0.0   0.1;
    0.1   0.1;

    10.0 10.0;
    10.1 10.0;
    10.0 10.1;
    10.1 10.1
]

    tree = _hdbscan_prims(
        X;
        min_samples=2,
        alpha=1.0,
        metric="euclidean",
        algo="kd_tree",
        leaf_size=40,
    )

    println("single_linkage_tree =")
    println(tree)
    =#
    
function _process_mst(min_spanning_tree::Vector{MST_edge_t})
    order = sortperm([e.distance for e in min_spanning_tree])
    min_spanning_tree = min_spanning_tree[order]
    return make_single_linkage(min_spanning_tree)
end

    #=
    _process_mst(mst_from_data_matrix(
        [0.0 0.0;
        1.0 0.0;
        0.0 1.0;
        1.0 1.0;
        3.0 3.0],

        [1.0,
        1.0,
        1.0,
        1.0,
        2.5],

        Euclidean(),

        1.0))

    4-element Vector{HIERARCHY_t}:
    HIERARCHY_t(1, 2, 1.0, 2)
    HIERARCHY_t(6, 3, 1.0, 3)
    HIERARCHY_t(7, 4, 1.0, 4)
    HIERARCHY_t(8, 5, 2.8284271247461903, 5)

    -> Correct
    =#


    function fit(model::HDBSCAN, X; y=nothing)

    if model.copy == "warn"
        @warn "The default value of `copy` will change from false to true in a future version. Explicitly set `copy` to silence this warning."
        _copy = false
    else
        _copy = model.copy
    end

    if model.metric == "precomputed" && model.store_centers !== nothing
        throw(ArgumentError(
            "Cannot store centers when using a precomputed distance matrix."
        ))
    end

    model._metric_params = isnothing(model.metric_params) ? Dict() : copy(model.metric_params)

    all_finite = true
    finite_index = nothing
    infinite_index = Int[]
    missing_index = Int[]
    internal_to_raw = Dict{Int,Int}()

    # -----------------------------
    # input handling
    # -----------------------------
    if model.metric != "precomputed"

        X = Matrix{Float64}(X)   # TODO: replace with validate_data equivalent
        model._raw_data = X

        all_finite = all(isfinite, X)

        if !all_finite
            reduced_X = reduce_rows(X)

            missing_index = findall(isnan, reduced_X)
            infinite_index = findall(isinf, reduced_X)

            finite_index = get_finite_row_indices(X)
            internal_to_raw = Dict(i => finite_index[i] for i in eachindex(finite_index))

            X = X[finite_index, :]
        end

    elseif issparse(X)
        throw(ArgumentError("Sparse precomputed matrices not yet supported in this Julia port"))

    else
        X = Matrix{Float64}(X)

        if any(isnan, X)
            throw(ArgumentError("NaN values found in precomputed dense distance matrix"))
        end
    end

    if size(X, 1) == 1
        throw(ArgumentError("n_samples = 1 while HDBSCAN requires more than one sample"))
    end

    model.n_features_in_ = ndims(X) == 2 ? size(X, 2) : size(X, 1)

    model._min_samples =
        isnothing(model.min_samples) ? model.min_cluster_size : model.min_samples

    if model._min_samples > size(X, 1)
        throw(ArgumentError(
            "min_samples ($(model._min_samples)) must be at most the number of samples in X ($(size(X,1)))"
        ))
    end

    mst_func = nothing
    algo = nothing

    # -----------------------------
    # algorithm validation / selection
    # -----------------------------
    if model.algorithm == "kd_tree" && !(model.metric in KD_TREE_VALID_METRICS)
        throw(ArgumentError(
            "$(model.metric) is not a valid metric for a KDTree-based algorithm."
        ))
    elseif model.algorithm == "ball_tree" && !(model.metric in BALL_TREE_VALID_METRICS)
        throw(ArgumentError(
            "$(model.metric) is not a valid metric for a BallTree-based algorithm."
        ))
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

    # -----------------------------
    # backend-specific MST call
    # -----------------------------
    if mst_func === _hdbscan_brute
        model._single_linkage_tree = _hdbscan_brute(
            X;
            min_samples=model._min_samples,
            alpha=model.alpha,
            metric=model.metric
        )

    elseif mst_func === _hdbscan_prims
        model._single_linkage_tree = _hdbscan_prims(
            X;
            min_samples=model._min_samples,
            alpha=model.alpha,
            metric=model.metric,
            leaf_size=model.leaf_size,
            algo=algo
        )

    else
        error("No MST backend selected")
    end

    # -----------------------------
    # tree -> labels
    # -----------------------------
    model.labels_, model.probabilities_ = tree_to_labels(
        model._single_linkage_tree,
        model.min_cluster_size,
        model.cluster_selection_method,
        model.allow_single_cluster,
        model.cluster_selection_epsilon,
        model.max_cluster_size,
    )

    # -----------------------------
    # remap non-finite rows
    # -----------------------------
    if model.metric != "precomputed" && !all_finite
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
        new_probabilities[infinite_index] .= _OUTLIER_ENCODING["infinite"].prob
        new_probabilities[missing_index] .= _OUTLIER_ENCODING["missing"].prob
        model.probabilities_ = new_probabilities
    end

    # -----------------------------
    # optional centers
    # -----------------------------
    if model.store_centers !== nothing
        _weighted_cluster_center(model, X)
    end

    return model
end

#=
        X = [
        0.0   0.0;
        0.1   0.0;
        0.0   0.1;
        0.1   0.1;

        10.0 10.0;
        10.1 10.0;
        10.0 10.1;
        10.1 10.1
    ]

    hdb = init_HDBSCAN(
        2,              # min_cluster_size
        2,              # min_samples
        0.0,            # cluster_selection_epsilon
        nothing,        # max_cluster_size
        "euclidean",    # metric
        Dict(),         # metric_params
        1.0,            # alpha
        "brute",        # algorithm
        40,             # leaf_size
        nothing,        # n_jobs
        "eom",          # cluster_selection_method
        false,          # allow_single_cluster
        "both",         # store_centers
        true            # copy
    )

    fit(hdb, X)

    println("labels_ = ", hdb.labels_)
    println("probabilities_ = ", hdb.probabilities_)
    println("centroids_ = ")
    println(hdb.centroids_)
    println("medoids_ = ")
    println(hdb.medoids_)
    println("_single_linkage_tree = ")
    println(hdb._single_linkage_tree)
=#

    function fit_predict(hdb::HDBSCAN, X)
        fit(hdb, X)
        return hdb.labels_
    end

    #=
    
    =#

    function _get_finite_row_indices(X::SparseMatrixCSC)
        row_mask = trues(size(X,1))

        for col in 1:size(X,2)
            for ptr in X.colptr[col]:(X.colptr[col+1]-1)
                row = X.rowval[ptr]
                if !isfinite(X.nzval[ptr])
                    row_mask[row] = false
                end
            end
        end

        return findall(row_mask)
    end

    #=
        _get_finite_row_indices(sparse_distance_matrix)
        5-element Vector{Int64}:
        1
        2
        3
        4
        5

        -> Correct
    =#

    function remap_single_linkage_tree(
        tree::Vector{HIERARCHY_t},
        internal_to_raw::Dict{Int,Int},
        non_finite::Vector{Bool})

        finite_count = length(internal_to_raw)

        outlier_count = length(non_finite)

        for i in eachindex(tree)
            left = tree[i].left_node
            right = tree[i].right_node

            left_remapped =
                left <= finite_count ?
                internal_to_raw[left] :
                left + outlier_count

            right_remapped =
                right <= finite_count ?
                internal_to_raw[right] :
                right + outlier_count

            tree[i] = HIERARCHY_t(
                left_remapped,
                right_remapped,
                tree[i].value,
                tree[i].cluster_size,)
        end

        outlier_tree = Vector{HIERARCHY_t}(undef, length(non_finite))

        last_cluster_id = max(
            tree[end].left_node,
            tree[end].right_node,
        )
        last_cluster_size = tree[end].cluster_size

        for i in eachindex(non_finite)
            outlier = non_finite[i]

            outlier_node = Int(outlier) + 1

            outlier_tree[i] = HIERARCHY_t(
                outlier_node,
                last_cluster_id + 1,
                Inf,
                last_cluster_size + 1,
            )

            last_cluster_id += 1
            last_cluster_size += 1
        end

        return vcat(tree, outlier_tree)
    end

    #= 
    remap_tree = [
    HIERARCHY_t(1, 2, 1.0, 2),
    HIERARCHY_t(4, 3, 2.0, 3)]

    internal_to_raw = Dict(
    1 => 1,
    2 => 3,
    3 => 4)

    non_finite = [false, true, false, false, true]

    remap_single_linkage_tree(remap_tree, internal_to_raw, non_finite)

    7-element Vector{HIERARCHY_t}:
    HIERARCHY_t(1, 3, 1.0, 2)
    HIERARCHY_t(9, 4, 2.0, 3)
    HIERARCHY_t(1, 10, Inf, 4)
    HIERARCHY_t(2, 11, Inf, 5)
    HIERARCHY_t(1, 12, Inf, 6)
    HIERARCHY_t(1, 13, Inf, 7)
    HIERARCHY_t(2, 14, Inf, 8)

    -> Correct
    =#


function _metric_object(metric::String)
    if metric == "euclidean"
        return Euclidean()
    elseif metric == "manhattan"
        return Cityblock()
    elseif metric == "cityblock"
        return Cityblock()
    elseif metric == "chebyshev"
        return Chebyshev()
    elseif metric == "minkowski"
        return Minkowski()
    else
        throw(ArgumentError("Unsupported metric: $metric"))
    end
end

function _weighted_cluster_center(model::HDBSCAN, X::Matrix{Float64})

    cluster_ids = sort(collect(setdiff(Set(model.labels_), Set([-1, -2]))))
    n_clusters = length(cluster_ids)

    make_centroids = model.store_centers in ("centroid", "both")
    make_medoids   = model.store_centers in ("medoid", "both")

    n_features = size(X, 2)

    if make_centroids
        model.centroids_ = Matrix{Float64}(undef, n_clusters, n_features)
    end

    if make_medoids
        model.medoids_ = Matrix{Float64}(undef, n_clusters, n_features)
    end

    for (idx, cluster_label) in enumerate(cluster_ids)
        mask = model.labels_ .== cluster_label
        data = X[mask, :]
        strength = Float64.(model.probabilities_[mask])

        if make_centroids
            total_weight = sum(strength)

            if total_weight == 0.0
                # fallback: unweighted mean if all strengths are zero
                model.centroids_[idx, :] = vec(mean(data, dims=1))
            else
                centroid = vec(sum(data .* strength, dims=1) ./ total_weight)
                model.centroids_[idx, :] = centroid
            end
        end

        if make_medoids
            n_points = size(data, 1)

            dist_mat = zeros(Float64, n_points, n_points)

            for i in 1:n_points
                for j in 1:n_points
                    dist_metric = _metric_object(model.metric)
                    dist_mat[i, j] = evaluate(dist_metric, view(data, i, :), view(data, j, :))
                end
            end

            weighted_dist = dist_mat .* reshape(strength, 1, :)

            medoid_index = argmin(vec(sum(weighted_dist, dims=2)))
            model.medoids_[idx, :] = data[medoid_index, :]
        end
    end

    return nothing
end
    #=
            X = [
        0.0   0.0;   # cluster 0
        2.0   0.0;   # cluster 0
        10.0 10.0;   # cluster 1
        12.0 10.0;   # cluster 1
        10.0 12.0    # cluster 1
    ]

    hdb = init_HDBSCAN(
        5,                 # min_cluster_size
        nothing,           # min_samples
        0.0,               # cluster_selection_epsilon
        nothing,           # max_cluster_size
        "euclidean",       # metric
        Dict(),            # metric_params
        1.0,               # alpha
        "auto",            # algorithm
        40,                # leaf_size
        nothing,           # n_jobs
        "eom",             # cluster_selection_method
        false,             # allow_single_cluster
        "both",            # store_centers
        "warn"             # copy
    )

    hdb.labels_ = [0, 0, 1, 1, 1]
    hdb.probabilities_ = [1.0, 0.5, 1.0, 0.5, 0.25]

    _weighted_cluster_center(hdb, X)

    println("centroids_ =")
    println(hdb.centroids_)

    [0.05 0.05; 10.05 10.05]

    println("\nmedoids_ =")
    println(hdb.medoids_)

    [0.0 0.0; 10.0 10.0]

    =#