const INFTY = Inf
const NOISE = -1

mutable struct HIERARCHY_t
    left_node::Int
    right_node::Int
    value::Float64
    cluster_size::Int
end

mutable struct CONDENSED_t
    parent::Int
    child::Int
    value::Float64
    cluster_size::Int
end


function tree_to_labels(single_linkage_tree,
    min_cluster_size::Int=10,
    cluster_selection_method="eom",
    allow_single_cluster::Bool=false,
    cluster_selection_epsilon::Float64=0.0,
    max_cluster_size=nothing,)

    condensed_tree = _condense_tree(single_linkage_tree, min_cluster_size)
    labels, probabilities = _get_clusters(condensed_tree,
        _compute_stability(condensed_tree),
        cluster_selection_method,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,)

    return (labels, probabilities)
end

function bfs_from_hierarchy(hierarchy, bfs_root)

    process_queue = [bfs_root]#18
    result = []
    n_samples = length(hierarchy) + 1 #10

    while !isempty(process_queue)
        append!(result, process_queue)
        process_queue = [x - n_samples for x in process_queue if x >= n_samples]

        if !isempty(process_queue)
            next_queue = []
            for node in process_queue
                push!(next_queue, hierarchy[node + 1].left_node)
                push!(next_queue, hierarchy[node + 1].right_node)
            end
            process_queue = next_queue
        end
    end
    return result
end

#= test_tree, bfs_root = 18 -> 19-element Vector{Any}:
 18
 17
 16
 15
 14
 12
 13
 10
 11
  8
  9
  4
  5
  6
  7
  0
  1
  2
  3
 Validato con Python =#


function _condense_tree(hierarchy, min_cluster_size::Int=10)

    root = 2 * length(hierarchy)
    n_samples = length(hierarchy) + 1
    next_label = n_samples + 1 #11

    node_list = bfs_from_hierarchy(hierarchy, root)

    relabel = zeros(Int, root + 1)
    relabel[root + 1] = n_samples

    result_list = []
    ignore = falses(length(node_list))

    for node in node_list
        if ignore[node + 1] || node < n_samples
            continue
        end

        children = hierarchy[node - n_samples + 1]
        left = children.left_node
        right = children.right_node
        distance = children.value

        if distance > 0.0
            lambda_value = 1.0 / distance
        else
            lambda_value = INFTY
        end

        if left >= n_samples
            left_count = hierarchy[left - n_samples + 1].cluster_size
        else
            left_count = 1
        end
        
        if right >= n_samples
            right_count = hierarchy[right - n_samples + 1].cluster_size
        else
            right_count = 1
        end

        if left_count >= min_cluster_size && right_count >= min_cluster_size
            relabel[left + 1] = next_label
            next_label += 1
            push!(result_list, CONDENSED_t(relabel[node + 1], relabel[left + 1],
                lambda_value, left_count))

            relabel[right + 1] = next_label
            next_label += 1
            push!(result_list, CONDENSED_t(relabel[node + 1],
                relabel[right + 1], lambda_value, right_count))

        elseif left_count < min_cluster_size && right_count < min_cluster_size
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node < n_samples
                    push!(result_list, CONDENSED_t(relabel[node + 1],
                        sub_node, lambda_value, 1))
                end
                ignore[sub_node + 1] = true
            end

            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node < n_samples
                    push!(result_list, CONDENSED_t(relabel[node + 1],
                        sub_node, lambda_value, 1))
                end
                ignore[sub_node + 1] = true
            end

        elseif left_count < min_cluster_size
            relabel[right + 1] = relabel[node + 1]
            for sub_node in bfs_from_hierarchy(hierarchy, left)
                if sub_node < n_samples
                    push!(result_list, CONDENSED_t(relabel[node + 1],
                    sub_node, lambda_value, 1))
                end
                ignore[sub_node + 1] = true
            end

        else
            relabel[left + 1] = relabel[node + 1]
            for sub_node in bfs_from_hierarchy(hierarchy, right)
                if sub_node < n_samples
                    push!(result_list, CONDENSED_t(relabel[node + 1],
                    sub_node,lambda_value, 1))
                end
                ignore[sub_node + 1] = true
            end
        end
    end

    return result_list
end

#=
_condense_tree(hierarchy_test_tree_python, 10)
10-element Vector{Any}:
 CONDENSED_t(10, 8, 1.0, 1)
 CONDENSED_t(10, 9, 1.0, 1)
 CONDENSED_t(10, 0, 1.0, 1)
 CONDENSED_t(10, 1, 1.0, 1)
 CONDENSED_t(10, 2, 1.0, 1)
 CONDENSED_t(10, 3, 1.0, 1)
 CONDENSED_t(10, 4, 1.0, 1)
 CONDENSED_t(10, 5, 1.0, 1)
 CONDENSED_t(10, 6, 1.0, 1)
 CONDENSED_t(10, 7, 1.0, 1)
 Validato con Python
 =#

function _compute_stability(condensed_tree)
    parents = [c.parent for c in condensed_tree]

    largest_child = maximum([c.child for c in condensed_tree]) #9
    smallest_cluster = minimum(parents) #10
    num_clusters = maximum(parents) - smallest_cluster + 1 # 1

    largest_child = max(largest_child, smallest_cluster)
    births = fill(NaN, largest_child + 1)

    for idx in eachindex(condensed_tree)
        condensed_node = condensed_tree[idx]
        births[condensed_node.child + 1] = condensed_node.value
    end

    births[smallest_cluster + 1] = 0.0

    result = zeros(Float64, maximum(parents) + 1)

    for idx in eachindex(condensed_tree)
        condensed_node = condensed_tree[idx]
        parent = condensed_node.parent + 1
        lambda_val = condensed_node.value
        cluster_size = condensed_node.cluster_size
        
        result_index = parent - smallest_cluster
        result[result_index] = result[result_index] +
            ((lambda_val - births[parent]) * cluster_size)
    end

    stability_dict = Dict{Int, Float64}()

    for idx in 1:num_clusters
        stability_dict[idx + smallest_cluster - 1] = result[idx]
    end

    return stability_dict
end

#= Dict{Int64, Float64} with 1 entry:
  10 => 10.0 
Validato con python  
=#

function bfs_from_cluster_tree(condensed_tree, bfs_root)
    result = []
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

#=

11-element Vector{Any}:
 10
  8
  9
  0
  1
  2
  3
  4
  5
  6
  7

  Validato con Python
=#

function max_lambdas(condensed_tree)
    largest_parent = maximum([c.parent for c in condensed_tree]) + 1
    deaths = zeros(Float64, largest_parent)

    current_parent = condensed_tree[1].parent + 1
    max_lambda = condensed_tree[1].value

    for i in 2:length(condensed_tree)
        parent = condensed_tree[i].parent + 1
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

#=
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
=#

mutable struct TreeUnionFind
    data::Array{Int,2}
    is_component::Vector{Bool}
end

function init_TreeUnionFind(size)
    data = zeros(Int, size, 2)
    for i in 1:size
        data[i,1] = i - 1
    end
    is_component = trues(size)
    tuf = TreeUnionFind(data, is_component)
    return tuf
end

#= size 5
TreeUnionFind([0 0; 1 0; Åc ; 3 0; 4 0], [1, 1, 1, 1, 1])
=#


function union(tuf, x, y)

    x += 1
    y += 1

    x_root = find(tuf, x - 1) + 1
    y_root = find(tuf, y - 1) + 1

    if tuf.data[x_root,2] < tuf.data[y_root, 2]
        tuf.data[x_root,1] = y_root -1
    elseif tuf.data[x_root,2] > tuf.data[y_root, 2]
        tuf.data[y_root,1] = x_root -1
    else
        tuf.data[y_root,1] = x_root - 1
    end
    return tuf
    #print(tuf.data)
end

#= union(tuf, 2, 4)
print -> [1 0; 2 2; 3 0; 2 0; 5 0]
=#

function find(tuf, x)
    x = x + 1
    if tuf.data[x,1] != x - 1
        tuf.data[x,1] = find(tuf, tuf.data[x,1])
        tuf.is_component[x] = false
    end
    return tuf.data[x,1]
end

# find(tuf, 3) -> 3


function labelling_at_cut(linkage, cut, min_cluster_size)
    root = 2 * length(linkage) + 1 #1 index
    n_samples = div(root, 2) + 1

    result = zeros(Int, n_samples)
    union_find = init_TreeUnionFind(root - 1)

    cluster = n_samples
    for node in linkage
        if node.value < cut
            union_find = union(union_find, node.left_node, cluster - 1)
            union_find = union(union_find, node.right_node, cluster - 1)
        end
        cluster += 1
    end

    cluster_size = zeros(Int, cluster + 1)

    for n in 0:n_samples
        cluster = find(union_find, n)
        cluster_size[cluster] += 1
        result[n + 1] = cluster
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

function _do_labelling(condensed_tree,
    clusters,
    cluster_label_map,
    allow_single_cluster,
    cluster_selection_epsilon)

    child_array = [c.child for c in condensed_tree]
    parent_array = [c.parent for c in condensed_tree]
    lambda_array = [c.value for c in condensed_tree]

    root_cluster = minimum(parent_array)
    result = fill(0, root_cluster)

    union_find = init_TreeUnionFind(maximum(parent_array) + 1)

    for i in eachindex(condensed_tree)
        child = child_array[i]
        parent = parent_array[i]
        if !(child in clusters)
            union(union_find, parent, child)
        end
    end

    for n in 1:root_cluster
        cluster = find(union_find, n)
        label = NOISE

        if cluster != root_cluster
            label = cluster_label_map[cluster]

        elseif length(clusters) == 1 && allow_single_cluster
            parent_lambda = maximum([lambda_array[i]
                for i in eachindex(child_array)
                if child_array[i] == n])

            if cluster_selection_epsilon != 0.0
                threshold = 1 / cluster_selection_epsilon
            else
                threshold = maximum([lambda_array[i]
                    for i in eachindex(parent_array)
                    if parent_array[i] == cluster])
            end

            if parent_lambda >= threshold
                label = cluster_label_map[cluster]
            end
        end

        result[n] = label
    end

    return result
end

function get_probabilities(condensed_tree, cluster_map, labels)

    child_array = [c.child for c in condensed_tree]
    parent_array = [c.parent for c in condensed_tree]
    lambda_array = [c.value for c in condensed_tree]

    result = zeros(Float64, length(labels))
    deaths = max_lambdas(condensed_tree)
    root_cluster = minimum(parent_array) + 1

    for i in eachindex(condensed_tree)
        point = child_array[i] + 1
        if point >= root_cluster
            continue
        end

        cluster_num = labels[point]
        if cluster_num == -1
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
        println(
    "point=", point,
    " label=", cluster_num,
    " cluster=", cluster,
    " lambda=", lambda_array[i],
    " death=", max_lambda
)
    end
    
    return result
end

#= get_probabilities([CONDENSED_t(10, 0, 0.4, 1),
    CONDENSED_t(10, 1, 0.7, 1),
    CONDENSED_t(11, 2, 1.2, 1),
    CONDENSED_t(11, 3, 0.9, 1),
    CONDENSED_t(12, 4, 2.0, 1),],
    Dict(0 => 10,
        1 => 11,
        2 => 12),
    [0, 0, 1, 1, 2]

)
=#


function recurse_leaf_dfs(cluster_tree, current_node)
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

#=  recurse_leaf_dfs(_condense_tree(hierarchy_test_tree_python), 10)
10-element Vector{Any}:
 8
 9
 0
 1
 2
 3
 4
 5
 6
 7
 =#

function get_cluster_tree_leaves(cluster_tree)

    if isempty(cluster_tree)
        return []
    end
    root = minimum([c.parent for c in cluster_tree])
    return recurse_leaf_dfs(cluster_tree, root)
end

#=
10-element Vector{Any}:
 8
 9
 0
 1
 2
 3
 4
 5
 6
 7
=#

function traverse_upwards(cluster_tree,
    cluster_selection_epsilon,
    leaf,
    allow_single_cluster)

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
traverse_upwards(_condense_tree(hierarchy_test_tree_python), 0.5, 5, true)
10
=#

function epsilon_search(leaves,
    cluster_tree,
    cluster_selection_epsilon,
    allow_single_cluster)

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

function _get_clusters(condensed_tree,
    stability,
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
        for c in condensed_tree if c.cluster_size == 1]) + 1

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
            subtree_stability = sum(stability[c] for c in children)

            if subtree_stability > stability[node] ||
                cluster_sizes[node] > max_cluster_size

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

# Testing

hierarchy_test_tree_python = [HIERARCHY_t(0, 1, 0.1, 2),
        HIERARCHY_t(2, 3, 0.2, 2),
        HIERARCHY_t(4, 5, 0.3, 2),
        HIERARCHY_t(6, 7, 0.4, 2),
        HIERARCHY_t(8, 9, 0.5, 2),
        HIERARCHY_t(10, 11, 0.6, 4),
        HIERARCHY_t(12, 13, 0.7, 4),
        HIERARCHY_t(15, 14, 0.8, 6),
        HIERARCHY_t(17, 16, 1.0, 10)]