import sklearn.cluster._hdbscan._tree as tree
import numpy as np

print(dir(tree))

'''
['CONDENSED_dtype', 'HIERARCHY_dtype', 'TreeUnionFind', '__builtins__',
'__doc__', '__file__', '__loader__', '__name__', '__package__',
'__pyx_unpickle_TreeUnionFind', '__spec__', '__test__', '_condense_tree',
'_do_labelling', 'get_cluster_tree_leaves', 'labelling_at_cut', 'np',
'recurse_leaf_dfs', 'tree_to_labels']
'''
'''
HIERARCHY_dtype = np.dtype([
    ("left_node", np.intp),
    ("right_node", np.intp),
    ("value", np.float64),
    ("cluster_size", np.intp),
])

CONDENSED_dtype = np.dtype([
    ("parent", np.intp),
    ("child", np.intp),
    ("value", np.float64),
    ("cluster_size", np.intp),
])

hierarchy_test_tree = [
    np.array([(0, 1, 0.1, 2),
    (2, 3, 0.2, 2),
    (4, 5, 0.3, 2),
    (6, 7, 0.4, 2),
    (8, 9, 0.5, 2),
    (10, 11, 0.6, 4),
    (12, 13, 0.7, 4),
    (15, 14, 0.8, 6),
    (17, 16, 1.0, 10),] , dtype=HIERARCHY_dtype)]



def tree_to_labels_test():
    print("Results tree_to_labels_test")
    for i in hierarchy_test_tree:
        print("Test ", i, "for tree_to_labels")
        print(tree.tree_to_labels(i, 4, "eom", False, 0.0, None))


def labelling_at_cut_test():
    print("Results labelling_at_cut_test")
    for i in hierarchy_test_tree:
        print("Test ", i, "for labelling_at_cut")
        print(tree.labelling_at_cut(i, 5, 10))


def recurse_leaf_dfs_test():
    print("Results recurse_leaf_dfs")
    for i in hierarchy_test_tree:
        print("Test ", i, "recurse_leaf_dfs")
        print(tree.recurse_leaf_dfs(tree._condense_tree(i, 10), 15))


def test_tree():
    tree_to_labels_test()
    labelling_at_cut_test()
    recurse_leaf_dfs_test()

test_tree()
'''
'''
Results tree_to_labels_test
Test  [(0, 1, 1., 2)] for tree_to_labels
(array([-1, -1]), array([0., 0.]))
Test  [( 0,  1, 0.1,  2) ( 2,  3, 0.2,  2) ( 4,  5, 0.3,  2) ( 6,  7, 0.4,  2)
 ( 8,  9, 0.5,  2) (10, 11, 0.6,  4) (12, 13, 0.7,  4) (15, 14, 0.8,  6)
 (17, 16, 1. , 10)] for tree_to_labels
(array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0]), array([1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 0.75, 0.75]))


Results labelling_at_cut_test
Test  [(0, 1, 1., 2)] for labelling_at_cut
[-1 -1]
Test  [( 0,  1, 0.1,  2) ( 2,  3, 0.2,  2) ( 4,  5, 0.3,  2) ( 6,  7, 0.4,  2)
 ( 8,  9, 0.5,  2) (10, 11, 0.6,  4) (12, 13, 0.7,  4) (15, 14, 0.8,  6)
 (17, 16, 1. , 10)] for labelling_at_cut
[0 0 0 0 0 0 0 0 0 0]


Results recurse_leaf_dfs
Test  [(0, 1, 1., 2)] recurse_leaf_dfs
[15]
Test  [( 0,  1, 0.1,  2) ( 2,  3, 0.2,  2) ( 4,  5, 0.3,  2) ( 6,  7, 0.4,  2)
 ( 8,  9, 0.5,  2) (10, 11, 0.6,  4) (12, 13, 0.7,  4) (15, 14, 0.8,  6)
 (17, 16, 1. , 10)] recurse_leaf_dfs
[15]
'''

import sklearn.metrics._dist_metrics as DistanceMetrics64
print(dir(DistanceMetrics64))

'''

['CONDENSED_dtype', 'HIERARCHY_dtype', 'TreeUnionFind', '__builtins__',
'__doc__', '__file__', '__loader__', '__name__', '__package__',
'__pyx_unpickle_TreeUnionFind', '__spec__', '__test__', '_condense_tree',
'_do_labelling', 'get_cluster_tree_leaves', 'labelling_at_cut', 'np',
'recurse_leaf_dfs', 'tree_to_labels']
['BOOL_METRICS', 'BrayCurtisDistance32', 'BrayCurtisDistance64',
'CanberraDistance32', 'CanberraDistance64', 'ChebyshevDistance32',
'ChebyshevDistance64', 'DEPRECATED_METRICS', 'DiceDistance32',
'DiceDistance64', 'DistanceMetric', 'DistanceMetric32', 'DistanceMetric64',
'EuclideanDistance32', 'EuclideanDistance64', 'HammingDistance32',
'HammingDistance64', 'HaversineDistance32', 'HaversineDistance64',
'JaccardDistance32', 'JaccardDistance64', 'KulsinskiDistance32',
'KulsinskiDistance64', 'METRIC_MAPPING32', 'METRIC_MAPPING64',
'MahalanobisDistance32', 'MahalanobisDistance64', 'ManhattanDistance32',
'ManhattanDistance64', 'MatchingDistance32', 'MatchingDistance64',
'MinkowskiDistance32', 'MinkowskiDistance64', 'PyFuncDistance32',
'PyFuncDistance64', 'RogersTanimotoDistance32', 'RogersTanimotoDistance64',
'RussellRaoDistance32', 'RussellRaoDistance64', 'SEuclideanDistance32',
'SEuclideanDistance64', 'SokalMichenerDistance32', 'SokalMichenerDistance64',
'SokalSneathDistance32', 'SokalSneathDistance64', '__builtins__', '__doc__',
'__file__', '__loader__', '__name__', '__package__',
'__pyx_unpickle_DistanceMetric', '__spec__', '__test__', 'check_array',
'csr_matrix', 'get_valid_metric_ids', 'issparse', 'newObj', 'np',
'parse_version', 'sp_base_version']
['HIERARCHY_dtype', 'MST_edge_dtype', '__builtins__', '__doc__', '__file__',
'__loader__', '__name__', '__package__', '__spec__', '__test__',
'make_single_linkage', 'mst_from_data_matrix',
'mst_from_mutual_reachability', 'np']

'''
import sklearn.metrics._dist_metrics as distance_metric
import sklearn.cluster._hdbscan._linkage as linkage
print(dir(linkage))

'''
['HIERARCHY_dtype', 'MST_edge_dtype', '__builtins__', '__doc__', '__file__',
'__loader__', '__name__', '__package__', '__spec__', '__test__',
'make_single_linkage', 'mst_from_data_matrix',
'mst_from_mutual_reachability', 'np']


def test_linkage():
    make_single_linkage_test()
    mst_from_data_matrix()
    mst_from_mutual_reachability()
'''
'''
raw_data = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [3.0, 3.0]
], dtype=np.float64)

core_distances = np.array([
    1.0,
    1.0,
    1.0,
    1.0,
    2.5
], dtype=np.float64)

dist_metric = distance_metric.EuclideanDistance64()
alpha = 1.0


data_matrix = linkage.mst_from_data_matrix(raw_data, core_distances,
    dist_metric, alpha)


for i in data_matrix:
    print(*i, sep=" ")
'''
'''
mst = np.array([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 3, 1.5),
    (3, 4, 2.8)], dtype=linkage.MST_edge_dtype)

s_linkage = linkage.make_single_linkage(mst) 

for row in s_linkage:
    print(*row, sep=" ")
'''
'''
0 1 1.0 2
5 2 1.0 3
6 3 1.5 4
7 4 2.8 5
'''

import sklearn.cluster._hdbscan._reachability as reachability
print(dir(reachability))

'''
['__builtins__', '__doc__', '__file__', '__loader__', '__name__', '__package__',
 '__spec__', '__test__', '_dense_mutual_reachability_graph',
 '_sparse_mutual_reachability_graph', 'issparse', 'mutual_reachability_graph',
 'np']
'''