using Test
using HDBSCAN

@testset "bfs_from_hierarchy balanced" begin

    hierarchy = [
        HDBSCAN.HierarchyTree(1, 2, 0.1, 2),
        HDBSCAN.HierarchyTree(3, 4, 0.2, 2),
        HDBSCAN.HierarchyTree(5, 6, 0.3, 4),
    ]

    # root = 7

    @test HDBSCAN.bfs_from_hierarchy(hierarchy, 7) == [7, 5, 6, 1, 2, 3, 4]

end

@testset "bfs_from_hierarchy chain" begin

    hierarchy = [
        HDBSCAN.HierarchyTree(1, 2, 0.1, 2),
        HDBSCAN.HierarchyTree(3, 4, 0.2, 3),
        HDBSCAN.HierarchyTree(5, 6, 0.3, 4),
        HDBSCAN.HierarchyTree(7, 8, 0.4, 5),
    ]

    @test HDBSCAN.bfs_from_hierarchy(hierarchy, 9) == [9, 7, 8, 3, 4, 5, 6, 1, 2]

end

@testset "_compute_stability" begin

    tree = HDBSCAN.CondensedTree[
        HDBSCAN.CondensedTree(6, 7, 2.0, 5),
        HDBSCAN.CondensedTree(6, 8, 2.0, 5),
        HDBSCAN.CondensedTree(7, 1, 3.0, 1),
        HDBSCAN.CondensedTree(7, 2, 3.0, 1),
        HDBSCAN.CondensedTree(8, 3, 4.0, 1),
        HDBSCAN.CondensedTree(8, 4, 4.0, 1),
    ]

    stability = HDBSCAN._compute_stability(tree)

    @test stability[6] == 20.0
    @test stability[7] == 2.0
    @test stability[8] == 4.0

end

@testset "max_lambdas" begin

    tree = HDBSCAN.CondensedTree[
        HDBSCAN.CondensedTree(6, 7, 2.0, 5),
        HDBSCAN.CondensedTree(6, 8, 5.0, 5),
        HDBSCAN.CondensedTree(7, 1, 3.0, 1),
        HDBSCAN.CondensedTree(7, 2, 4.0, 1),
    ]

    deaths = HDBSCAN.max_lambdas(tree)

    @test deaths[6] == 5.0
    @test deaths[7] == 4.0

end

@testset "TreeUnionFind" begin

    uf = HDBSCAN.TreeUnionFind(8)

    @test HDBSCAN.tuf_find!(uf, 1) == 1
    @test HDBSCAN.tuf_find!(uf, 2) == 2

    HDBSCAN.tuf_union!(uf, 1, 2)

    @test HDBSCAN.tuf_find!(uf, 1) == HDBSCAN.tuf_find!(uf, 2)

    HDBSCAN.tuf_union!(uf, 2, 3)

    @test HDBSCAN.tuf_find!(uf, 3) == HDBSCAN.tuf_find!(uf, 1)

end

@testset "TreeUnionFind path compression" begin

    uf = HDBSCAN.TreeUnionFind(10)

    HDBSCAN.tuf_union!(uf, 1, 2)
    HDBSCAN.tuf_union!(uf, 2, 3)
    HDBSCAN.tuf_union!(uf, 3, 4)

    root = HDBSCAN.tuf_find!(uf, 4)

    @test uf.data[4, 1] == root

end
