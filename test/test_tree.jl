using Test
using Hdbscan

@testset "bfs_from_hierarchy balanced" begin

    hierarchy = [
        Hdbscan.HierarchyTree(1, 2, 0.1, 2),
        Hdbscan.HierarchyTree(3, 4, 0.2, 2),
        Hdbscan.HierarchyTree(5, 6, 0.3, 4),
    ]

    # root = 7

    @test Hdbscan.bfs_from_hierarchy(hierarchy, 7) == [7, 5, 6, 1, 2, 3, 4]

end

@testset "bfs_from_hierarchy chain" begin

    hierarchy = [
        Hdbscan.HierarchyTree(1, 2, 0.1, 2),
        Hdbscan.HierarchyTree(3, 4, 0.2, 3),
        Hdbscan.HierarchyTree(5, 6, 0.3, 4),
        Hdbscan.HierarchyTree(7, 8, 0.4, 5),
    ]

    @test Hdbscan.bfs_from_hierarchy(hierarchy, 9) == [9, 7, 8, 3, 4, 5, 6, 1, 2]

end

@testset "_compute_stability" begin

    tree = Hdbscan.CondensedTree[
        Hdbscan.CondensedTree(6, 7, 2.0, 5),
        Hdbscan.CondensedTree(6, 8, 2.0, 5),
        Hdbscan.CondensedTree(7, 1, 3.0, 1),
        Hdbscan.CondensedTree(7, 2, 3.0, 1),
        Hdbscan.CondensedTree(8, 3, 4.0, 1),
        Hdbscan.CondensedTree(8, 4, 4.0, 1),
    ]

    stability = Hdbscan._compute_stability(tree)

    @test stability[6] == 20.0
    @test stability[7] == 2.0
    @test stability[8] == 4.0

end

@testset "max_lambdas" begin

    tree = Hdbscan.CondensedTree[
        Hdbscan.CondensedTree(6, 7, 2.0, 5),
        Hdbscan.CondensedTree(6, 8, 5.0, 5),
        Hdbscan.CondensedTree(7, 1, 3.0, 1),
        Hdbscan.CondensedTree(7, 2, 4.0, 1),
    ]

    deaths = Hdbscan.max_lambdas(tree)

    @test deaths[6] == 5.0
    @test deaths[7] == 4.0

end

@testset "TreeUnionFind" begin

    uf = Hdbscan.TreeUnionFind(8)

    @test Hdbscan.tuf_find!(uf, 1) == 1
    @test Hdbscan.tuf_find!(uf, 2) == 2

    Hdbscan.tuf_union!(uf, 1, 2)

    @test Hdbscan.tuf_find!(uf, 1) == Hdbscan.tuf_find!(uf, 2)

    Hdbscan.tuf_union!(uf, 2, 3)

    @test Hdbscan.tuf_find!(uf, 3) == Hdbscan.tuf_find!(uf, 1)

end

@testset "TreeUnionFind path compression" begin

    uf = Hdbscan.TreeUnionFind(10)

    Hdbscan.tuf_union!(uf, 1, 2)
    Hdbscan.tuf_union!(uf, 2, 3)
    Hdbscan.tuf_union!(uf, 3, 4)

    root = Hdbscan.tuf_find!(uf, 4)

    @test uf.data[4, 1] == root

end
