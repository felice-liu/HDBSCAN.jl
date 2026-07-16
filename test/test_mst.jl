using Test
using Hdbscan

@testset "UnionFind" begin

    uf = Hdbscan.UnionFind(4)

    @test Hdbscan.uf_find!(uf, 1) == 1
    @test Hdbscan.uf_find!(uf, 2) == 2

    Hdbscan.uf_union!(uf, 1, 2)

    root = uf.next_label-1

    @test Hdbscan.uf_find!(uf, 1) == root
    @test Hdbscan.uf_find!(uf, 2) == root

end

@testset "make_single_linkage" begin

    mst =
        [Hdbscan.MSTEdge(1, 2, 0.1), Hdbscan.MSTEdge(3, 4, 0.2), Hdbscan.MSTEdge(5, 6, 0.3)]

    tree = Hdbscan.make_single_linkage(mst)

    @test tree[1].left_node == 1
    @test tree[1].right_node == 2
    @test tree[1].value == 0.1
    @test tree[1].cluster_size == 2

    @test tree[2].left_node == 3
    @test tree[2].right_node == 4
    @test tree[2].value == 0.2
    @test tree[2].cluster_size == 2

    @test tree[3].cluster_size == 4

end

@testset "mst_from_mutual_reachability" begin

    M = [
        0 1 5 6;
        1 0 2 8;
        5 2 0 3;
        6 8 3 0
    ]

    mst = Hdbscan.mst_from_mutual_reachability(Float64.(M))

    expected =
        [Hdbscan.MSTEdge(1, 2, 1), Hdbscan.MSTEdge(2, 3, 2), Hdbscan.MSTEdge(3, 4, 3)]

    @test mst == expected

end

@testset "_process_mst" begin

    mst =
        [Hdbscan.MSTEdge(1, 2, 3.0), Hdbscan.MSTEdge(2, 3, 1.0), Hdbscan.MSTEdge(3, 4, 2.0)]

    hierarchy = Hdbscan._process_mst(mst)

    @test hierarchy[1].value == 1.0
    @test hierarchy[2].value == 2.0
    @test hierarchy[3].value == 3.0

end
