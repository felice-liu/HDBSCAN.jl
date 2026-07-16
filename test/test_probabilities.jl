using Test
using Hdbscan

@testset "get_probabilities" begin

    tree = Hdbscan.CondensedTree[
        Hdbscan.CondensedTree(6, 1, 2.0, 1),
        Hdbscan.CondensedTree(6, 2, 4.0, 1),
        Hdbscan.CondensedTree(6, 3, 1.0, 1),
    ]

    cluster_map = Dict(0 => 6)

    labels = [0, 0, 0]

    probs = Hdbscan.get_probabilities(tree, cluster_map, labels)

    @test probs[1] == 0.5
    @test probs[2] == 1.0
    @test probs[3] == 0.25

end
