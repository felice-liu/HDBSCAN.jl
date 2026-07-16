using Test
using Hdbscan


@testset "labelling_at_cut" begin

    hierarchy = [
        Hdbscan.HierarchyTree(1, 2, 0.1, 2),
        Hdbscan.HierarchyTree(3, 4, 0.2, 2),
        Hdbscan.HierarchyTree(5, 6, 5.0, 4),
    ]

    labels = Hdbscan.labelling_at_cut(hierarchy, 0.15, 2)

    @test labels == [0, 0, -1, -1]

end
