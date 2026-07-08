using Test
using HDBSCAN


@testset "labelling_at_cut" begin

    hierarchy = [
        HDBSCAN.HierarchyTree(1, 2, 0.1, 2),
        HDBSCAN.HierarchyTree(3, 4, 0.2, 2),
        HDBSCAN.HierarchyTree(5, 6, 5.0, 4),
    ]

    labels = HDBSCAN.labelling_at_cut(hierarchy, 0.15, 2)

    @test labels == [0, 0, -1, -1]

end
