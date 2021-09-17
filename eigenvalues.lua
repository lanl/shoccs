simulation = {
    mesh = {
        index_extents = {21},
        domain_bounds = {1}
    },
    shapes = {
        {
            type = "yz_rect",
            psi = 0.001,
            normal = 1,
            boundary_condition = "dirichlet"
        },
        {
            type = "yz_rect",
            psi = 0.9,
            normal = -1,
            boundary_condition = "floating"
        }
    },
    scheme = {
        order = 1,
        type = "E2-poly",
        floating_alpha = {13 / 100, 7 / 50, 3 / 20, 4 / 25, 17 / 100, 9 / 50},
        dirichlet_alpha = {3 / 25, 13 / 100, 7 / 50}
    },
    system = {
        type = "eigenvalues"
    }
}
