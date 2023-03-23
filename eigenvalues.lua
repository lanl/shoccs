local floating_success = {
   0.7039278390946743,
   0.5390086175376538,
      -0.647109821986589,
   0.2051508287133347,
   0.6062051039572746,
   0.8148425279273044
}

local dirichlet_success = {
      -0.10739761225713096, 0.8736492896991024, 0.40413606410467495
}

local uniform = 1 - 1e-8


simulation = {
    mesh = {
        index_extents = {21},
        domain_bounds = {1}
    },
    shapes = {
        {
            type = "yz_rect",
            psi = uniform,
            normal = 1,
            boundary_condition = "dirichlet"
        },
        {
            type = "yz_rect",
            psi = uniform,
            normal = -1,
            boundary_condition = "floating"
        }
    },
    scheme = {
        order = 1,
        type = "E2-poly",
        floating_alpha = floating_success,
        dirichlet_alpha = dirichlet_success
    },
    system = {
        type = "eigenvalues"
    }
}
