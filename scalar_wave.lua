simulation = {
    mesh = {
        index_extents = {71, 71},
        domain_bounds = {2, 2}
    },
    domain_boundaries = {
        xmin = "dirichlet",
        ymin = "dirichlet"
    },
    shapes = {
        {
            type = "sphere",
            center = {0, 0}, --1.053, 0.901},
            radius = math.pi / 10,
            boundary_condition = "dirichlet"
        },
        {
            type = "sphere",
            center = {2, 2},
            radius = math.pi / 10,
            boundary_condition = "floating"
        }
    },
    scheme = {
        order = 1,
        type = "E2",
        alpha = {-1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644}
    },
    system = {
        type = "scalar wave"
        -- diffusivity = 1 / 30
    },
    integrator = {
        type = "rk4"
    },
    step_controller = {
        max_time = 0.2,
        --max_step = 2,
        cfl = {
            hyperbolic = 0.8,
            parabolic = 0.2
        }
    },
    io = {
        write_every_step = 1
        --write_every_time = 0.1
    }
}
