simulation = {
    mesh = {
        index_extents = {51, 51},
        domain_bounds = {2, 2}
    },
    shapes = {
        {
            type = "sphere",
            center = {1.053, 0.901},
            radius = 0.2,
            boundary_condition = "dirichlet"
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
        max_time = 1000,
        cfl = {
            hyperbolic = 0.1,
            parabolic = 0.2
        }
    },
    io = {
        write_every_time = 5.1
    }
}
