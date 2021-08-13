dx = 2 / 50

simulation = {
    mesh = {
        index_extents = {51, 51},
        domain_bounds = {2, 2}
    },
    domain_boundaries = {
        xmin = "dirichlet",
        xmax = "dirichlet",
        ymin = "dirichlet",
        ymax = "dirichlet"
    },
    shapes = {
        {
            type = "sphere",
            center = {16 / 17, 25 / 22},
            radius = math.sqrt(3) / 10,
            boundary_condition = "floating"
        }
    },
    scheme = {
        order = 2,
        type = "E2"
    },
    system = {
        type = "heat",
        diffusivity = 1 / 30
    },
    integrator = {
        type = "rk4"
    },
    step_controller = {
        max_time = 0.3,
        cfl = {
            parabolic = 0.5
        }
    },
    manufactured_solution = {
        type = "gaussian",
        --[[
            {
            center = {0.4, 0.5},
            variance = {0.5, 0.2},
            amplitude = 2,
            frequency = 0.5
        },
        {
            center = {0.2, 2},
            variance = {1 / 3, 0.8},
            amplitude = 0.5,
            frequency = 1
        },
        {
            center = {1.5, 1.6},
            variance = {0.3, 0.8},
            amplitude = -1.2,
            frequency = 0.8
        },
        {
            center = {1.8, 0.3},
            variance = {2 / 3, 0.9},
            amplitude = 3,
            frequency = 0.2
        }
        ]]
        {
            center = {1, 1},
            variance = {1, 1},
            amplitude = 2,
            frequency = 0.5
        }
    },
    io = {
        write_every_time = 0.01
    }
}
