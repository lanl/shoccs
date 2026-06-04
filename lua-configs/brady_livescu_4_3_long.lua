-- Brady-Livescu 2019 §4.3 two-dimensional varying-coefficient scalar wave test.
-- Long-time (max_time=100) variant of brady_livescu_4_3.lua at N=31 for
-- stability checks beyond the short-run smoke test.

simulation = {
    mesh = {
        index_extents = {31, 31},
        domain_bounds = {math.sqrt(2), math.sqrt(2)}
    },
    domain_boundaries = {
        xmin = "dirichlet",
        ymin = "dirichlet"
    },
    shapes = {},
    scheme = {
        order = 1,
        type = "E4u",
        alpha = {-0.7733323791884821, 0.1623961700641681}
    },
    system = {
        type = "scalar wave",
        center = {-0.25, -0.25, 0},
        radius = 0,
        max_error = 10.0
    },
    integrator = {
        type = "rk4"
    },
    step_controller = {
        max_time = 100.0,
        cfl = {
            hyperbolic = 0.8
        }
    },
    io = {
        write_every_step = 10
    }
}
