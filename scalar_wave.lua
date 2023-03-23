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
          center = {0, 0},
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
        type = "E2-poly",
        floating_alpha = {
           0.7039278390946743,
           0.5390086175376538,
              -0.647109821986589,
           0.2051508287133347,
           0.6062051039572746,
           0.8148425279273044
        },
        dirichlet_alpha = {
           -0.10739761225713096, 0.8736492896991024, 0.40413606410467495
        },
        interpolant_alpha = {0.01, -0.01, 0.01, 0.04}
    },
    system = {
       type = "scalar wave",
       center = {-1, -1},
       radius = 0,
       max_error = 2.0
    },
    integrator = {
        type = "rk4"
    },
    step_controller = {
        max_time = 500,
        max_step = 20,
        cfl = {
            hyperbolic = 0.8,
            parabolic = 0.2
        }
    },
    io = {
        write_every_step = 1
        --write_every_time = 1.1
    }
}
