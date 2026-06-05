-- Brady-Livescu 2019 §4.3 two-dimensional varying-coefficient scalar wave test.
-- Uniform rectangular domain (no embedded geometry). Radial flow field
-- emanating from center = (-0.25, -0.25). Exact solution matches
-- scalar_wave.cpp's solution_at: sin(2*pi*(|x-c| - r - t)).
--
-- This file serves as a TEMPLATE for scripts/stencil_gen/stencil_gen/cpp_bridge.py.
-- The placeholder tokens in the body (N, T_FINAL, SCHEME_TABLE wrapped in
-- double-curly-braces inside a Lua line comment) are substituted by
-- make_brady2d_lua before shoccs is invoked. Because those tokens start with
-- double-dashes (Lua line-comment syntax), this file is NOT directly runnable
-- standalone; use brady_livescu_4_3_n61.lua or brady_livescu_4_3_long.lua for
-- standalone runs.

simulation = {
    mesh = {
        index_extents = {--{{N}}--, --{{N}}--},
        domain_bounds = {math.sqrt(2), math.sqrt(2)}
    },
    domain_boundaries = {
        xmin = "dirichlet",
        ymin = "dirichlet"
    },
    shapes = {},
    scheme = --{{SCHEME_TABLE}}--,
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
        max_time = --{{T_FINAL}}--,
        cfl = {
            hyperbolic = 0.8
        }
    },
    io = {
        write_every_step = 10
    }
}
