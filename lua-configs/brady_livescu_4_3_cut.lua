-- Brady-Livescu 2019 §4.3 two-dimensional varying-coefficient scalar wave test
-- with an EMBEDDED SPHERE (cut-cell variant of brady_livescu_4_3.lua; plan 48
-- item 4.4). The exact solution is unchanged — system.center/radius take
-- precedence over shapes for the solution (scalar_wave.cpp:143-147) — while
-- the sphere adds cut geometry whose ψ-parameterized closures and CSR wall
-- coupling are exercised end-to-end. The sphere center/radius are chosen so
-- the realized ψ set spans near-0 to near-1 across N ∈ {31, 61}; the radius
-- default lives in cpp_bridge.CUT_SPHERE_RADIUS and the center must match
-- cpp_bridge.CUT_SPHERE_CENTER (asserted in tests/test_cpp_bridge.py).
--
-- This file serves as a TEMPLATE for scripts/stencil_gen/stencil_gen/cpp_bridge.py.
-- The placeholder tokens in the body (N, T_FINAL, SCHEME_TABLE, RADIUS, CFL
-- wrapped in double-curly-braces inside a Lua line comment) are substituted by
-- make_brady2d_lua before shoccs is invoked. Because those tokens start with
-- double-dashes (Lua line-comment syntax), this file is NOT directly runnable
-- standalone.

simulation = {
    mesh = {
        index_extents = {--{{N}}--, --{{N}}--},
        domain_bounds = {math.sqrt(2), math.sqrt(2)}
    },
    domain_boundaries = {
        xmin = "dirichlet",
        ymin = "dirichlet"
    },
    shapes = {
        {
            type = "sphere",
            center = {0.8441, 0.6177, 0},
            radius = --{{RADIUS}}--,
            boundary_condition = "dirichlet"
        }
    },
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
            hyperbolic = --{{CFL}}--
        }
    },
    io = {
        write_every_step = 10
    }
}
