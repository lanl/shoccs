"""Standalone parameter-space exploration sweeps for PHS/RBF stencils.

This package contains scripts that explore parameter spaces (epsilon, sigma,
gamma, nextra) to discover optimal values for RBF-augmented finite difference
stencils. These are research tools, not regression tests.

The workflow:
    1. Run a sweep script to explore a parameter space.
    2. The script prints results and writes optimal values to known_values.json.
    3. Regression tests in tests/test_phs.py load from known_values.json and
       verify stability at those values.

Usage:
    uv run python -m sweeps epsilon --scheme E2
    uv run python -m sweeps tension --scheme E4
    uv run python -m sweeps all --quick
"""
