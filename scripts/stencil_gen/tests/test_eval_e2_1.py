"""Regression test for the E2_1.cpp fixture evaluator tool."""

import os
import sys

import pytest

# Add tools/ to path so we can import the evaluator
_tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, _tools_dir)

from eval_e2_1 import make_eval_function  # noqa: E402


@pytest.fixture(scope="module")
def eval_fn():
    """Build the E2_1 evaluator from the C++ source."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cpp_path = os.path.join(repo_root, "src", "stencils", "E2_1.cpp")
    if not os.path.exists(cpp_path):
        pytest.skip(f"E2_1.cpp not found at {cpp_path}")
    fn, _src = make_eval_function(cpp_path)
    return fn


# Expected h=1 left-boundary values at h=2, psi=1.0, alpha=[1,2,3,-1].
# These are C++ nbs_floating output (already /h), so *h recovers h=1 values.
# The raw C++ /h values are listed; the test multiplies by h=2 internally.
EXPECTED_H2 = [
    3, -5, 0.5, 1.5, 0,
    -0.5, 0, 1, -0.5, 0,
    0.02631578947368421, -0.32894736842105265,
    0.07894736842105263, 0.2236842105263158,
    0, 0, 0, -0.25, 0, 0.25,
]


def test_h2_psi1_alpha_1_2_3_m1(eval_fn):
    """Evaluate at h=2, psi=1.0, alpha=[1,2,3,-1] and compare against expected."""
    result = eval_fn(h=2, psi=1.0, alpha=[1, 2, 3, -1])
    assert len(result) == 20
    for k in range(20):
        assert abs(result[k] - EXPECTED_H2[k]) < 1e-12, (
            f"c[{k}]: got {result[k]!r}, expected {EXPECTED_H2[k]!r}"
        )


def test_alpha_zero_no_crash(eval_fn):
    """alpha=[0,0,0,0] at psi=0.5 should not crash."""
    result = eval_fn(h=1, psi=0.5, alpha=[0, 0, 0, 0])
    assert len(result) == 20
    # All values should be finite
    for k, v in enumerate(result):
        assert v == v, f"c[{k}] is NaN"  # NaN != NaN


def test_singular_alpha_at_psi1(eval_fn):
    """alpha=[1,0,0,0] at psi=1.0 is known to be singular (division by zero)."""
    with pytest.raises(ZeroDivisionError):
        eval_fn(h=1, psi=1.0, alpha=[1, 0, 0, 0])
