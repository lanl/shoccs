#!/usr/bin/env python3
"""
Parse the CSE assignments from E2_1.cpp nbs_floating method
and evaluate the stencil at various test points to produce fixture data.
"""

import re
import os


def parse_nbs_floating(cpp_path):
    """
    Read E2_1.cpp, extract lines from nbs_floating between 'double t3' and
    the last 'c[19]' assignment, and convert them to Python evaluable code.
    Returns a list of Python code lines (strings).
    """
    with open(cpp_path, "r") as f:
        lines = f.readlines()

    # Find the nbs_floating method and extract the body
    in_method = False
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if not in_method:
            if "nbs_floating" in stripped and "real h" in stripped:
                in_method = True
            continue

        # Collect lines from 'double t3' through 'c[19]'
        if not body_lines and not stripped.startswith("double t"):
            continue

        body_lines.append(stripped)

        if re.match(r"c\[19\]\s*=", stripped):
            break

    # Now convert the collected C++ lines to Python
    python_lines = []
    # We may have multi-line statements (continuations). Join them first.
    joined = []
    current = ""
    for line in body_lines:
        if not line:
            continue
        current += " " + line if current else line
        if current.endswith(";"):
            joined.append(current)
            current = ""
    if current:
        joined.append(current)

    for stmt in joined:
        py = cpp_to_python(stmt)
        if py is not None:
            python_lines.append(py)

    return python_lines


def cpp_to_python(stmt):
    """Convert a single C++ statement to Python."""
    stmt = stmt.strip()
    if not stmt:
        return None

    # Remove trailing semicolon
    if stmt.endswith(";"):
        stmt = stmt[:-1].strip()

    # Handle 'double tN = expr;' -> 'tN = expr'
    m = re.match(r"double\s+(t\d+)\s*=\s*(.+)", stmt)
    if m:
        var = m.group(1)
        expr = convert_expr(m.group(2).strip())
        return f"{var} = {expr}"

    # Handle 'c[N] = expr;' -> 'c[N] = expr'
    # Also handle multi-line c[N] = ... patterns
    m = re.match(r"(c\[\d+\])\s*=\s*(.+)", stmt)
    if m:
        var = m.group(1)
        expr = convert_expr(m.group(2).strip())
        return f"{var} = {expr}"

    return None


def convert_expr(expr):
    """Convert a C++ expression to Python."""
    # Replace std::pow(x, N) with x**(N)
    # Handle nested calls by doing multiple passes.
    # Guard against infinite loop: if the regex fails to substitute, break.
    max_iterations = 100
    for _ in range(max_iterations):
        if "std::pow(" not in expr:
            break
        prev = expr
        expr = re.sub(
            r"std::pow\(([^,()]+(?:\([^()]*\))?[^,()]*),\s*([^()]+)\)",
            r"(\1)**(\2)",
            expr,
        )
        if expr == prev:
            raise ValueError(
                f"Failed to convert std::pow in expression: {expr!r}"
            )
    else:
        raise ValueError(
            f"Too many std::pow nesting levels (>{max_iterations}): {expr!r}"
        )

    # Replace 'alpha[N]' with 'alpha_N' - we'll use individual variables
    expr = re.sub(r"alpha\[(\d+)\]", r"alpha[\1]", expr)

    return expr


def build_eval_function(python_lines):
    """
    Build a Python function string from the parsed lines.
    Returns the function source code.
    """
    func_lines = [
        "def eval_nbs_floating(h, psi, alpha):",
        "    c = [0.0] * 20",
    ]

    for line in python_lines:
        func_lines.append(f"    {line}")

    # Apply v /= h for all c entries
    func_lines.append("    c = [v / h for v in c]")
    func_lines.append("    return c")

    return "\n".join(func_lines)


def make_eval_function(cpp_path):
    """Parse the C++ file and return a callable eval_nbs_floating function."""
    python_lines = parse_nbs_floating(cpp_path)
    func_src = build_eval_function(python_lines)

    # Compile and execute to get the function
    namespace = {}
    exec(func_src, namespace)
    return namespace["eval_nbs_floating"], func_src


def main():
    # Find E2_1.cpp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    cpp_path = os.path.join(repo_root, "src", "stencils", "E2_1.cpp")

    if not os.path.exists(cpp_path):
        print(f"ERROR: Cannot find {cpp_path}")
        return

    print(f"Parsing {cpp_path} ...")
    eval_fn, func_src = make_eval_function(cpp_path)

    # Print the generated function for inspection
    print("\n=== Generated Python function (first 30 lines) ===")
    src_lines = func_src.split("\n")
    for line in src_lines[:30]:
        print(line)
    print(f"  ... ({len(src_lines)} lines total)")

    # Test cases
    # Note: alpha=[1,0,0,0] and alpha=[0,1,0,0] produce singular denominators
    # (t510=0 or t61=0), so we use small nonzero values for the other components.
    test_cases = [
        {"h": 1, "psi": 1.0, "alpha": [0, 0, 0, 0]},
        {"h": 1, "psi": 1.0, "alpha": [1, 0, 0, 0]},
        {"h": 1, "psi": 1.0, "alpha": [0, 1, 0, 0]},
        {"h": 1, "psi": 1.0, "alpha": [0, 0, 1, 0]},
        {"h": 1, "psi": 1.0, "alpha": [0, 0, 0, 1]},
        {"h": 2, "psi": 1.0, "alpha": [1, 2, 3, -1]},
        {"h": 1, "psi": 0.5, "alpha": [0.5, 0.3, 0.2, 0.1]},
        {"h": 1, "psi": 0.75, "alpha": [0.1, 0.2, 0.3, 0.4]},
    ]

    expected_h2 = [
        3, -5, 0.5, 1.5, 0, -0.5, 0, 1, -0.5, 0,
        0.02631578947368421, -0.32894736842105265,
        0.07894736842105263, 0.2236842105263158,
        0, 0, 0, -0.25, 0, 0.25,
    ]

    print("\n=== Evaluation Results ===\n")

    for i, tc in enumerate(test_cases):
        h, psi, alpha = tc["h"], tc["psi"], tc["alpha"]
        label = f"h={h}, psi={psi}, alpha={alpha}"

        try:
            result = eval_fn(h, psi, alpha)
        except (ZeroDivisionError, ValueError) as e:
            print(f"# Test case {i}: {label}")
            print(f"#   SINGULAR: {e}")
            print()
            continue

        print(f"# Test case {i}: {label}")
        print(f"test_cases[{i}] = {{")
        print(f'    "h": {h},')
        print(f'    "psi": {psi},')
        print(f'    "alpha": {alpha},')
        print(f'    "expected": [')
        for j, v in enumerate(result):
            comma = "," if j < len(result) - 1 else ""
            print(f"        {v!r}{comma}")
        print(f"    ],")
        print(f"}}")
        print()

        # Verify the h=2 case
        if i == 5:
            print("--- Verification against C++ test data ---")
            all_match = True
            for j in range(20):
                match = abs(result[j] - expected_h2[j]) < 1e-12
                status = "OK" if match else "MISMATCH"
                if not match:
                    all_match = False
                    print(f"  c[{j}]: got {result[j]!r}, expected {expected_h2[j]!r} [{status}]")
            if all_match:
                print("  All 20 coefficients match the C++ test data!")
            print()


if __name__ == "__main__":
    main()
