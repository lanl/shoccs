#!/usr/bin/env python3
"""Parse a ``polyE2_1.cpp`` file into Python-callable evaluators.

Extracts the four runtime methods —

  * ``interp_interior(y, c)``        -> sign(y) branch, 2 coeffs
  * ``interp_wall(i, y, psi, c, right)`` -> if(right)/else switch(i) case 0/1
  * ``nbs_floating(h, psi, c, right)``   -> CSE temporaries + 12 coeffs + /h
  * ``nbs_dirichlet(h, psi, c, right)``  -> CSE temporaries + 8 coeffs + /h

— and returns plain Python callables so a regenerated file can be numerically
compared against the committed hand-port to ``1e-12`` on a sampled parameter
grid.  This is the *arbiter* for the capstone equivalence test.

The C++ bodies are deliberately CSE'd with locally-scoped ``tN`` temporaries;
we parse the assignment lines in source order so the temporaries resolve
naturally when ``exec``'d.  No symbolic algebra is involved — purely literal
translation of the emitted arithmetic.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _convert_expr(expr: str) -> str:
    """Translate a C++ rvalue into a Python rvalue.

    Handles array subscripts (fa[0] -> fa[0], already valid), the ``1.0 /``
    reciprocal idiom, and ``std::pow`` (defensive; E2 has none)."""
    for _ in range(100):
        if "std::pow(" not in expr:
            break
        prev = expr
        expr = re.sub(
            r"std::pow\(([^,()]+(?:\([^()]*\))?[^,()]*),\s*([^()]+)\)",
            r"(\1)**(\2)",
            expr,
        )
        if expr == prev:
            raise ValueError(f"failed to convert std::pow: {expr!r}")
    return expr


def _join_statements(body: str) -> list[str]:
    """Collapse a brace-delimited body into a list of `;`-terminated statements,
    preserving order, ignoring braces / control keywords / comments."""
    # strip comments
    body = re.sub(r"//.*", "", body)
    stmts: list[str] = []
    current = ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        current += (" " if current else "") + line
        while ";" in current:
            idx = current.index(";")
            stmts.append(current[:idx].strip())
            current = current[idx + 1 :].strip()
    return stmts


def _extract_method_body(src: str, signature_marker: str) -> str:
    """Return the brace-balanced body (between the first { and matching }) of the
    method whose signature contains *signature_marker*."""
    pos = src.find(signature_marker)
    if pos < 0:
        raise ValueError(f"method marker not found: {signature_marker!r}")
    brace = src.index("{", pos)
    depth = 0
    for k in range(brace, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[brace + 1 : k]
    raise ValueError("unbalanced braces")


# ---------------------------------------------------------------------------
# block parsing: turn an ordered list of `tN = ...` / `c[N] = ...` statements
# into executable Python lines.
# ---------------------------------------------------------------------------
def _stmts_to_pylines(stmts: list[str]) -> list[str]:
    out: list[str] = []
    for stmt in stmts:
        m = re.match(r"(?:const\s+)?real\s+(t\d+)\s*=\s*(.+)", stmt)
        if m:
            out.append(f"{m.group(1)} = {_convert_expr(m.group(2).strip())}")
            continue
        m = re.match(r"(c\[\d+\])\s*=\s*(.+)", stmt)
        if m:
            out.append(f"{m.group(1)} = {_convert_expr(m.group(2).strip())}")
            continue
        # ignore subspan/return/for/if/switch/case/break statements
    return out


# ---------------------------------------------------------------------------
# interp_interior
# ---------------------------------------------------------------------------
def make_interp_interior(cpp_path: str):
    src = _read(cpp_path)
    body = _extract_method_body(
        src, "interp_interior(real y, std::span<real> c)"
    )
    # split into the two sign branches
    m = re.search(r"if\s*\(\s*y\s*>\s*0\s*\)\s*\{(.*?)\}\s*else\s*\{(.*?)\}", body, re.S)
    pos_lines = _stmts_to_pylines(_join_statements(m.group(1)))
    neg_lines = _stmts_to_pylines(_join_statements(m.group(2)))

    def fn(y):
        c = [0.0, 0.0]
        env = {"c": c, "y": y}
        for ln in (pos_lines if y > 0 else neg_lines):
            exec(ln, {}, env)
        return c[:2]

    return fn


# ---------------------------------------------------------------------------
# interp_wall
# ---------------------------------------------------------------------------
def make_interp_wall(cpp_path: str):
    src = _read(cpp_path)
    body = _extract_method_body(
        src,
        "interp_wall(int i, real y, real psi, std::span<real> c, bool right)",
    )
    m = re.search(r"if\s*\(\s*right\s*\)\s*\{(.*)\}\s*else\s*\{(.*)\}\s*$", body, re.S)
    if not m:
        # find the outer if(right){...}else{...} robustly via brace matching
        idx = body.index("if (right)")
        brace = body.index("{", idx)
        depth = 0
        for k in range(brace, len(body)):
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    right_body = body[brace + 1 : k]
                    rest = body[k + 1 :]
                    break
        eidx = rest.index("else")
        ebrace = rest.index("{", eidx)
        depth = 0
        for k in range(ebrace, len(rest)):
            if rest[k] == "{":
                depth += 1
            elif rest[k] == "}":
                depth -= 1
                if depth == 0:
                    left_body = rest[ebrace + 1 : k]
                    break
    else:
        right_body, left_body = m.group(1), m.group(2)

    def parse_branch(branch: str):
        # temporaries before switch, then per-case c[] assignments
        switch_idx = branch.index("switch")
        pre = branch[:switch_idx]
        pre_lines = _stmts_to_pylines(_join_statements(pre))
        # split cases
        cases: dict[int, list[str]] = {}
        for cm in re.finditer(r"case\s+(\d+)\s*:(.*?)break\s*;", branch, re.S):
            ci = int(cm.group(1))
            cases[ci] = _stmts_to_pylines(_join_statements(cm.group(2)))
        return pre_lines, cases

    right_pre, right_cases = parse_branch(right_body)
    left_pre, left_cases = parse_branch(left_body)

    def fn(i, y, psi, fa, da, ia, right):
        c = [0.0, 0.0, 0.0, 0.0]
        env = {
            "c": c, "y": y, "psi": psi,
            "fa": fa, "da": da, "ia": ia,
        }
        pre, cases = (right_pre, right_cases) if right else (left_pre, left_cases)
        for ln in pre:
            exec(ln, {}, env)
        for ln in cases[i]:
            exec(ln, {}, env)
        return c[:4]

    return fn


# ---------------------------------------------------------------------------
# nbs_floating / nbs_dirichlet
# ---------------------------------------------------------------------------
def _make_nbs(cpp_path: str, marker: str, n_coeffs: int):
    src = _read(cpp_path)
    body = _extract_method_body(src, marker)
    pylines = _stmts_to_pylines(_join_statements(body))

    def fn(h, psi, fa, da, ia, right=False):
        c = [0.0] * n_coeffs
        env = {"c": c, "h": h, "psi": psi, "fa": fa, "da": da, "ia": ia}
        for ln in pylines:
            exec(ln, {}, env)
        c = [v / h for v in c]
        if right:
            c = [-v for v in reversed(c)]
        return c

    return fn


def make_nbs_floating(cpp_path: str):
    return _make_nbs(
        cpp_path,
        "nbs_floating(real h, real psi, std::span<real> c, bool right)",
        12,
    )


def make_nbs_dirichlet(cpp_path: str):
    return _make_nbs(
        cpp_path,
        "nbs_dirichlet(real h, real psi, std::span<real> c, bool right)",
        8,
    )


def make_all(cpp_path: str) -> dict:
    """Return all four evaluators keyed by method name."""
    return {
        "interp_interior": make_interp_interior(cpp_path),
        "interp_wall": make_interp_wall(cpp_path),
        "nbs_floating": make_nbs_floating(cpp_path),
        "nbs_dirichlet": make_nbs_dirichlet(cpp_path),
    }


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    cpp = os.path.join(repo, "src", "stencils", "polyE2_1.cpp")
    ev = make_all(cpp)
    fa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    da = [3 / 25, 13 / 100, 7 / 50]
    ia = [0.7, 0.8, 0.9, 1.0]
    print("interp_interior(0.3):", ev["interp_interior"](0.3))
    print("interp_interior(-0.3):", ev["interp_interior"](-0.3))
    print("interp_wall left0:", ev["interp_wall"](0, 0.3, 0.8, fa, da, ia, False))
    print("nbs_floating:", ev["nbs_floating"](1.0, 0.2, fa, da, ia))
    print("nbs_dirichlet:", ev["nbs_dirichlet"](1.0, 0.001, fa, da, ia))
