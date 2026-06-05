"""Custom C++ code printer for stencil generation.

Converts SymPy expressions into C++ source fragments matching the style
in the existing hand-written stencil files (E4u_1.cpp, polyE2_1.cpp, etc.).
"""

from sympy import Symbol
from sympy.printing.c import C99CodePrinter


class StencilCodePrinter(C99CodePrinter):
    """SymPy-to-C++ printer with project-specific formatting rules."""

    def __init__(self, symbol_map: dict[Symbol, str] | None = None):
        super().__init__(settings={"precision": 17})
        self._symbol_map = symbol_map or {}

    def _print_Pow(self, expr):
        exp = expr.exp
        base_str = self._print(expr.base)

        # Parenthesize compound bases (Add or Mul)
        if expr.base.is_Add or expr.base.is_Mul:
            base_paren = f"({base_str})"
        else:
            base_paren = base_str

        if exp == -1:
            # Reciprocal: 1.0 / (base)
            return f"1.0 / ({base_str})"
        elif exp == 2:
            # Square: base * base
            return f"{base_paren} * {base_paren}"
        elif exp == 3:
            # Cube: base * base * base
            return f"{base_paren} * {base_paren} * {base_paren}"
        elif isinstance(exp, int) or (hasattr(exp, 'is_integer') and exp.is_integer):
            exp_int = int(exp)
            if exp_int >= 4:
                # High power: std::pow(base, n)
                return f"std::pow({base_str}, {exp_int})"
            elif exp_int <= -2:
                # Negative integer: 1.0 / (base^(-exp))
                from sympy import Pow
                pos_pow = Pow(expr.base, -exp)
                return f"1.0 / ({self._print(pos_pow)})"

        # Fallback to parent
        return super()._print_Pow(expr)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        if q == 1:
            # Integer: plain int literal
            return str(p)
        # Rational: numerator.0 / denominator
        return f"{p}.0 / {q}"

    def _print_Integer(self, expr):
        return str(int(expr))

    def _print_Symbol(self, expr):
        if expr in self._symbol_map:
            return self._symbol_map[expr]
        return expr.name


def build_symbol_map(
    param_arrays: dict[str, int],
    has_psi: bool = False,
    scalar_params: list[str] | None = None,
) -> dict[Symbol, str]:
    """Build a mapping from SymPy Symbols to C++ variable strings.

    Args:
        param_arrays: Maps array name to element count.
            E.g. {"alpha": 2} or {"fa": 6, "da": 3, "ia": 4}.
        has_psi: If True, include psi -> "psi" mapping.
        scalar_params: Runtime scalar parameter names. Each name `n` maps
            `Symbol(n) -> "n"` (no subscript), companion to `param_arrays`
            which emits subscripted array accesses.

    Returns:
        Dict mapping Symbol objects to C++ strings.
    """
    smap: dict[Symbol, str] = {}
    for name, count in param_arrays.items():
        for i in range(count):
            smap[Symbol(f"{name}_{i}")] = f"{name}[{i}]"
    for name in scalar_params or []:
        smap[Symbol(name)] = name
    if has_psi:
        smap[Symbol("psi")] = "psi"
    smap[Symbol("h")] = "h"
    return smap
