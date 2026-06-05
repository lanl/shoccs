"""Unit tests for the custom C++ stencil code printer."""

from sympy import Integer, Pow, Rational, Symbol, symbols

from stencil_gen.printer import StencilCodePrinter, build_symbol_map


def test_pow_reciprocal():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(1 + x, -1)) == "1.0 / (x + 1)"


def test_pow_square():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 2)) == "x * x"


def test_pow_square_compound():
    """Compound base gets parenthesized."""
    p = StencilCodePrinter()
    x, y = symbols("x y")
    assert p.doprint(Pow(x + y, 2)) == "(x + y) * (x + y)"


def test_pow_cube():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 3)) == "x * x * x"


def test_pow_high():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, 5)) == "std::pow(x, 5)"


def test_pow_neg2():
    p = StencilCodePrinter()
    x = Symbol("x")
    assert p.doprint(Pow(x, -2)) == "1.0 / (x * x)"


def test_rational():
    p = StencilCodePrinter()
    assert p.doprint(Rational(1, 12)) == "1.0 / 12"
    assert p.doprint(Rational(2, 3)) == "2.0 / 3"
    assert p.doprint(Rational(-5, 6)) == "-5.0 / 6"


def test_integer():
    p = StencilCodePrinter()
    assert p.doprint(Integer(3)) == "3"
    assert p.doprint(Rational(3, 1)) == "3"
    assert p.doprint(Integer(-1)) == "-1"


def test_symbol_map():
    smap = build_symbol_map({"alpha": 2}, has_psi=True)
    p = StencilCodePrinter(symbol_map=smap)
    alpha_0 = Symbol("alpha_0")
    psi = Symbol("psi")
    assert p.doprint(alpha_0) == "alpha[0]"
    assert p.doprint(psi) == "psi"


def test_symbol_unmapped():
    """CSE temporaries not in the map print as their name."""
    p = StencilCodePrinter(symbol_map={})
    t5 = Symbol("t5")
    assert p.doprint(t5) == "t5"


class TestScalarParams:
    """build_symbol_map scalar_params emits subscript-free names."""

    def test_single_scalar(self):
        smap = build_symbol_map({}, scalar_params=["sigma"])
        p = StencilCodePrinter(symbol_map=smap)
        assert p.doprint(Symbol("sigma")) == "sigma"

    def test_multiple_scalars(self):
        smap = build_symbol_map({}, scalar_params=["sigma", "epsilon"])
        p = StencilCodePrinter(symbol_map=smap)
        assert p.doprint(Symbol("sigma")) == "sigma"
        assert p.doprint(Symbol("epsilon")) == "epsilon"

    def test_scalar_alongside_array(self):
        smap = build_symbol_map({"alpha": 2}, scalar_params=["sigma"])
        p = StencilCodePrinter(symbol_map=smap)
        assert p.doprint(Symbol("alpha_0")) == "alpha[0]"
        assert p.doprint(Symbol("alpha_1")) == "alpha[1]"
        assert p.doprint(Symbol("sigma")) == "sigma"

    def test_scalar_not_subscripted(self):
        """Regression: scalar name must NOT be rendered as `sigma[0]`."""
        smap = build_symbol_map({}, scalar_params=["sigma"])
        p = StencilCodePrinter(symbol_map=smap)
        assert "[" not in p.doprint(Symbol("sigma"))

    def test_scalar_default_none(self):
        """scalar_params=None (default) is accepted and yields no mappings."""
        smap = build_symbol_map({"alpha": 1})
        assert Symbol("alpha_0") in smap
        # A symbol whose name was not passed should not be present.
        assert Symbol("sigma") not in smap

    def test_scalar_default_no_arg(self):
        """Omitting scalar_params entirely still works (backward compat)."""
        smap = build_symbol_map({"alpha": 1}, has_psi=True)
        p = StencilCodePrinter(symbol_map=smap)
        assert p.doprint(Symbol("alpha_0")) == "alpha[0]"
        assert p.doprint(Symbol("psi")) == "psi"
