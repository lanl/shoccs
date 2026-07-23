"""Cached lambdified cut-cell closure evaluator (plan 48 item 1.1).

Every downstream normal-mode item (psi-parameterized Kreiss checks,
reflection coefficients, interval operators) consumes cut-cell closure rows
at (psi, alpha) points.  Deriving the symbolic stencil per call is fine for
E2_1 (~0.03 s) but measured > 170 s for E4_1 — unusable in any sweep or
optimizer loop (see ``group_velocity.psi_sweep_group_velocity`` for the
derive-per-call antipattern this module replaces).  Here the symbolic
derivation runs once per (scheme, bc) — cached via ``functools.lru_cache``
— and every entry of the chosen block is ``sympy.lambdify``-compiled over
``(psi, *alpha_symbols)`` once, so subsequent evaluations cost microseconds.

The **Dirichlet (wall-eliminated) block is the canonical analysis object**:
it is what ``derivative.cpp`` time-steps (the wall column is stripped into
the CSR B matrix and the remaining (R-1) x (T-1) dense block feeds the
inner_block).  Homogeneous Dirichlet is the correct GKS setting — the
determinant condition tests the homogeneous resolvent problem — so
``CutCellEvaluator.rows`` returns the closure rows with the wall column
dropped.  The floating block is never analyzed for stability.

Row/column geometry (matches the repo-wide assembly convention, e.g. the
test-local builder in ``tests/test_group_velocity.py`` and
``derivative.cpp``'s placement of ``nbs_dirichlet`` rows): the Dirichlet
block has R-1 rows and T columns; column 0 is the wall (at fractional
position -psi relative to grid node 0), columns 1..T-1 are grid nodes
0..T-2.  After dropping the wall column, Dirichlet row i sits at grid node
i, so ``row_indices = [0, ..., R-2]``.

Sign convention: the returned weights are the *raw derivative-operator*
rows (the same convention ``brady2d_stability._extract_stencil_data``
feeds to ``kreiss_stability_check``, whose module analyzes
``u_t = -sum_k a_k u_{n+k}``); no signs are flipped here.
"""

from __future__ import annotations

import functools

import numpy as np
from sympy import Expr, Symbol, cancel, factor_list, lambdify

from stencil_gen.gks_kreiss import BoundaryRow
from stencil_gen.interior import derive_interior, full_gamma_array
from stencil_gen.temo import (
    CutCellResult,
    Dimensions,
    SchemeParams,
    derive_cut_cell_paper,
    derive_cut_cell_scheme,
)

__all__ = ["CutCellEvaluator", "make_cutcell_evaluator"]


def _collect_denominators(block) -> frozenset:
    """Collect irreducible non-constant denominator factors of *block*.

    Each entry is put in canonical rational form with ``cancel`` (so
    removable factors do not produce false infeasibility flags), its
    denominator extracted, and the denominator split into irreducible
    factors via ``factor_list``.  Constant factors are discarded; the
    remaining polynomials in (psi, alpha) are the parameterization
    singularities downstream feasibility guards must test (plan 48 item
    1.2: denominator vanishing is a *parameterization* failure, never a
    stability verdict).
    """
    dens: set[Expr] = set()
    for entry in block:
        den = cancel(entry).as_numer_denom()[1]
        if not den.free_symbols:
            continue
        for fac, _mult in factor_list(den)[1]:
            if fac.free_symbols:
                dens.add(fac)
    return frozenset(dens)


class CutCellEvaluator:
    """Fast numeric evaluator for a symbolically derived cut-cell closure.

    Built once per (scheme, bc) by :func:`make_cutcell_evaluator`; do not
    construct directly.  All symbolic work (derivation, lambdification,
    denominator collection) happens in ``__init__``; evaluation via
    :meth:`rows` is pure floating point.

    Attributes
    ----------
    scheme : SchemeParams
        The scheme this evaluator was built for.
    bc : str
        Boundary-condition variant ("dirichlet" — the canonical analysis
        object; see module docstring).
    dims : Dimensions
        Stencil dimensions (r, t, R, T, X).
    alpha_symbols : list[Symbol]
        Free alpha symbols of the derived closure, in canonical order.
    denominators : frozenset[Expr]
        Irreducible non-constant denominator polynomials in (psi, alpha)
        collected from the block entries, for feasibility pre-checks.
    """

    def __init__(
        self,
        scheme: SchemeParams,
        bc: str,
        cc_result: CutCellResult,
        psi_symbol: Symbol,
    ) -> None:
        self.scheme = scheme
        self.bc = bc
        self.dims: Dimensions = cc_result.dims
        self.alpha_symbols: list[Symbol] = list(cc_result.alpha_symbols)
        self._psi_symbol = psi_symbol
        # The canonical analysis block: (R-1) x T Dirichlet rows
        # (wall column still present here; rows() drops it).
        self._block = cc_result.dirichlet

        syms = (psi_symbol, *self.alpha_symbols)
        # Lambdify every entry of the block once (plan 48 item 1.1).
        self._entry_funcs = [
            [lambdify(syms, self._block[i, j]) for j in range(self._block.cols)]
            for i in range(self._block.rows)
        ]
        self.denominators = _collect_denominators(self._block)

        # Matching interior stencil (explicit schemes only: s == 0).
        if scheme.s == 0:
            coeffs = derive_interior(scheme.s, scheme.p, scheme.nu)
            self._interior_weights = np.array(
                [float(g) for g in full_gamma_array(coeffs)], dtype=float
            )
            self._interior_offsets = np.arange(-scheme.p, scheme.p + 1)
        else:
            self._interior_weights = None
            self._interior_offsets = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _alpha_args(self, alpha: dict) -> list[float]:
        """Order the alpha dict into the lambdify argument list.

        Keys may be the alpha Symbols themselves or their string names.
        Every alpha symbol must be supplied; unknown keys are rejected
        (a silent default would mask typos).
        """
        known = {s.name for s in self.alpha_symbols}
        by_name: dict[str, float] = {}
        for k, v in alpha.items():
            name = k.name if isinstance(k, Symbol) else str(k)
            if name not in known:
                raise ValueError(
                    f"Unknown alpha key {name!r}; this scheme's free "
                    f"parameters are {sorted(known)}"
                )
            by_name[name] = float(v)
        missing = [s.name for s in self.alpha_symbols if s.name not in by_name]
        if missing:
            raise KeyError(
                f"Missing alpha values for {missing}; all of "
                f"{[s.name for s in self.alpha_symbols]} are required"
            )
        return [by_name[s.name] for s in self.alpha_symbols]

    def rows(self, psi: float, alpha: dict) -> tuple[list[BoundaryRow], np.ndarray]:
        """Evaluate the Dirichlet closure rows at (psi, alpha).

        Returns the (R-1) closure rows with the **wall column dropped**
        (homogeneous Dirichlet is the correct GKS setting; the determinant
        condition tests the homogeneous resolvent problem), integer column
        offsets renumbered to grid nodes 0..T-2, and
        ``row_indices = [0, ..., R-2]`` (Dirichlet row i sits at grid
        node i).

        Parameters
        ----------
        psi : float
            Fractional wall position in (0, 1].
        alpha : dict
            Mapping from alpha symbol (or name) to numeric value; must
            cover all of :attr:`alpha_symbols`.

        Returns
        -------
        (boundary_rows, row_indices)
            ``boundary_rows`` is a list of ``(weights, col_offsets)``
            pairs (the ``gks_kreiss.BoundaryRow`` layout); ``row_indices``
            is ``np.arange(R-1)``.

        Notes
        -----
        Evaluating exactly on a denominator zero raises
        ``ZeroDivisionError``.  Guarding against near-singular
        parameterizations is the caller's job (plan 48 item 1.2's
        feasibility guard uses :attr:`denominators`).
        """
        args = (float(psi), *self._alpha_args(alpha))
        n_rows = self._block.rows
        n_cols = self._block.cols  # T (wall + grid nodes 0..T-2)
        col_offsets = np.arange(n_cols - 1)
        out: list[BoundaryRow] = []
        for i in range(n_rows):
            funcs = self._entry_funcs[i]
            # Skip column 0 (the wall column).
            weights = np.array([funcs[j](*args) for j in range(1, n_cols)],
                               dtype=float)
            out.append((weights, col_offsets.copy()))
        return out, np.arange(n_rows)

    def interior(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the matching interior stencil ``(weights, offsets)``.

        Derived from ``interior.py`` (exact rational coefficients,
        converted to float): ``weights[k]`` multiplies ``u_{n+offsets[k]}``
        with ``offsets = [-p, ..., p]``.
        """
        if self._interior_weights is None:
            raise NotImplementedError(
                f"interior() supports explicit schemes only (s == 0); "
                f"got s = {self.scheme.s}"
            )
        return self._interior_weights.copy(), self._interior_offsets.copy()


@functools.lru_cache(maxsize=None)
def make_cutcell_evaluator(
    scheme: SchemeParams, bc: str = "dirichlet"
) -> CutCellEvaluator:
    """Build (or fetch from cache) the cut-cell evaluator for *scheme*.

    Dispatches by ``scheme.zeros`` exactly as production does
    (``derive_cut_cell_paper`` for schemes with structural uniform zeros
    like E4_1, ``derive_cut_cell_scheme`` otherwise — the same dispatch as
    ``group_velocity.psi_sweep_group_velocity``), then lambdifies every
    entry of the Dirichlet block once.  ``SchemeParams`` is a frozen
    dataclass, hence hashable, so the cache is keyed on the scheme
    directly.

    Parameters
    ----------
    scheme : SchemeParams
        Scheme parameters (must be hashable; frozen dataclass).
    bc : str
        Only ``"dirichlet"`` is supported: the wall-eliminated Dirichlet
        block is the canonical analysis object (plan 48 constraint — the
        floating block is never analyzed for stability).

    Returns
    -------
    CutCellEvaluator
        The cached evaluator; repeated calls with equal arguments return
        the identical object.
    """
    if bc != "dirichlet":
        raise ValueError(
            f"Unsupported bc {bc!r}: the Dirichlet (wall-eliminated) block "
            "is the canonical analysis object (plan 48); the floating "
            "block is never analyzed for stability"
        )
    psi = Symbol("psi")
    if scheme.zeros:
        cc = derive_cut_cell_paper(scheme, psi)
    else:
        cc = derive_cut_cell_scheme(scheme, psi)
    return CutCellEvaluator(scheme, bc, cc, psi)
