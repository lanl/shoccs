Matrix Formats
===

In the discussion of [](discrete_operators.md) it became apparent that several sparse matrix formats were needed

Sparse Block Matrix
-------------------

The "main" operator is made up of a repeated circulant interior matrix bounded with small dense matrices.
Following the discussion of [](lazyness.md), the matrix needs to have lazy operators:

```c++
// facilitate Ox * U
template<random_access_range R>
constexpr auto operator*(R&& rng) const;

// facilitate (Ox + Bx)
constexpr auto operator+(const CSR& Bx) const;
```

Given the desired interface above, the operators need to be constructed with mapping functions so they can take appropriate stride through the 
fluid/boundary domain.

```{note}
Both `matrix::dense` and `matrix::circulant` provide `begin` and `end` methods so one can iterate over their
coefficients.  This doesn't seem all that useful.
```

### Small Dense Corner Matrix

The coefficient data associated with distinct matrices will (in general) be different.  Those associated with the Cartesian domain boundaries will be the same.  Rather than treating the embedded and domain boundaries differently, let's use the same infrastructure.  This should also make it easier to treat objects intersecting these boundaries.

Currently supported operators:

```c++
template <ranges::random_access_range R>
friend constexpr auto operator*(const dense& mat, R&& rng);
```

Given the use case, it is unlikely to need more.  The coefficients are stored in row major order.


### Circulant Interior Matrix

Since the circulant interior matrix is repeated over the whole mesh with the same coefficients, we would like to only store the coefficients once.  As such, having a separate circulant matrix class with its own storage wouldn't be efficient.  For non-owning, we have two good options:

1.  The circulant matrix holds a `std::span<const real>` member which points to the coefficient data
2.  The apply method could take a `std::span<const real>` of the coefficient data.

Using option 1 allows us to write a lazy, range-based `operator*`.  The coefficients for the circulant matrix start at a given column offset and repeat for a number of rows.  When being applied to a range, the circulant matrix will simply drop the first `offset` elements and then compute inner products along each row.

### 1D Block Matrix

A container for a left boundary (dense), interior (circulant), and right boundary (dense) matrix.  Handles properly moving through the input range and concatenating the ranges resulting from the individual applications.

CSR matrix
-------

`matrix::csr` is to be used for boundary handling the impact of the boundary point on the fluid operators.  At the moment, it is implemented in a typical `(u, v, w)` form and produces a "dense" range upon application.