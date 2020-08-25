Finite Difference Stencils
==========

The sizes of the different stencils are currently hardcoded as NTTP's.  Ideally, it would be nice 
to allow the user to specify a stencil in lua and just use that.  The NTTP approach allows
for compact objects but since polymorphic use is required, they must all be wrapped up in TE
class.  The TE class could also serve as a wrapper.

The benefit of using NTTP's was that is was possible to return std::array's but maybe that wasn't such a good idea
anyway. A general approach could easily just pass in a std::span after a size_query.

Following the discussion in [](discrete_operators.md), it seems best to break the stencils up into the first row, $B^{r_0}$,
the first column, $B^{c_0}$ (not including the first point in $B^{r_0}$) and the fluid portion of the boundary matrix, $B^I$.
This seems like an unreasonable constraint to place on the stencil implementations.  Given a size in row/columns, all of this could be extracted from a simply interface which returned the full boundary matrix.  Additionally, Neumann boundaries have extra coefficients.

The stencil functions need to take parameters, $\psi$, $\Delta x$, flag indicating a left or right wall (or `ray_outside` in the geometry notation).  We could split the functions up into `dirichlet`, `neumann`, and `floating` or we could just represent the bcs with an enum and pass that in so there is just a single interface:

```c++
struct stencil_info {
    int rows;
    int columns;
    int nextra; // for neumann
};

enum class boundary_t {dirichlet, neumann, floating};

stencil_info query(boundary_t) const;

std::span<real> coefficients(dx, psi, boundary_t, bool, std::span<real>, std::span<real> extra)

```

Adopting a type erased strategy allows for value semantics