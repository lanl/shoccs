Discrete Operators
====

In the paper, we discuss that a 1D cut-cell operator is formulated as:

$$
\begin{align*}
    \mathbf{O}^{x,n}_1 = 
    \begin{bmatrix}
    \mathbf{B}_l(\psi_{l,n}, \boldsymbol{\alpha}) & \\
    & \mathbf{C} & \\
    & & \mathbf{B}_r(\psi_{r,n},\boldsymbol{\alpha})
    \end{bmatrix}
\end{align*}
$$

And that such an operator, over the whole domain will look like:

$$
\begin{align}\label{eq:operator_cut}
    \mathbf{O}^x = \begin{bmatrix}
    \mathbf{O}^{x,0}_1 \\
    & \mathbf{O}^{x,1}_1 \\
    & & \mathbf{O}^{x,2}_1 \\
    & & & \ddots \\
    & & & & \mathbf{O}^{x,Q^x}_1 \\
    \end{bmatrix}
\end{align}
$$
where, the size of each 1D operator will be different and the number of
operators, $Q^x$ is a function of geometry.  As written, the operator
is meant to be applied to field data without any holes.  This, however, requires involved
mapping functions for operators applied in different directions and recombined.

We also suggested another approach, which was to split the operators up into those
which operate over $\mathcal{F}$ and those which operate over the boundary.

The part which operates over $\mathcal{F}$ would be invariant, with the exception of
reordering, to the direction of application.  The boundary operator would be a CSR style
sparse matrix which would augment derivatives operator over $\mathcal{F}$.  What does
such a situation look like?

How much of the operator should go into CSR?  Just the boundary points?  
What about the case where 2 boundary points are associated with a single solid
coord?  How do Neumann boundary conditions work?

Ideally, it would be nice to reuse the "holes" in the data layout representing the solid points so we could do something simple like:

$$
\dfrac{\partial}{\partial x} U \approx (O^x + B^x) U
$$
where $B^x$ would be our CSR representation of all the boundary matrices.  There would have to be a 'pre' step where appropriate data is copied to the right points in $U$ and a 'post' phase
where the data is copied back out.

Potential Matrix Formats
---

Some options and their implications using the assumptions

1.  There is a persistent set of points (i.e. $R^x$) associated with all boundary data.  This
means that given a field array of data we can freely overwrite the data associated with 
embedded `solid_coord`'s or the interior solid points.

2. There is a mapping $T : R^x \to S^x$ which allows for mapping data associated with 
the embedded boundary, $R^x$, to the solid points in the domain, $S^x$.  The inverse mapping, $T^{-1}$, also exists.  There are used to fill in field data with boundary values and populate the sparse boundary matrices as well as copy updated data back out
of the solid points


#### Move all of $\mathbf{B}$ to $B^x$

In this case, the discrete operator, $\mathbf{O}$ is made up of the circulant matrices, $C$,
with lots of empty rows in the matrix.  The layout of the data to which $\mathbf{O}$ is applied
is unambiguous since it is applied only on the fluid domain, $\mathcal{F}$, and only on a subset of that.

In the event that a `solid_coord` on the boundary is uncontested we could copy the boundary data to its corresponding `solid_coord` in $U$.  The boundary matrix $\mathbf{B}$ is then 
copied into $B^x$.  The copy will have the same coordinates and storage structure but be represented differently.

In the event than a `solid_coord` is contested by two points, we would give preference to 1 and go with the above case for handling it.  The "loser", however, would need to assign it's data value to an interior solid point in $U$ and construct it's CSR representation into $B^x$ accordingly.  It will be a similar representation to the previous case except that one of the points will be far away from the rest.  This will impact the row of the data point for all but the first row of $\mathbf{B}$ but will also impact the column of the data for the first row of $\mathbf{B}$.

Thus some amount of sorting on the data is required (for performance) before constructing $B^x$.

If the only change from contested points is the location of the boundary point and therefore, the treatment of the first row/column of $\mathbf{B} than maybe the that is the only portion that should be placed into $B^x$


#### Move first row/column of $\mathbf{B}$ to $B^x$
We start
by splitting the boundary stencil matrices $\mathbf{B}$ into the boundary row,
$B^{r_0}$, the boundary column, $B^{c_0}$, and the rest, $B^I$:

$$
\begin{align}
\mathbf{B}_l = \begin{bmatrix}
& B^{r_0} & & \\
\hline
& | \\
B^{c_0} & |  & B^I & \\
\end{bmatrix}
\end{align}
$$
Note that $B^{c_0}$ does not include the corner point, since it makes more sense to keep all of $B^{r_0} together at the moment.
Under this approach, the discrete operator, $O$, is made up of the circulant matrices, $C$, as well as the $B^I$ portions of the boundary matrices.  Note that while $C$ is independent of $\psi$ and $\boldsymbol{\alpha}$, $B^I$ is a function of them as well as the type of boundary condition (i.e. Neumann, Dirichlet, Floating).  By convention, for Dirichlet conditions, $B^{r_0} = 0$.  

Given an efficient storage strategy, it doesn't seem unreasonable to store different
versions of $O$ which contain the various $B^I$'s related to the different boundary conditions required by the simulation.  However, when we develop the method for moving objects, the extents of $C$ will be functions of time as well the coefficients of $B^I$.

With this, the coefficients in $B^x$ will be $B^{c_0}$ and $B^{r_0}$.  Each boundary stencil is associated with a particular boundary point.  That point may be the `solid_coord` or it may be an interior point. 

```{note}
Q. Is storing the `solid_coord` simply an optimization of the CSR structure? Does it really provide any benefit if only $B^{c_0}$ and $B^{r_0}$ are stored in $B^x$?

A. It will serve as a starting point for the cross-derivative rays but doesn't need
to be used in constructing $B^x$.
```

Let the point associated with the boundary stencil be denoted by $p$.  This point is somewhere in the solid body.  The boundary data for for this point has been copied to point $p$ in $U$.  If the rows of $B^I$ are given by the set of indices $\{q\}$, Copying $B^{c_0}$ into $B^x$ means mapping the elements of $B^{c_0}$ to $(q,p)$ in $B^x$.  Likewise, if the columns of $B^{r_0}$ are associated with the set of indices $\{s\}$, Copying $B^{r_0}$ into $B^x$ means mapping the first element of $B^{r_0}$ to $(p,p)$ and the rest to $(p,s)$.

After computing any updates, the boundary elements of $U$ would need to by copied back to their respective members in $R^x$.

Handling Neumann boundaries involves an extra step of imposing the bc on $u'$.  Imposing this bc takes place on all rows of $\mathbf{B}$, but is essentially just adding a sparse array, $N^x$, to computed derivative.  We should just treat it that way.

I like this approach but it won't work for compact.  Or rather it will, but the resulting system will not necessarily be a narrow banded linear system so the solve will be more expensive.  Realistically, solving the equations on the boundary would also result in a non narrow-banded system.

What is needed to implement this?

* The operator won't necessarily know what the full operator with BC's should be
because the BC's will be determined by the system being solved.  Maybe the system
should be constructed with an "operator factory" that produces operators with the
appropriate BC matrices.

* The mappings $T$, $T^{-1}$ are only functions of geometry.  These will be needed to form $B^x$

* The functions producing $\mathbf{B}_l(\psi,\boldsymbol\alpha)$ should produce $B^{c_0}$, $B^{r_0}$, and $B^I$

* Sparse matrix formats for $O^x$, $B^x$, and $N^x$.  $O^x$ is a sparse block matrix comprised of $C$, and $\mathbf{B}_{l/r}$ matrices

1D Operator
----
What does a 1D operator need to do/know?  Should $B^x$ actually be exposed to the user at all or should $O^x$ and $B^x$ be members of a single operator.  They are not really independent things that can be mixed and matched so it makes sense to wrap them up.  So, we wrap them up and then have our operator, $D$, which will be applied to a random access range of the data.  If we pass in boundary condition information  to the operator rather than pre-apply it to the range we can more easily wrap up a series of 1D operators into a 3D operator.

However, requiring boundary condition information means abandoning `operator*` for the discrete operators.  If we keep it, then we need to make the user call a method that sets the boundary condition information before the call `operator*`.

How should we pass in boundary condition values?  The type of boundary conditions has already been incorporated by the construction of the operator via $O^x$ and $B^x$.  We could pass in functions which will then operate on $R^x$ mapped to $S^x$.  But how would we do something line the Carpenter test with $v_0 = u_n$ and $u_0 = v_n$.  That kind of boundary condition would require a lambda capture on the boundary points or a weird calling convention.  It seems better to pre-compute the object boundary conditions and pass them in as a span (conforming to $R^x$ which is mapped to $S^x$ in the operator).  What about domain bc's? dirichlet values could be written on the range
before passing it in.  Floating doesn't have an impact.  Neumann requires some thought (also for the object bc's).  In general, the Neumann boundary is not a function of the field so it can simply be added to the computed field after the fact.  We might require Neumann conditions on domain bcs and the object.  Neumann conditions could simply be represented as a CSR matrix of coefficients which are applied to boundary values rather than the field data.

For the gradient operator, the above boils down to passing in Dirichlet boundary data on $R^x$, $R^y$, and $R^z$.  Data should also be passed in for floating since information on the boundary will be overwritten. Same is true for the fluid data with neumann boundaries.

In addition, for Neumann, we need to pass in data for $f'$ possibly at the domain boundaries as well as on $R^x$, $R^y$ and $R^z$.  This suggests another full domain range that is used just for Neumann bcs and is overwritten with boundary data.  
This does pose a problem for domain corners where it is not strictly necessary to enforce $\partial f /\partial x = \partial f / \partial y$.  We could simply list that
as a solver limitation to simplify the implementation

Note that for Neumann BC's we need both the derivative information in the Neumann CSR matrix, $N^x$, and also coefficient data depending on the value information in $B^x$.  Furthermore, we are using a simple index based mapping scheme for boundary and derivative values $R^x \to S^x$.  If we return ranges from the gradient application operator we need to return a range associate with values and another with Neumann corrections.  We can't do much more because the ranges that result from applying the operator are not random access.  It would then be the callers' responsibility to extract the boundary
information on $R^x$ for both sets of boundary conditions and combine them in a meaningful way.

On the other hand we could adopt an interface that returns two spans, a full domain span
and a boundary span.  Ideally we would like to not split up the data in terms of boundary conditions but simply return the computed derivative on the boundary points.  The full domain span would have the data from the solid points zeroed out.  

Rather than returning spans, we could have the user specify two random access ranges to use as output ranges.  These probably can't be reused from the input field ranges or we will run into data access update conflicts.  What should this look like when calling the gradient operator?

```c++
// this?
grad.with_boundary_values(dirichlet_Rx, Neumann_Rx).with_outputs(field_rng, deriv_rng).apply(a + b * c)

// this?
grad(a + b * c, dirichlet_Rx, Neumann_Rx, field_rng, deriv_rng)?
```
Should I abandon spans and just use a custom type?  

Operator Generator
----

The discrete operator can be partially constructed using the information about the stencil and geometry information.  However, it can't be finished until it knows about the boundary conditions for the problem.  Given the different sources of knowledge, it makes sense to have an "Operator Generator" or "Operator Builder" class with an interface like:

```c++
class discrete_operator;

struct object_bc_map {
    int shape_id;
    boundary b;
};

struct domain_bc_map {
    boundary xmin, xmax;
    boundary ymin, ymax;
    boundary zmin, zmax;

};

class operator_builder {
    public:
    operator_builder(const geometry& geom, const stencil& st);

    discrete_operator build(domain_bc_map, std::vector<bc_map>);
};
```

What to do about `gradient`, `divergence`, `laplacian` operators? should they be separate classes?  The `operator*` method needs to templated on a range so we can't make use of any virtuals.  Should the `build` method be split into `build_gradient`, `build_divergence`, `build_laplacian`?  The operator will already have knowledge of the solid geometry and bc's so it should be able to wrap the necessary computations with input about the bc values (rather than just the bc type).  Let's make them separate to simplify the implementation of the system solvers

Gradient Operator
-----

Assuming an `operator_builder` has been properly constructed with the geometry and scheme information, we construct a `gradient_opererator` using the `build` interface described above.  The idea is then evaluate a gradient simply as

```c++
gradient_operator grad = builder.build_gradient(domain_bc_map, std::vector<bc_map>);

auto rng = /* some computations on field variables that yields a random_access_range*/ ;

           // grad * rng
auto rhs = grad(rng); // we could convert this to a vector or leave as range
```