# Design

In shoccs, we solve sets of equations (systems) consisting of scalar and vector fields defined on a cut-cell domain. We take an object-oriented approach to thinking about the the problem domain.

1. Fields (scalar or vector) are not simply passive containers of data to be indexed. Rather, they have behaviors consistent with mathematical expectations. We can add fields, multiply them, etc. At some level, they still need to be indexable so we can apply our discrete operators, but that remains a low-level detail. Rather than focus on indexing, fields respond to "selections". At a high level, I rarely want the data at a certain index just becuase. Instead, I want all the data corresponding to a certain selection criteria. I may want all the data on the xmin boundary... I may want the portion of the field associate with all the dirichlet boundaries. Fields in shoccs are designed with these 2 behaviors in mind: selection and math.

The design of shoccs is driven by a desire to make it simple to add new sets equations to solve. For a strictly Cartesian mesh with no additional cut-cell geometry this doesn't really pose any difficulties. However, as outlined in {cite}`BradyLivescu2021`, solving a set of governing equations on a cut-cell mesh requires keeping track of an additional set of boundary points for each mesh direction. For this reason, it would be advantageous to stay away from containers such as `std::vector` in the high level api so we can more easily package up the field variables associated with the cartesian domain, $F$, as well as on the cut-cell points, $R$

```{bibliography} references.bib

```
