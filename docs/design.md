Design
===

The design of shoccs is driven by a desire to make it simple to add new sets equations to solve.  For a strictly Cartesian mesh with no additional cut-cell geometry this doesn't really pose any difficulties.  However, as outlined in {cite}`BradyLivescu2021`, solving a set of governing equations on a cut-cell mesh requires keeping track of an additional set of boundary points for each mesh direction.  For this reason, it would be advantageous to stay away from containers such as `std::vector` in the high level api so we can more easily package up the field variables associated with the cartesian domain, $F$, as well as on the cut-cell points, $R$


```{bibliography} references.bib
```
