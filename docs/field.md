Fields
===

The core data structures for representing scalar and vector fields are `scalar_field` and `vector_field`, respectively.  We use these data structures rather than raw `std::vector` and `std::span` to encapsulate operations like transposing, applying boundary conditions, and applying operators.  The field structures are also used for lazy operations

Allocation of memory in scalar fields should be informed by the mesh and the geometry information (i.e. number of solid points)

It doesn't seem like a field should own the solid points since this would be difficult to extract during the lazy math operations.  but it could support an operation like pass in a range of solid points (idicies) and a range of values and it sets them.