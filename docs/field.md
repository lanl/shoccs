Fields
===

The core data structures for representing scalar and vector fields are `scalar_field` and `vector_field`, respectively.  We use these data structures rather than raw `std::vector` and `std::span` to encapsulate operations like transposing, applying boundary conditions, and applying operators.  The field structures are also used for lazy operations