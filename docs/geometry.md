Geometry
====

The intersection of the embedded objects with the Cartesian mesh lines is a simple ray-tracing implementation.  When an intersection is found, several data points must be recorded which may be split into 2 categories:

1. Geometry information
    * (x, y, z) location of the intersection which will be needed to enforce boundary conditions that depend on position
    *  A label for the embedded object since different objects may have different BC's associated with him.
    * In the future we would also a surface normal, center-of-mass, orientation angles.
2. Mesh information
    * (i, j, k) coordinate of "solid" point will be needed to form the sparse matrix containing the boundary point operators


Rays
----
For the FD method used here we "shoot" rays from a starting point into Cartesian mesh directions.  The object/mesh intersection points all originate from mesh points and therefore occur along mesh lines.  The evaluation of the governing equations at boundary points requires we shoot rays from the intersection points.  Therefore we require rays to have arbitrary starting points but the direction of travel is along mesh lines.

In general, if a ray has an origin, $O$, and a direction, $D$, then the position of a point along the ray is given by $O + t D$, where $t$ is a scalar.

Thus, for objects with an analytic description, intersections are simply a matter of checking for valid $t$.

Objects
----
The embedded objects have an interface requiring one function
```c++
struct hit_info {
    real t;           // t for which the incident ray struck the object
    real3 position;   // ray.position(t)
    bool ray_outside; // true if ray came from outside object
    int shape_id;
};

std::optional<hit_info> hit(ray, t_min, t_max) const;
```

Adding a new object requires adding a factory function to `geometry/shapes.hpp` and defining the object and factory function implementation in a `.cpp` file.  For example, the factory function for a sphere is

```c++
shape make_sphere(int id, const real3& origin, real radius);
``` 

```{admonition} Constraints on *shape_id*
:class: warning

At the moment, the user is responsible for esuring that shapes are assigned contiguous unique id's increasing from 0. 
```


Geometry
-------
The primary way the driver will interact with the embedded geometry is through the `geometry` object.  This object is initialized by a `span` of shapes and some mesh information.  The query operations are analagous to those described in the paper:

```c++
// record psi and solid_coord for use in operator construction
struct mesh_object_info {
    real psi;
    real3 position;
    bool ray_outside;
    int3 solid_coord;
    int shape_id;
};

// information about the the ray/object intersections 
// for rays in x and shape `shape_id`
std::span<const mesh_object_info> Rx(int shape_id) const;

// information about all the ray/object intersections
// for rays in x and all shapes
std::span<const mesh_object_info> Rx() const;

// All solid points (boundary and interior) for all shapes
std::span<int> Sx() const;
```

During construction, the geometry object computes all the mesh/ray intersections and returns const views of them when requested.

The `solid_coord` member indicates the coordinate of the "hole" left in the mesh due to this point.  This coordinate is *not* unique but will be the anchor for many of the
interpolation operations needed in the future. 



```{admonition} Constraints on geometry
:class: warning

We do not correctly handle situations where whole lines are solid.  This situation
will be encountered frequently when the algorithm is made parallel.
```