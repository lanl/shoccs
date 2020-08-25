add_library(geometry
    sphere.cpp
    geometry.cpp)

target_include_directories(geometry PUBLIC ..)

add_unit_test(shapes "shapes" geometry)
add_unit_test(geometry "geometry" geometry)
