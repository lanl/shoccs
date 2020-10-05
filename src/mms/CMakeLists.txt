add_library(mms gauss1d.cpp gauss2d.cpp gauss3d.cpp)
target_include_directories(mms PUBLIC ..)

add_unit_test(mms "mms" mms)
