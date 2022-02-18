# Stable High-Order Cut-Cell Solver (shoccs)

This cartesian cut-cell solver is the code counterpart for the
numerical algorithm given in the [cut cell
paper](https://doi.org/10.1016/j.jcp.2020.109794).  Currently, the
documentation is woefully incomplete but there are several `.lua`
example files demonstrating how one could run the code.

The code makes heavy use of concepts and
[range-v3](https://github.com/ericniebler/range-v3) to explore the
possiblity of a ranges based dsl for numerical software.

# Building
shoccs depends on [lua](https://www.lua.org),
[sol2](https://github.com/ThePhD/sol2),
[Catch2](https://github.com/catchorg/Catch2),
[range-v3](https://github.com/ericniebler/range-v3),
[fmt](https://github.com/fmtlib/fmt), [pugixml](https://pugixml.org/),
[spdlog](https://github.com/gabime/spdlog), and
[cxxopts](https://github.com/jarro2783/cxxopts).  The libraries can
all be installed from
[shoccs-tpl](https://github.com/pbrady/shoccs-tpl).  Assuming all the
libraries have been installed in `/some/dir/shoccs-tpl` and the
present repository has been cloned into `/other/dir/shoccs`, shoccs can be built using cmake:

```shell
    $ cd /other/dir/shoccs
    $ cmake -Bbuild -H. -DSHOCCS_TPL_DIR=/some/dir/shoccs-tpl
    $ cd build && make
```
If enabled, the tests can be run via `ctest`


## Misc
Copyright assertion C20039
