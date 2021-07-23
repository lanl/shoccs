#include <fmt/core.h>
#include <fmt/ranges.h>
#include <string>

#include <array>

#include <range/v3/all.hpp>

using int3 = std::array<int, 3>;
using real3 = std::array<double, 3>;

namespace vs = ranges::views;

int main()
{

    int3 extents{2, 4, 8};
    real3 origin{1.0, 1.1, 0.8};
    real3 bounds{1.1, 1.2, 1.3};
    int grid_num = 0;
    double time = 0.0;

    std::string msg =
        fmt::format(R"(<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
<Domain>
<Topology TopologyType="3DCoRectMesh" Dimensions="{}"/>
<Geometry GeometryType="Origin_DxDyDz">
    <DataItem Format="XML" NumberType="Float" Dimensions="3">
        {}
    </DataItem>
    <DataItem Format="XML" NumberType="Float" Dimensions="3">
        {}
    </DataItem>
</Geometry>
<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    <Grid Name="{}" GridType="Uniform">
        <Topology Reference="/Xdmf/Domain/Topology[1]"/>
        <Geometry Reference="/Xdmf/Domain/Geometry[1]">
        <Time Value="{}"/>
        <Attribute Name="U">
            <DataItem Dimensions="{}" NumberType="Float" Precision="8" Format="XML">
                {}
            </DataItem>
        </Attribute>
    </Grid>
</Grid>
</Domain>
</Xdmf>
)",
                    fmt::join(extents, " "),
                    fmt::join(origin, " "),
                    fmt::join(bounds, " "),
                    grid_num,
                    time,
                    fmt::join(extents, " "),
                    fmt::join(vs::iota(0, extents[0] * extents[1] * extents[2]), " "));
    fmt::print(msg);

    int suffix_length = 6;
    grid_num++;

    // msg = fmt::format("U.{:06d}", grid_num);
    fmt::print("U.{:0{}d}\n", grid_num, suffix_length);

    fmt::print("y,{}", fmt::join(vs::repeat_n("Wall,psi", 3), ","));
}
