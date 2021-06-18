#include <iomanip>
#include <iostream>
#include <pugixml.hpp>
#include <string>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "xdmf.hpp"

#include <range/v3/view/zip.hpp>

namespace ccs::xdmf
{

namespace
{

void
append_xdmf(pugi::xml_document& doc,
            int step,
            real time,
            std::span<const std::string> var_names,
            std::span<const std::string> file_names,
            const std::string& dimensions)
{
    // Get first grid node
    auto time_series = doc.root().first_element_by_path("Xdmf/Domain/Grid");

    // Add a grid to main grid
    auto grid = time_series.append_child("Grid");
    grid.append_attribute("Name") = step;
    grid.append_attribute("GridType") = "Uniform";

    // topology
    auto topo = grid.append_child("Topology");
    topo.append_attribute("Reference") = "/Xdmf/Domain/Topology[1]";
    // geometry
    auto geom = grid.append_child("Geometry");
    geom.append_attribute("Reference") = "/Xdmf/Domain/Geometry[1]";
    // time
    auto tm = grid.append_child("Time");
    tm.append_attribute("Value") = time;

    for (auto&& [v, f] : vs::zip(var_names, file_names)) {
        auto attr = grid.append_child("Attribute");
        attr.append_attribute("Name") = v.c_str();
        auto item = attr.append_child("DataItem");
        item.append_attribute("Dimensions") = dimensions.c_str();
        item.append_attribute("NumberType") = "Float";
        item.append_attribute("Precision") = "8";
        item.append_attribute("Format") = "Binary";
        item.text() = f.c_str();
    }
}

std::string header(const int3& i, const domain_extents& d)
{

    auto&& [min, max] = d;
    return fmt::format(R"(<?xml version="1.0" encoding="utf-8"?>
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
</Grid>
</Domain>
</Xdmf>
)",
                       fmt::join(i, " "),
                       fmt::join(min, " "),
                       fmt::join(max, " "));

    // // capture dimensions for later use
    // std::ostringstream dims{};
    // dims << bounds[0] << " " << bounds[1];
    // if (bounds[2] > 1) dims << " " << bounds[2];
    // dimensions = dims.str();

    // std::ostringstream ss{};
    // ss << R"(<?xml version="1.0" encoding="utf-8"?>
    //              <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    //              <Xdmf Version="3.0">
    //              <Domain>
    //             )";
    // if (bounds[2] == 1) {
    //     ss << R"(<Topology TopologyType="2DCoRectMesh" Dimensions=")" << dimensions
    //        << R"("/>
    //                      <Geometry GeometryType="Origin_DxDy">
    //                      <DataItem Format="XML" NumberType="Float" Dimensions="2">)"
    //        << 0.0 << " " << 0.0 << R"(</DataItem>
    //                      <DataItem Format="XML" NumberType="Float" Dimensions="2">)"
    //        << length[0] << " " << length[1] << "</DataItem></Geometry>";
    // } else {
    //     ss << R"(<Topology TopologyType="3DCoRectMesh" Dimensions=")" << dimensions
    //        << R"("/>
    //                     <Geometry GeometryType="Origin_DxDyDz">
    //                     <DataItem Format="XML" NumberType="Float" Dimensions="3">)"
    //        << 0.0 << " " << 0.0 << " " << 0.0 << R"(</DataItem>
    //                     <DataItem Format="XML" NumberType="Float" Dimensions="3">)"
    //        << length[0] << " " << length[1] << " " << length[2]
    //        << "</DataItem></Geometry>";
    // }
    // ss << R"(<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal"/>
    //             </Domain></Xdmf>)";
    // header = ss.str();
}
} // namespace

//
// Main interface to writing xmf file
//
std::iostream& write(std::iostream& stream,
                     index_extents i,
                     const domain_extents& d,
                     int grid_number,
                     real time,
                     std::span<const std::string> var_names,
                     std::span<const std::string> file_names)
{
    pugi::xml_document doc{};

    auto p = stream.tellp();

    // if initial write then we need to generate the header
    if (grid_number == 0) {
        std::string h = header(i, d);

        // build file from scratch
        if (!doc.load_string(h.c_str(), pugi::parse_full)) {
            std::cerr << "Failed to load initial xdmf string\n";
            std::terminate();
        }
    } else {
        // load file
        if (!doc.load(stream, pugi::parse_full)) {
            std::cerr << "Failed to reload xdmf file\n";
            std::terminate();
        }
    }

    append_xdmf(doc,
                grid_number,
                time,
                var_names,
                file_names,
                fmt::format("{}", fmt::join(i.extents, " ")));

    // reset stream to begining so we can save it to file
    stream.seekp(p);

    doc.save(stream);

    return stream;
}

// std::iostream&
// xdmf::readwrite(std::iostream& stream,
//                 int step,
//                 double time,
//                 const std::vector<std::pair<std::string, std::string>>& vars_and_files)
// {
//     pugi::xml_document doc{};

//     auto p = stream.tellp();

//     // load file
//     if (!doc.load(stream, pugi::parse_full)) {
//         std::cerr << "Failed to reload xdmf file\n";
//         std::terminate();
//     }

//     append_xdmf(doc, step, time, vars_and_files, dimensions);

//     stream.seekp(p);

//     doc.save(stream);

//     return stream;
// }

} // namespace ccs::xdmf
