#include <iomanip>
#include <iostream>
#include <pugixml.hpp>
#include <string>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "real3_operators.hpp"
#include "xdmf.hpp"

#include <range/v3/view/zip.hpp>

using namespace fmt::literals;

namespace ccs
{

namespace
{

void append_xdmf(pugi::xml_document& doc,
                 int step,
                 real time,
                 std::span<const std::string> var_names,
                 std::span<const std::string> file_names,
                 const std::string& dimensions,
                 tuple<std::span<const mesh_object_info>,
                       std::span<const mesh_object_info>,
                       std::span<const mesh_object_info>> t,
                 unsigned long f_sz)
{
    // Get first grid node
    auto time_series = doc.root().first_element_by_path("Xdmf/Domain/Grid");

    // Add a grid to main grid
    auto grid_col = time_series.append_child("Grid");
    grid_col.append_attribute("Name") = step;
    grid_col.append_attribute("GridType") = "Collection";
    grid_col.append_attribute("CollectionType") = "Spatial";

    // // time
    auto tm = grid_col.append_child("Time");
    tm.append_attribute("Value") = time;

    auto sub_grid = [&](auto&& rng,
                        const std::string& g_name,
                        int topo_idx,
                        const std::string& dims,
                        unsigned long offset) {
        auto sz = rs::size(rng);
        if (sz == 0) return offset;

        auto grid = grid_col.append_child("Grid");
        //        grid.append_attribute("Name") = fmt::format("{}{}", g_name,
        //        step).c_str();
        grid.append_attribute("Name") = g_name.c_str();
        grid.append_attribute("GridType") = "Uniform";

        // topology
        auto topo = grid.append_child("Topology");
        topo.append_attribute("Reference") =
            fmt::format("/Xdmf/Domain/Topology[{}]", topo_idx).c_str();
        // geometry
        auto geom = grid.append_child("Geometry");
        geom.append_attribute("Reference") =
            fmt::format("/Xdmf/Domain/Geometry[{}]", topo_idx).c_str();

        for (auto&& [v, f] : vs::zip(var_names, file_names)) {
            auto attr = grid.append_child("Attribute");
            attr.append_attribute("Name") = v.c_str();
            auto item = attr.append_child("DataItem");
            item.append_attribute("Dimensions") = dims.c_str();
            item.append_attribute("NumberType") = "Float";
            item.append_attribute("Precision") = "8";
            item.append_attribute("Format") = "Binary";
            if (offset > 0) { item.append_attribute("Seek") = offset; }
            item.text() = f.c_str();
        }

        return offset + sz * sizeof(real);
    };

    auto offset = sub_grid(vs::repeat_n(0, f_sz), "F", 1, dimensions, 0);
    auto&& [x, y, z] = t;
    offset = sub_grid(x, "RX", 2, fmt::format("{} 1", rs::size(x)), offset);
    offset = sub_grid(y, "RY", 3, fmt::format("{} 1", rs::size(y)), offset);
    offset = sub_grid(z, "RZ", 4, fmt::format("{} 1", rs::size(z)), offset);
}

std::string header(const int3& i,
                   const domain_extents& d,
                   tuple<std::span<const mesh_object_info>,
                         std::span<const mesh_object_info>,
                         std::span<const mesh_object_info>> t)
{
    auto&& [min, max] = d;
    real3 dxyz = (max - min) / clamp_lo(i - 1.0, 1.0);
    std::string header = fmt::format(R"(<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
<Domain>
<Topology TopologyType="3DCoRectMesh" Dimensions="{dims}"/>
<Geometry GeometryType="Origin_DxDyDz">
<DataItem Format="XML" NumberType="Float" Dimensions="3">{origin}</DataItem>
<DataItem Format="XML" NumberType="Float" Dimensions="3">{dxyz}</DataItem>
</Geometry>
<Topology TopologyType="Polyvertex" NumberOfElements="{rx}"/>
<Geometry GeometryType="XYZ">
<DataItem NumberType="Float" Precision="8" Format="Binary" Dimensions="{rx} 3">rx</DataItem>
</Geometry>
<Topology TopologyType="Polyvertex" NumberOfElements="{ry}"/>
<Geometry GeometryType="XYZ">
<DataItem NumberType="Float" Precision="8" Format="Binary" Dimensions="{ry} 3">ry</DataItem>
</Geometry>
<Topology TopologyType="Polyvertex" NumberOfElements="{rz}"/>
<Geometry GeometryType="XYZ">
<DataItem NumberType="Float" Precision="8" Format="Binary" Dimensions="{rz} 3">rz</DataItem>
</Geometry>
<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
</Grid>
</Domain>
</Xdmf>
)",
                                     "dims"_a = fmt::join(i.begin(), i.end(), " "),
                                     "origin"_a = fmt::join(min.begin(), min.end(), " "),
                                     "dxyz"_a = fmt::join(dxyz.begin(), dxyz.end(), " "),
                                     "rx"_a = rs::size(get<0>(t)),
                                     "ry"_a = rs::size(get<1>(t)),
                                     "rz"_a = rs::size(get<2>(t)));
    return header;
}
} // namespace

//
// Main interface to writing xmf file
//
void xdmf::write(int grid_number,
                 real time,
                 std::span<const std::string> var_names,
                 std::span<const std::string> file_names,
                 tuple<std::span<const mesh_object_info>,
                       std::span<const mesh_object_info>,
                       std::span<const mesh_object_info>> tp,
                 const logs& logger) const
{

    // if initial write then we need to generate the file with an outline
    if (grid_number == 0) {
        std::string h = header(ix, bounds, tp);

        // build file from scratch and overwrite whatever is there
        pugi::xml_document doc{};
        if (!doc.load_string(h.c_str(), pugi::parse_full)) {
            std::cerr << "Failed to load initial xdmf string\n";
            std::terminate();
        }
        doc.save_file(xmf_filename.c_str());
    }

    // process existing document
    pugi::xml_document doc{};
    if (!doc.load_file(xmf_filename.c_str(), pugi::parse_full)) {
        std::cerr << "Failed to reload xdmf file\n";
        std::terminate();
    }

    append_xdmf(doc,
                grid_number,
                time,
                var_names,
                file_names,
                fmt::format("{}", fmt::join(ix.extents, " ")),
                tp,
                ix[0] * ix[1] * ix[2]);

    doc.save_file(xmf_filename.c_str());

    logger(spdlog::level::info,
           "Update xdmf file: {}, with grid {} at time {}",
           xmf_filename,
           grid_number,
           time);
}

} // namespace ccs
