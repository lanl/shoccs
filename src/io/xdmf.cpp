#include <iomanip>
#include <iostream>
#include <pugixml.hpp>
#include <sstream>
#include <string>

#include "FieldIO.hpp"

namespace ccs
{

static void
append_xdmf(pugi::xml_document& doc,
            int step,
            double time,
            const std::vector<std::pair<std::string, std::string>>& vars_and_files,
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

        for (const auto& [varname, filename] : vars_and_files) {
                auto attr = grid.append_child("Attribute");
                attr.append_attribute("Name") = varname.c_str();
                auto item = attr.append_child("DataItem");
                item.append_attribute("Dimensions") = dimensions.c_str();
                item.append_attribute("NumberType") = "Float";
                item.append_attribute("Precision") = "8";
                item.append_attribute("Format") = "Binary";
                item.text() = filename.c_str();
        }
}

xdmf::xdmf(const int3& bounds, const real3& length)
{
        // capture dimensions for later use
        std::ostringstream dims{};
        dims << bounds[0] << " " << bounds[1];
        if (bounds[2] > 1)
                dims << " " << bounds[2];
        dimensions = dims.str();

        std::ostringstream ss{};
        ss << R"(<?xml version="1.0" encoding="utf-8"?>
                 <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                 <Xdmf Version="3.0">
                 <Domain>
                )";
        if (bounds[2] == 1) {
                ss << R"(<Topology TopologyType="2DCoRectMesh" Dimensions=")"
                   << dimensions << R"("/>
                         <Geometry GeometryType="Origin_DxDy">
                         <DataItem Format="XML" NumberType="Float" Dimensions="2">)"
                   << 0.0 << " " << 0.0 << R"(</DataItem>
                         <DataItem Format="XML" NumberType="Float" Dimensions="2">)"
                   << length[0] << " " << length[1] << "</DataItem></Geometry>";
        } else {
                ss << R"(<Topology TopologyType="3DCoRectMesh" Dimensions=")"
                   << dimensions << R"("/>
                        <Geometry GeometryType="Origin_DxDyDz">
                        <DataItem Format="XML" NumberType="Float" Dimensions="3">)"
                   << 0.0 << " " << 0.0 << " " << 0.0 << R"(</DataItem>
                        <DataItem Format="XML" NumberType="Float" Dimensions="3">)"
                   << length[0] << " " << length[1] << " " << length[2]
                   << "</DataItem></Geometry>";
        }
        ss << R"(<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal"/>
                </Domain></Xdmf>)";
        header = ss.str();
}

std::ostream&
xdmf::write(std::ostream& stream,
            const std::vector<std::pair<std::string, std::string>>& vars_and_files)
{
        pugi::xml_document doc{};

        // build file from scratch
        if (!doc.load_string(header.c_str(), pugi::parse_full)) {
                std::cerr << "Failed to load initial xdmf string\n";
                std::terminate();
        }

        append_xdmf(doc, 0, 0.0, vars_and_files, dimensions);

        doc.save(stream);

        return stream;
}

std::iostream&
xdmf::readwrite(std::iostream& stream,
                int step,
                double time,
                const std::vector<std::pair<std::string, std::string>>& vars_and_files)
{
        pugi::xml_document doc{};

        auto p = stream.tellp();

        // load file
        if (!doc.load(stream, pugi::parse_full)) {
                std::cerr << "Failed to reload xdmf file\n";
                std::terminate();
        }

        append_xdmf(doc, step, time, vars_and_files, dimensions);

        stream.seekp(p);

        doc.save(stream);

        return stream;
}

} // namespace ccs
