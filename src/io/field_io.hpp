#pragma once
#include <array>
#include <string>
#include <vector>

#include "field_data.hpp"
#include "interval.hpp"
#include "logging.hpp"
#include "mesh/mesh_types.hpp"
#include "types.hpp"
#include "xdmf.hpp"

#include <sol/forward.hpp>

namespace ccs
{
// Forward decls
class step_controller;

class field_io
{

    xdmf xdmf_w;
    field_data field_data_w;

    d_interval dump_interval;
    std::string io_dir;
    int suffix_length;
    logs logger;

public:
    field_io() = default;

    field_io(xdmf&& xdmf_w,
             field_data&& fied_data_w,
             d_interval&& dump_interval,
             std::string&& io_dir,
             int suffix_length,
             const logs& = {});

    bool write(std::span<const std::string>,
               std::span<const scalar_view> scalars,
               const step_controller& controller,
               real dt,
               std::array<std::span<const mesh_object_info>, 3>);

    static std::optional<field_io> from_lua(const sol::table&, const logs& = {});
};

} // namespace ccs
