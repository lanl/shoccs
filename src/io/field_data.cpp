#include "field_data.hpp"
#include <fstream>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/algorithm/reverse_copy.hpp>
#include <range/v3/view/transform.hpp>

namespace ccs
{

void field_data::write_geom(std::span<const std::string> filenames,
                            tuple<std::span<const mesh_object_info>,
                                  std::span<const mesh_object_info>,
                                  std::span<const mesh_object_info>> t) const
{
    auto f = [&]<int I>() {
        auto rng = get<I>(t);
        std::ofstream o(filenames[I]);
        rs::for_each(rng | vs::transform(&mesh_object_info::position),
                     [&o, this](auto&& pos) {
                         if (ix[2] == 1) {
                             real3 tmp{pos[2], pos[1], pos[0]};
                             const real* d = tmp.data();
                             o.write(reinterpret_cast<const char*>(d),
                                     rs::size(tmp) * sizeof(real));
                         } else {
                             const real* d = pos.data();
                             o.write(reinterpret_cast<const char*>(d),
                                     rs::size(pos) * sizeof(real));
                         }
                     });
    };

    f.template operator()<0>();
    f.template operator()<1>();
    f.template operator()<2>();
}

void field_data::write(field_view f, std::span<const std::string> filenames) const
{
    unsigned long sz = ix[0] * ix[1] * ix[2] * sizeof(real);

    for (auto&& [fname, sc] : vs::zip(filenames, f.scalars())) {
        std::ofstream o(fname);

        const real* d = get<si::D>(sc).data();
        o.write(reinterpret_cast<const char*>(d), sz);

        auto g = [&]<int I>(auto&& r) {
            if (auto&& rng = get<I>(r); rs::size(rng) > 0) {
                d = rng.data();
                o.write(reinterpret_cast<const char*>(d), rs::size(rng) * sizeof(*d));
            }
        };
        g.template operator()<0>(sc | sel::R);
        g.template operator()<1>(sc | sel::R);
        g.template operator()<2>(sc | sel::R);
    }
}

} // namespace ccs
