#include "field_data.hpp"
#include <fstream>


namespace ccs
{

void field_data::write_geom(std::span<const std::string> filenames,
                            std::array<std::span<const mesh_object_info>, 3> t) const
{
    auto f = [&]<int I>() {
        auto rng = get<I>(t);
        std::ofstream o(filenames[I]);
        for (auto&& info : rng) {
            auto&& pos = info.position;
            if (ix[2] == 1) {
                real3 tmp{pos[2], pos[1], pos[0]};
                const real* d = tmp.data();
                o.write(reinterpret_cast<const char*>(d),
                        tmp.size() * sizeof(real));
            } else {
                const real* d = pos.data();
                o.write(reinterpret_cast<const char*>(d),
                        pos.size() * sizeof(real));
            }
        }
    };

    f.template operator()<0>();
    f.template operator()<1>();
    f.template operator()<2>();
}

void field_data::write(std::span<const scalar_view> scalars,
                       std::span<const std::string> filenames) const
{
    unsigned long sz = ix[0] * ix[1] * ix[2] * sizeof(real);

    for (size_t idx = 0; idx < filenames.size(); ++idx) {
        auto& fname = filenames[idx];
        auto& sc = scalars[idx];
        std::ofstream o(fname);

        const real* d = sc.D.data();
        o.write(reinterpret_cast<const char*>(d), sz);

        auto write_component = [&](std::span<const real> rng) {
            if (rng.size() > 0) {
                d = rng.data();
                o.write(reinterpret_cast<const char*>(d), rng.size() * sizeof(*d));
            }
        };
        write_component(sc.Rx);
        write_component(sc.Ry);
        write_component(sc.Rz);
    }
}

} // namespace ccs
