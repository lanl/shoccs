#pragma once
#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "interval.hpp"
#include "types.hpp"
#include "fields/SystemField.hpp"

namespace ccs
{
// Forward decls
class StepController;

class xdmf
{
    std::string dimensions;
    std::string header;

public:
    xdmf() = default;

    xdmf(const int3& bounds, const real3& length);

    // this routine reads the xdmf file associated with the stream and writes
    // a new file into the output stream
    std::iostream&
    readwrite(std::iostream& stream,
              int step,
              real time,
              const std::vector<std::pair<std::string, std::string>>& vars_and_files);

    // this routine writes a new file into the output stream
    std::ostream&
    write(std::ostream& stream,
          const std::vector<std::pair<std::string, std::string>>& vars_and_files);
};

class field_data
{
    int sz; // won't work on large meshes

public:
    field_data() = default;

    field_data(const int3& bounds);

    std::ostream& write(std::ostream& stream, const double* data);
};

class FieldIO
{
#if 0
    // the string member are the variable names and the pointers are to the data
    std::vector<std::tuple<std::string, const double*>> v;
    xdmf xdmf_writer;
    field_data field_data_writer;
    interval<int> step_interval;
    interval<double> time_interval;
    std::string io_dir;
    // don't currently see a need to expose these to the user
    int suffix_length{6};
    int current_step{0};
    bool wont_write{true};
#endif
public:
    FieldIO() = default;

#if 0
    // this constructor takes r-value reference to writers to emphasize that we're
    // taking over
    field_io(xdmf&& xwriter,
             field_data&& fwriter,
             interval<int> step_interval = interval<int>{},
             interval<double> time_interval = interval<double>{},
             std::string io_dir = std::string{"io_dir"});
    // register a variable name and pointer to be written
    void add(const std::string& name, const double* data);
#endif

    void write(const SystemField& field, const StepController& controller, real dt);
};

} // namespace ccs
