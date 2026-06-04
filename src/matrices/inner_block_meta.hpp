#pragma once

namespace ccs::matrix
{

// POD struct holding per-line metadata for the TeamPolicy kernel in block::operator().
// One instance per inner_block (i.e., per line in the mesh). All offsets refer to
// positions within the flat input/output spans and the concatenated coefficient array.
struct inner_block_meta {
    int row_offset;
    int col_offset;
    int stride;
    int left_rows;
    int left_cols;
    int left_coeff_offset;
    int interior_rows;
    int interior_coeff_offset;
    int stencil_width;
    int right_rows;
    int right_cols;
    int right_coeff_offset;
    int right_col_offset;
};

} // namespace ccs::matrix
