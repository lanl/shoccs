#include "kokkos_types.hpp"
#include "matrices/block.hpp"
#include "matrices/csr.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

using namespace ccs;

TEST_CASE("Graph POC: 3-node chain")
{
    constexpr int N = 64;
    using view_t = Kokkos::View<double*, memory_space>;

    view_t a("a", N);
    view_t b("b", N);
    view_t c("c", N);

    // Initialize a[i] = i + 1
    Kokkos::parallel_for(
        "init_a", Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { a(i) = static_cast<double>(i + 1); });
    Kokkos::fence();

    SECTION("chain A -> B -> C")
    {
        // Build graph: A: b = a*2, B: c = b+1, C: a = c*3
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                auto node_a = root.then_parallel_for(
                    "A", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { b(i) = a(i) * 2.0; });

                auto node_b = node_a.then_parallel_for(
                    "B", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { c(i) = b(i) + 1.0; });

                node_b.then_parallel_for(
                    "C", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { a(i) = c(i) * 3.0; });
            });

        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        // Verify: a[i] = ((i+1)*2 + 1) * 3
        auto a_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
        for (int i = 0; i < N; ++i) {
            double expected = (static_cast<double>(i + 1) * 2.0 + 1.0) * 3.0;
            REQUIRE(a_h(i) == expected);
        }
    }

    SECTION("parallel_for then parallel_reduce")
    {
        // Result must be a View, not a scalar — graph nodes are async
        Kokkos::View<double, memory_space> sum("sum");

        // Build graph: A fills b[i] = i+1, then B reduces sum = Σ b[i]
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                auto node_a = root.then_parallel_for(
                    "fill", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { b(i) = static_cast<double>(i + 1); });

                node_a.then_parallel_reduce(
                    "reduce",
                    Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i, double& val) { val += b(i); },
                    sum);
            });

        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        // Verify: sum = N*(N+1)/2 = 64*65/2 = 2080
        auto sum_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sum);
        REQUIRE(sum_h() == static_cast<double>(N) * (N + 1) / 2.0);
    }

    SECTION("TeamPolicy graph node")
    {
        // Simulate a block-matrix-like operation: each "team" handles a row-group,
        // each "thread" handles one row, vector lanes do the dot product.
        // Matrix: 4 groups of 4 rows, each row dot-product with 8 columns.
        constexpr int n_teams = 4;
        constexpr int rows_per_team = 4;
        constexpr int cols = 8;
        constexpr int total = n_teams * rows_per_team * cols;
        constexpr int vector_len = 8;

        view_t matrix("matrix", total);
        view_t x_vec("x_vec", n_teams * cols);
        view_t result("result", n_teams * rows_per_team);

        // Initialize matrix[team][row][col] = (team * rows_per_team + row) * cols + col + 1
        // Initialize x[team * cols + col] = 1.0 (simple sum of coefficients)
        Kokkos::parallel_for(
            "init_mat", Kokkos::RangePolicy<execution_space>(0, total),
            KOKKOS_LAMBDA(int i) { matrix(i) = static_cast<double>(i + 1); });
        Kokkos::parallel_for(
            "init_x", Kokkos::RangePolicy<execution_space>(0, n_teams * cols),
            KOKKOS_LAMBDA(int i) { x_vec(i) = 1.0; });
        Kokkos::fence();

        using team_policy = Kokkos::TeamPolicy<execution_space>;
        using member_type = typename team_policy::member_type;

        auto mat = matrix;
        auto xv = x_vec;
        auto res = result;

        // Build graph with a single TeamPolicy node that performs blocked matvec
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                root.then_parallel_for(
                    "team_matvec", team_policy(n_teams, Kokkos::AUTO, vector_len),
                    KOKKOS_LAMBDA(const member_type& team) {
                        int t = team.league_rank();
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, rows_per_team),
                            [&](int row) {
                                double dot = 0;
                                Kokkos::parallel_reduce(
                                    Kokkos::ThreadVectorRange(team, cols),
                                    [&](int col, double& s) {
                                        int mat_idx =
                                            (t * rows_per_team + row) * cols + col;
                                        s += mat(mat_idx) * xv(t * cols + col);
                                    },
                                    dot);
                                Kokkos::single(Kokkos::PerThread(team), [&]() {
                                    res(t * rows_per_team + row) = dot;
                                });
                            });
                    });
            });

        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        // Verify: each result[t*rows_per_team + row] = sum of matrix row coefficients
        // Row i has coefficients (i*cols+1) .. (i*cols+cols), sum = cols*(i*cols+1) + cols*(cols-1)/2
        //   = cols * (i*cols + 1 + (cols-1)/2) = cols * (i*cols + (cols+1)/2)
        // For cols=8: sum_row_i = 8*i*8 + 8*(8+1)/2 = 64*i + 36
        auto res_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
        for (int i = 0; i < n_teams * rows_per_team; ++i) {
            // Row i: coefficients are i*8+1, i*8+2, ..., i*8+8
            // Sum = 8*(i*8) + (1+2+...+8) = 64*i + 36
            double expected = 64.0 * i + 36.0;
            REQUIRE(res_h(i) == expected);
        }
    }

    SECTION("fan-out and fan-in with when_all")
    {
        view_t d("d", N);

        // Reinitialize a
        Kokkos::parallel_for(
            "reinit_a", Kokkos::RangePolicy<execution_space>(0, N),
            KOKKOS_LAMBDA(int i) { a(i) = static_cast<double>(i + 1); });
        Kokkos::fence();

        // Build graph: root -> {A: b = a*2, B: c = a+10} -> when_all -> C: d = b + c
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                auto node_a = root.then_parallel_for(
                    "A", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { b(i) = a(i) * 2.0; });

                auto node_b = root.then_parallel_for(
                    "B", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { c(i) = a(i) + 10.0; });

                auto join = Kokkos::Experimental::when_all(node_a, node_b);

                join.then_parallel_for(
                    "C", Kokkos::RangePolicy<execution_space>(0, N),
                    KOKKOS_LAMBDA(int i) { d(i) = b(i) + c(i); });
            });

        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        // Verify: d[i] = (i+1)*2 + (i+1)+10 = 3*(i+1) + 10
        auto d_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
        for (int i = 0; i < N; ++i) {
            double expected = 3.0 * (i + 1) + 10.0;
            REQUIRE(d_h(i) == expected);
        }
    }
}

using Catch::Matchers::Approx;

TEST_CASE("CSR graph_node")
{
    using T = std::vector<real>;

    // 5x5 identity CSR
    T w{1, 1, 1, 1, 1};
    std::vector<integer> v{0, 1, 2, 3, 4};
    std::vector<integer> u{0, 1, 2, 3, 4, 5};
    const matrix::csr A{w, v, u};

    T x{1.0, 2.0, 3.0, 4.0, 5.0};

    SECTION("graph matches eager")
    {
        T b_eager(5, 0.0);
        A(x, b_eager);
        Kokkos::fence();

        T b_graph(5, 0.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                A.graph_node(root, x.data(), b_graph.data());
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        REQUIRE_THAT(b_graph, Approx(b_eager));
    }

    SECTION("sparse graph matches eager")
    {
        // Non-trivial sparse matrix (rows with multiple entries, some empty rows)
        auto bld = matrix::csr::builder();
        bld.add_point(0, 1, 3.0);
        bld.add_point(0, 3, -1.5);
        bld.add_point(2, 0, 2.0);
        bld.add_point(2, 2, 1.0);
        bld.add_point(4, 4, 5.0);
        const auto S = bld.to_csr(5);

        T b_eager(5, 0.0);
        S(x, b_eager);
        Kokkos::fence();

        T b_graph(5, 0.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                S.graph_node(root, x.data(), b_graph.data());
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        REQUIRE_THAT(b_graph, Approx(b_eager));
    }

    SECTION("empty CSR")
    {
        const matrix::csr empty_csr{};

        T b_sentinel(5, 42.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                empty_csr.graph_node(root, x.data(), b_sentinel.data());
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        // Empty CSR executes zero iterations — sentinel values unchanged
        for (int i = 0; i < 5; ++i)
            REQUIRE(b_sentinel[i] == 42.0);
    }
}

TEST_CASE("Block graph_node")
{
    using T = std::vector<real>;

    // Identity-like block: 4x4 left, width-1 circulant (identity), 2x2 right
    T left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    T int_c{1.0};
    T right_c{1, 0, 0, 1};

    auto bld = matrix::block::builder();
    bld.add_inner_block(16, 1, 1, 1,
                        matrix::dense{4, 4, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 2, right_c});
    const auto A = MOVE(bld).to_block();

    T x(A.rows());
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        x[i] = static_cast<real>(i + 1);

    SECTION("graph matches eager with eq")
    {
        T b_eager(x.size(), 0.0);
        A(x, b_eager);
        Kokkos::fence();

        T b_graph(x.size(), 0.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                A.graph_node(root, x.data(), b_graph.data());
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        REQUIRE_THAT(b_graph, Approx(b_eager));
    }

    SECTION("graph matches eager with plus_eq")
    {
        T b_eager(x.size(), 1.0);
        A(x, b_eager, plus_eq);
        Kokkos::fence();

        T b_graph(x.size(), 1.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                A.graph_node(root, x.data(), b_graph.data(), plus_eq);
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        REQUIRE_THAT(b_graph, Approx(b_eager));
    }

    SECTION("empty block")
    {
        const matrix::block empty_blk{};

        T b_sentinel(5, 42.0);
        auto graph = Kokkos::Experimental::create_graph<execution_space>(
            [&](auto root) {
                empty_blk.graph_node(root, x.data(), b_sentinel.data());
            });
        graph.instantiate();
        graph.submit();
        Kokkos::fence();

        for (int i = 0; i < 5; ++i)
            REQUIRE(b_sentinel[i] == 42.0);
    }
}

TEST_CASE("Block + CSR graph chain")
{
    using T = std::vector<real>;

    // Block: identity-like, 16 columns starting at offset 0
    T left_c{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    T int_c{1.0};
    T right_c{1, 0, 0, 1};

    auto bld = matrix::block::builder();
    bld.add_inner_block(16, 0, 0, 1,
                        matrix::dense{4, 4, left_c},
                        matrix::circulant{10, int_c},
                        matrix::dense{2, 2, right_c});
    const auto blk = MOVE(bld).to_block();

    // CSR: adds 2*x[1] to row 0, 3*x[0] to row 5
    auto csr_bld = matrix::csr::builder();
    csr_bld.add_point(0, 1, 2.0);
    csr_bld.add_point(5, 0, 3.0);
    const auto csr_mat = csr_bld.to_csr(16);

    T x(16);
    for (int i = 0; i < 16; ++i) x[i] = static_cast<real>(i + 1);

    // Eager: block sets (eq), then CSR accumulates (+=)
    T b_eager(16, 0.0);
    blk(x, b_eager, eq);
    csr_mat(x, b_eager);
    Kokkos::fence();

    // Graph: chain block -> CSR
    T b_graph(16, 0.0);
    auto graph = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) {
            auto node_blk = blk.graph_node(root, x.data(), b_graph.data());
            csr_mat.graph_node(node_blk, x.data(), b_graph.data());
        });
    graph.instantiate();
    graph.submit();
    Kokkos::fence();

    REQUIRE_THAT(b_graph, Approx(b_eager));
}
