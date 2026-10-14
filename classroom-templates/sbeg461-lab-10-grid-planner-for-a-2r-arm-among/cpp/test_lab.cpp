// Auto-checks for SBEG 461 Lab 10 (grid planner for a planar 2R arm among obstacles) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*astar*"          (run only some tests, like pytest -k)
// Ground truth (independent of your code): mrcpp fkin_space (the C++ port of modern_robotics) for the 2R arm,
// brute-force dense sampling of the link capsules (written below), and cpp_cases.json, which holds the random
// inputs that test_lab.py draws with NumPy, the scipy.sparse.csgraph shortest-path costs on the same grid graphs
// and the numbers of the Python reference, so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using lab::Cell;
using sbeg::as;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr int K_FAST = 36;  // coarse grid for the heavier tests (10 deg)
constexpr int K_LAB = 72;   // lab grid (5 deg)

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

Vector2d v2(const json& j) { return as<Vector2d>(j); }
Cell cell_of(const json& j) { return {j[0].get<int>(), j[1].get<int>()}; }

/// math.isclose(a, b) with the default rel_tol = 1e-9.
bool isclose(double a, double b, double rel = 1e-9, double abs_tol = 0.0) {
    return std::abs(a - b) <= std::max(rel * std::max(std::abs(a), std::abs(b)), abs_tol);
}

// ----------------------------------------------------------------------------- independent reference helpers
/// numpy.linspace(0, 1, n)[i]
double lin(int i, int n) { return i == n - 1 ? 1.0 : i * (1.0 / (n - 1)); }

/// Points along both link axes (brute force), 2n x 2 (row = point).
MatrixXd dense_points(const Vector2d& q, const Vector2d& L, int n = 400) {
    const Vector2d e = L(0) * Vector2d(std::cos(q(0)), std::sin(q(0)));
    const Vector2d tip = e + L(1) * Vector2d(std::cos(q(0) + q(1)), std::sin(q(0) + q(1)));
    MatrixXd P(2 * n, 2);
    for (int i = 0; i < n; ++i) {
        const double s = lin(i, n);
        P.row(i) = (s * e).transpose();
        P.row(n + i) = (e + s * (tip - e)).transpose();
    }
    return P;
}

double brute_clearance(const Vector2d& q, const lab::Scene& sc) {
    const MatrixXd P = dense_points(q, sc.L);
    double best = std::numeric_limits<double>::infinity();
    for (Eigen::Index m = 0; m < sc.circles.rows(); ++m) {
        const Eigen::RowVector2d c(sc.circles(m, 0), sc.circles(m, 1));
        best = std::min(best, (P.rowwise() - c).rowwise().norm().minCoeff() - sc.link_radius - sc.circles(m, 2));
    }
    if (sc.floor) best = std::min(best, P.col(1).minCoeff() - sc.link_radius - *sc.floor);
    return best;
}

const lab::Scene& ward() {
    static const lab::Scene s = lab::load_scene("ward");
    return s;
}
const lab::Scene& eod() {
    static const lab::Scene s = lab::load_scene("eod");
    return s;
}
const lab::Scene& scene(const std::string& name) { return name == "ward" ? ward() : eod(); }

/// The `grids` fixture: your C-space grid of a scene at K_FAST (computed by your cspace_grid).
lab::BoolGrid grid(const std::string& name) {
    const lab::BoolGrid free = lab::cspace_grid(scene(name), K_FAST);
    REQUIRE(free.rows() == K_FAST);
    REQUIRE(free.cols() == K_FAST);
    return free;
}

/// The reference C-space grid of a scene at K_FAST (from cpp_cases.json). The SciPy shortest-path values in
/// cpp_cases.json were computed on this grid, so the tests that use them search this grid (test_lab.py runs
/// SciPy on the grid it searches; your own grid is compared with this one in test_cspace_grid_matches_brute_force).
lab::BoolGrid reference_grid(const std::string& name) {
    const json& ref = cases()["cspace_grid_matches_brute_force"][name]["free"];
    lab::BoolGrid free(K_FAST, K_FAST);
    for (int i = 0; i < K_FAST; ++i)
        for (int j = 0; j < K_FAST; ++j)
            free(i, j) = ref[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<bool>();
    return free;
}

/// Undirected weighted graph from an edge list (letters -> node numbers), as graph_fns in test_lab.py.
struct Graph {
    std::map<int, std::vector<std::pair<int, double>>> E;
    lab::Neighbors neighbors() const {
        return [this](int n) { return E.at(n); };
    }
};
Graph graph_fns(const std::vector<std::tuple<char, char, double>>& edges) {
    Graph g;
    for (const auto& [a, b, c] : edges) {
        g.E[a].push_back({b, c});
        g.E[b].push_back({a, c});
    }
    return g;
}
std::vector<int> nodes(const std::string& s) { return std::vector<int>(s.begin(), s.end()); }  // "SACG" -> path
}  // namespace

// ============================================================================ Checkpoint 1: geometry & collision
TEST_CASE("test_fk_matches_modern_robotics") {
    for (const auto& c : cases()["fk_matches_modern_robotics"]) {
        const Vector2d q = v2(c["q"]);
        INFO("q = " << q.transpose());
        const lab::ArmPoints P = lab::fk_2r(q, lab::L_PLANAR_2R);
        const Eigen::Matrix4d T = mr::fkin_space(lab::M_PLANAR_2R, lab::SLIST_PLANAR_2R, Eigen::VectorXd(q));
        CHECK_MAT_NEAR(P.p2, Vector2d(T(0, 3), T(1, 3)), 1e-12);
        CHECK_MAT_NEAR(P.p0, Vector2d::Zero(), 1e-8);
        CHECK(isclose(P.p1.norm(), 0.6));
        CHECK_MAT_NEAR(P.p2, v2(c["p2"]), 1e-12);
    }
}

TEST_CASE("test_point_segment_distance") {
    const Vector2d a(0.0, 0.0), b(1.0, 0.0);
    CHECK(isclose(lab::point_segment_distance(Vector2d(0.5, 0.3), a, b), 0.3));
    CHECK(isclose(lab::point_segment_distance(Vector2d(-0.3, 0.4), a, b), 0.5));  // beyond end a
    CHECK(isclose(lab::point_segment_distance(Vector2d(1.6, -0.8), a, b), 1.0));  // beyond end b
    for (const auto& c : cases()["point_segment_distance"]) {
        const double d = lab::point_segment_distance(v2(c["p"]), v2(c["a"]), v2(c["b"]));
        CHECK_SCALAR_NEAR(d, c["ref"].get<double>(), 1e-6);  // ref: dense sampling of the segment
    }
}

TEST_CASE("test_capsule_circle_distance_sign") {
    const Vector2d a(0.0, 0.0), b(1.0, 0.0);
    CHECK(isclose(lab::capsule_circle_distance(a, b, 0.05, Vector2d(0.5, 0.3), 0.1), 0.15));
    CHECK(isclose(lab::capsule_circle_distance(a, b, 0.05, Vector2d(0.5, 0.1), 0.1), -0.05));  // penetration
}

TEST_CASE("test_clearance_hand_values") {
    // q = 0: arm straight along +x. Ward: head (0.70, 0.30) r 0.12 -> 0.30 - 0.03 - 0.12 = 0.15.
    CHECK_SCALAR_NEAR(lab::clearance(Vector2d(0.0, 0.0), ward()), 0.15, 1e-9);
    // EOD: bumper (0.80, 0.10) r 0.08, link radius 0.04 -> 0.10 - 0.12 = -0.02 (collision)
    CHECK_SCALAR_NEAR(lab::clearance(Vector2d(0.0, 0.0), eod()), -0.02, 1e-9);
    CHECK_FALSE(lab::config_free(Vector2d(0.0, 0.0), eod()));
    CHECK(lab::config_free(Vector2d(0.0, 0.0), ward()));
    // EOD floor y = -0.35: link 1 pointing straight down (theta1 = -90 deg) is in the ground
    CHECK(lab::clearance(Vector2d(-PI / 2, 0.0), eod()) < 0);
}

TEST_CASE("test_clearance_matches_brute_force") {
    for (const std::string name : {"ward", "eod"}) {
        for (const auto& c : cases()["clearance_matches_brute_force"][name]) {
            const Vector2d q = v2(c["q"]);
            INFO(name << ", q = " << q.transpose());
            CHECK_SCALAR_NEAR(lab::clearance(q, scene(name)), brute_clearance(q, scene(name)), 2e-3);
            CHECK_SCALAR_NEAR(lab::clearance(q, scene(name)), c["clearance"].get<double>(), 1e-9);
        }
    }
}

TEST_CASE("test_config_free_joint_limits") {
    CHECK(lab::config_free(Vector2d(0.0, 0.5), ward()));
    CHECK_FALSE(lab::config_free(Vector2d(3.3, 0.5), ward()));
    CHECK_FALSE(lab::config_free(Vector2d(0.0, -3.3), ward()));
}

TEST_CASE("test_start_and_goal_are_free") {
    for (const lab::Scene* sc : {&ward(), &eod()}) {
        INFO(sc->name);
        CHECK(lab::config_free(sc->q_start, *sc));
        CHECK(lab::config_free(sc->q_goal, *sc));
    }
}

// ============================================================================ Checkpoint 2: C-space grid & edges
TEST_CASE("test_grid_angles_and_cells") {
    const int k = 72;
    const Eigen::VectorXd th = lab::grid_angles(k);
    REQUIRE(th.size() == k);
    CHECK(isclose(th(0), -PI));
    CHECK(isclose(th(1) - th(0), 2 * PI / k));
    for (const auto& c : cases()["grid_angles_and_cells"]["cells"]) {
        const Cell cell = cell_of(c);
        INFO("cell = (" << cell.first << ", " << cell.second << ")");
        CHECK(lab::q_to_cell(lab::cell_to_q(cell, k), k) == cell);
    }
    const Cell wrapped(0, 36);
    CHECK(lab::q_to_cell(Vector2d(PI - 1e-9, 0.0), k) == wrapped);  // +pi wraps to the -pi cell
}

TEST_CASE("test_cspace_grid_matches_brute_force") {
    const json& C = cases()["cspace_grid_matches_brute_force"];
    const Eigen::VectorXd th = lab::grid_angles(K_FAST);
    for (const std::string name : {"ward", "eod"}) {
        INFO(name);
        const lab::BoolGrid free = grid(name);
        for (const auto& s : C[name]["samples"]) {
            const Cell c = cell_of(s["cell"]);
            const double bc = brute_clearance(Vector2d(th(c.first), th(c.second)), scene(name));
            if (std::abs(bc) > 3e-3) {  // skip grazing contacts
                INFO("cell (" << c.first << ", " << c.second << "), brute-force clearance " << bc);
                CHECK(free(c.first, c.second) == (bc > 0));
            }
        }
        // the whole grid equals the Python reference grid (no grid configuration is within 1e-9 of contact)
        const auto& ref = C[name]["free"];
        int wrong = 0;
        for (int i = 0; i < K_FAST; ++i)
            for (int j = 0; j < K_FAST; ++j)
                if (free(i, j) != ref[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get<bool>()) ++wrong;
        CHECK_MESSAGE(wrong == 0, wrong << " of " << K_FAST * K_FAST << " cells differ from the reference grid");
    }
}

TEST_CASE("test_link1_obstacle_is_a_full_band") {
    // A C-obstacle of link 1 blocks every theta2 (a vertical band, MR Fig. 10.2): the IV pole.
    const int k = K_LAB;
    const lab::BoolGrid free = lab::cspace_grid(ward(), k);
    REQUIRE(free.rows() == k);
    REQUIRE(free.cols() == k);
    const int i = lab::q_to_cell(Vector2d(std::atan2(0.55, -0.30), 0.0), k).first;
    REQUIRE((0 <= i && i < k));
    CHECK_FALSE(free.row(i).any());
    const double mean = static_cast<double>(free.count()) / static_cast<double>(k * k);
    CHECK(0.80 < mean);
    CHECK(mean < 0.93);
    CHECK(i == cases()["link1_obstacle_is_a_full_band"]["row"].get<int>());
}

TEST_CASE("test_motion_bound_is_an_upper_bound") {
    const Vector2d L = lab::L_PLANAR_2R;
    for (const auto& c : cases()["motion_bound_is_an_upper_bound"]) {
        const Vector2d qa = v2(c["qa"]), dq = v2(c["dq"]);
        const MatrixXd Pa = dense_points(qa, L, 200);
        double worst = 0.0;
        for (int i = 0; i < 21; ++i)
            worst = std::max(worst, (dense_points(qa + lin(i, 21) * dq, L, 200) - Pa).rowwise().norm().maxCoeff());
        INFO("qa = " << qa.transpose() << ", dq = " << dq.transpose() << ", worst = " << worst);
        CHECK(worst <= lab::motion_bound(dq, L) + 1e-9);
        CHECK_SCALAR_NEAR(lab::motion_bound(dq, L), c["bound"].get<double>(), 1e-12);
    }
}

TEST_CASE("test_sampled_check_can_miss_but_certified_cannot") {
    // Workshop/Practice 11 case: arm sweeps straight from 0 to 90 deg past a thin pole at (0.5, 0.5).
    lab::Scene sc;
    sc.L = Vector2d(0.6, 0.4);
    sc.link_radius = 0.03;
    sc.circles = MatrixXd(1, 3);
    sc.circles << 0.5, 0.5, 0.02;
    const Vector2d qa(0.0, 0.0), qb(PI / 2, 0.0);
    CHECK(lab::edge_free(qa, qb, sc, 18.0 * lab::ONE_DEG));  // samples 0, 18, ..., 90 all free
    CHECK_FALSE(lab::edge_free(qa, qb, sc, 2.0 * lab::ONE_DEG));
    CHECK_FALSE(lab::edge_free_certified(qa, qb, sc, 18.0 * lab::ONE_DEG));
}

TEST_CASE("test_certified_true_implies_dense_free") {
    int n_true = 0;
    for (const std::string name : {"ward", "eod"}) {
        for (const auto& c : cases()["certified_true_implies_dense_free"][name]) {  // pairs inside the limits
            const Vector2d qa = v2(c["qa"]), qb = v2(c["qb"]);
            if (lab::edge_free_certified(qa, qb, scene(name), 2.0 * lab::ONE_DEG, false)) {
                ++n_true;
                INFO(name << ": qa = " << qa.transpose() << ", qb = " << qb.transpose());
                bool all_free = true;
                for (int i = 0; i < 201; ++i)
                    all_free = all_free && brute_clearance(qa + lin(i, 201) * (qb - qa), scene(name)) > -1e-3;
                CHECK(all_free);
            }
        }
    }
    CHECK(n_true > 10);
}

TEST_CASE("test_edge_free_wraps_the_short_way") {
    const Vector2d qa(170 * lab::ONE_DEG, -60 * lab::ONE_DEG), qb(-170 * lab::ONE_DEG, -60 * lab::ONE_DEG);
    CHECK(lab::edge_free(qa, qb, ward(), lab::ONE_DEG, true));         // 20 deg through +-180: free
    CHECK_FALSE(lab::edge_free(qa, qb, ward(), lab::ONE_DEG, false));  // 340 deg the long way: IV-pole band
}

// ============================================================================ Checkpoint 3: graph search
// Nodes are the letters' character codes ('S', 'A', ...), so a path reads as a string.
TEST_CASE("test_astar_workshop_problem_3") {
    const Graph g = graph_fns({{'S', 'A', 3.0}, {'S', 'B', 4.0}, {'A', 'C', 3.5}, {'A', 'E', 4.5}, {'B', 'C', 4.5},
                               {'B', 'D', 3.0}, {'C', 'G', 4.5}, {'C', 'E', 3.7}, {'E', 'G', 4.2}, {'D', 'G', 6.2}});
    const std::map<int, double> H = {{'S', 9.4}, {'A', 7.0}, {'B', 8.0}, {'C', 4.4},
                                     {'D', 6.0}, {'E', 4.1}, {'G', 0.0}};
    const lab::SearchResult r = lab::astar('S', 'G', g.neighbors(), [&H](int n) { return H.at(n); });
    CHECK(r.success);
    CHECK(r.path == nodes("SACG"));
    CHECK(isclose(r.cost, 11.0));
    CHECK(r.expanded == 4);
    const lab::SearchResult d = lab::dijkstra('S', 'G', g.neighbors());
    CHECK(d.path == nodes("SACG"));
    CHECK(isclose(d.cost, 11.0));
    CHECK(d.expanded == 7);
}

TEST_CASE("test_inconsistent_heuristic_needs_reopening") {
    // Workshop 11 Problem 5: h admissible but not consistent (h(A) = 4 > c(A,B) + h(B) = 1).
    const Graph g = graph_fns({{'S', 'A', 1.0}, {'A', 'B', 1.0}, {'S', 'B', 3.0}, {'B', 'G', 3.0}});
    const std::map<int, double> H = {{'S', 2.0}, {'A', 4.0}, {'B', 0.0}, {'G', 0.0}};
    const lab::Heuristic h = [&H](int n) { return H.at(n); };
    lab::SearchResult r = lab::astar('S', 'G', g.neighbors(), h, 1.0, false);
    CHECK(r.path == nodes("SBG"));  // MR Algorithm 10.1: suboptimal
    CHECK(isclose(r.cost, 6.0));
    r = lab::astar('S', 'G', g.neighbors(), h, 1.0, true);
    CHECK(r.path == nodes("SABG"));
    CHECK(isclose(r.cost, 5.0));
}

TEST_CASE("test_astar_failure") {
    const Graph g = graph_fns({{'S', 'A', 1.0}, {'B', 'G', 1.0}});
    const lab::SearchResult r = lab::astar('S', 'G', g.neighbors(), [](int) { return 0.0; });
    CHECK_FALSE(r.success);
    CHECK(r.path.empty());
    CHECK(std::isinf(r.cost));
}

TEST_CASE("test_grid_neighbors") {
    const int k = 10;
    const double d = 2 * PI / 10;
    const auto n4 = lab::grid_neighbors({0, 0}, k, 4, true);
    const auto n8 = lab::grid_neighbors({0, 0}, k, 8, true);
    REQUIRE(n4.size() == 4);
    REQUIRE(n8.size() == 8);
    std::set<Cell> c4;
    for (const auto& [c, w] : n4) c4.insert(c);
    const std::set<Cell> expected4 = {{1, 0}, {9, 0}, {0, 1}, {0, 9}};
    CHECK(c4 == expected4);
    std::vector<double> w8;
    for (const auto& [c, w] : n8) w8.push_back(w / d);
    std::sort(w8.begin(), w8.end());
    for (int i = 0; i < 8; ++i) CHECK_SCALAR_NEAR(w8[static_cast<std::size_t>(i)], i < 4 ? 1.0 : std::sqrt(2.0), 5e-7);
    CHECK(lab::grid_neighbors({0, 0}, k, 4, false).size() == 2);
    CHECK(lab::grid_neighbors({0, 0}, k, 8, false).size() == 3);
    CHECK(lab::grid_neighbors({5, 5}, k, 8, false).size() == 8);
    lab::BoolGrid free = lab::BoolGrid::Constant(k, k, true);
    free(1, 0) = false;
    const Cell blocked(1, 0);
    for (const auto& [c, w] : lab::grid_neighbors({0, 0}, k, 4, true, free)) CHECK(c != blocked);
}

TEST_CASE("test_heuristics_admissible_and_consistent") {
    const json& C = cases()["heuristics_admissible_and_consistent"];
    const lab::BoolGrid free = reference_grid("eod");
    const int k = K_FAST;
    const bool wrap = false;
    const Cell goal = lab::q_to_cell(eod().q_goal, k);
    REQUIRE(goal == cell_of(C["goal"]));  // the cost-to-go tables were computed for this goal
    struct Case {
        int conn;
        std::vector<std::string> good, bad;
    };
    for (const Case& cs : {Case{4, {"zero", "manhattan", "euclidean", "octile"}, {}},
                           Case{8, {"zero", "euclidean", "octile"}, {"manhattan"}}}) {
        const json& T = C["cost_to_go_" + std::to_string(cs.conn)];  // scipy shortest paths; null = unreachable
        std::vector<std::pair<Cell, double>> cells;                   // free cells with a finite cost-to-go
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                const json& t = T[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (free(i, j) && !t.is_null()) cells.push_back({{i, j}, t.get<double>()});
            }
        REQUIRE(!cells.empty());
        for (const std::string& name : cs.good) {
            INFO(cs.conn << "-connected, heuristic " << name);
            bool admissible = true, consistent = true;
            for (const auto& [c, truth] : cells)
                admissible = admissible && lab::grid_heuristic(c, goal, k, name, wrap) <= truth + 1e-9;
            for (std::size_t m = 0; m < cells.size(); m += 7) {  // consistency on every edge of the cell
                const Cell c = cells[m].first;
                for (const auto& [n, w] : lab::grid_neighbors(c, k, cs.conn, wrap, free))
                    consistent = consistent && lab::grid_heuristic(c, goal, k, name, wrap) <=
                                                   w + lab::grid_heuristic(n, goal, k, name, wrap) + 1e-9;
            }
            CHECK(admissible);
            CHECK(consistent);
        }
        for (const std::string& name : cs.bad) {
            INFO(cs.conn << "-connected, heuristic " << name << " must overestimate somewhere");
            bool over = false;
            for (const auto& [c, truth] : cells) over = over || lab::grid_heuristic(c, goal, k, name, wrap) > truth + 1e-9;
            CHECK(over);
        }
    }
}

TEST_CASE("test_grid_astar_optimal_vs_scipy") {
    // pytest: 6 parametrized items (ward/eod x 4-manhattan, 8-euclidean, 8-octile); here one loop.
    const int k = K_FAST;
    for (const auto& c : cases()["grid_astar_optimal_vs_scipy"]) {
        const std::string name = c["name"].get<std::string>(), heur = c["heur"].get<std::string>();
        const int conn = c["conn"].get<int>();
        const lab::Scene& sc = scene(name);
        const bool wrap = sc.wrap;
        INFO(name << ", " << conn << "-connected, " << heur);
        const lab::BoolGrid free = reference_grid(name);  // the grid of the SciPy reference cost
        const Cell s = lab::q_to_cell(sc.q_start, k), g = lab::q_to_cell(sc.q_goal, k);
        REQUIRE(s == cell_of(c["start"]));
        REQUIRE(g == cell_of(c["goal"]));
        const double ref = c["ref"].get<double>();  // scipy.sparse.csgraph.shortest_path on the same grid graph
        const lab::GridSearchResult r = lab::grid_astar(free, s, g, conn, heur, 1.0, wrap);
        const lab::GridSearchResult dj = lab::grid_astar(free, s, g, conn, "zero", 1.0, wrap);
        REQUIRE(r.success);
        CHECK(isclose(r.cost, ref));
        CHECK(isclose(dj.cost, ref));
        CHECK(r.expanded < dj.expanded);  // the heuristic saves work
        // the path is a chain of free neighbouring cells whose costs add up
        const std::vector<Cell>& path = r.path;
        REQUIRE(!path.empty());
        CHECK(path.front() == s);
        CHECK(path.back() == g);
        double total = 0.0;
        for (std::size_t m = 0; m < path.size(); ++m) {
            CHECK(free(path[m].first, path[m].second));
            if (m + 1 == path.size()) break;
            bool found = false;
            for (const auto& [nb, w] : lab::grid_neighbors(path[m], k, conn, wrap))
                if (nb == path[m + 1]) {
                    found = true;
                    total += w;
                }
            CHECK(found);
        }
        CHECK(isclose(total, r.cost));
    }
}

TEST_CASE("test_inflated_astar_bound") {
    const lab::BoolGrid free = grid("ward");
    const int k = K_FAST;
    const Cell s = lab::q_to_cell(ward().q_start, k), g = lab::q_to_cell(ward().q_goal, k);
    const lab::GridSearchResult opt = lab::grid_astar(free, s, g, 8, "euclidean", 1.0, true);
    for (const double eta : {1.5, 3.0}) {
        INFO("eta = " << eta);
        const lab::GridSearchResult r = lab::grid_astar(free, s, g, 8, "euclidean", eta, true);
        CHECK(r.success);
        CHECK(opt.cost - 1e-9 <= r.cost);
        CHECK(r.cost <= eta * opt.cost + 1e-9);
    }
}

TEST_CASE("test_topology_wrap_matters") {
    // The ward IV-pole band cuts the square; only the torus (wrap) connects start and goal.
    const lab::BoolGrid free = grid("ward");
    const int k = K_FAST;
    const Cell s = lab::q_to_cell(ward().q_start, k), g = lab::q_to_cell(ward().q_goal, k);
    CHECK(lab::grid_astar(free, s, g, 8, "euclidean", 1.0, true).success);
    CHECK_FALSE(lab::grid_astar(free, s, g, 8, "euclidean", 1.0, false).success);
}

TEST_CASE("test_enclosed_goal_reports_failure") {
    const int k = 12;
    lab::BoolGrid free = lab::BoolGrid::Constant(k, k, true);
    free.block(4, 4, 5, 5).setConstant(false);
    free(6, 6) = true;  // goal inside a closed C-obstacle ring
    const lab::GridSearchResult r = lab::grid_astar(free, {0, 0}, {6, 6}, 8, "euclidean", 1.0, true);
    CHECK_FALSE(r.success);
    CHECK(r.path.empty());
}

// ============================================================================ Checkpoint 4: the planner
TEST_CASE("test_plan_2r") {
    // pytest: 2 parametrized items (ward, eod); here one loop.
    for (const std::string name : {"ward", "eod"}) {
        INFO(name);
        const lab::Scene& sc = scene(name);
        const lab::PlanResult res = lab::plan_2r(sc, K_LAB, 8, "euclidean", 1.0, true);
        REQUIRE(res.success);
        const MatrixXd& qp = res.q_path;
        REQUIRE(qp.rows() >= 2);
        REQUIRE(qp.cols() == 2);
        CHECK_MAT_NEAR(lab::wrap_angle(Vector2d(qp.row(0).transpose() - sc.q_start)), Vector2d::Zero(), 1e-8);
        CHECK_MAT_NEAR(lab::wrap_angle(Vector2d(qp.row(qp.rows() - 1).transpose() - sc.q_goal)), Vector2d::Zero(),
                       1e-8);
        const double max_step = std::sqrt(2.0) * 2 * PI / K_LAB + 1e-9;
        bool continuous = true, segments_free = true;
        for (Eigen::Index i = 0; i + 1 < qp.rows(); ++i) {
            const Vector2d qa = qp.row(i).transpose(), qb = qp.row(i + 1).transpose();
            continuous = continuous && (qb - qa).cwiseAbs().maxCoeff() <= max_step;  // continuous (unwrapped) path
            for (int m = 0; m < 25; ++m)                                             // every segment free (dense)
                segments_free = segments_free && brute_clearance(qa + lin(m, 25) * (qb - qa), sc) > -1e-3;
        }
        CHECK(continuous);
        CHECK(segments_free);
        if (!sc.wrap) CHECK(qp.cwiseAbs().maxCoeff() <= PI + 1e-9);  // never through the cable stop
        CHECK(lab::path_length(qp) > 0);
    }
}

TEST_CASE("test_unwrap_and_length") {
    MatrixXd qs(3, 2);
    qs << 3.0, 0.0, -3.1, 0.0, -3.0, 0.1;
    const MatrixXd u = lab::unwrap_path(qs);
    REQUIRE(u.rows() == 3);
    REQUIRE(u.cols() == 2);
    CHECK_MAT_NEAR(u.row(1), Eigen::RowVector2d(2 * PI - 3.1, 0.0), 1e-8);
    CHECK(isclose(lab::path_length(u), (2 * PI - 6.1) + std::hypot(0.1, 0.1)));
}
