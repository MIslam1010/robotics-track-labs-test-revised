// SBEG 461 Lab 10 -- Grid planner for a planar 2R arm among obstacles (MR Sec. 10.2-10.4), C++ STARTER.
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 10 --cpp
// Rules: do not use mrcpp (sbeg/mrcpp.hpp) or any planning library here -- the tests use mrcpp and the numbers
// of the Python reference (cpp_cases.json) as ground truth.
//
// Eigen reminders: Eigen::Vector2d v(x, y);  v(0), v.dot(w), v.norm(), v.cwiseAbs().maxCoeff(), (v.array() > 1).any();
//   BoolGrid free(k, k); free(i, j) = true;  free.size(), free.rows();  std::clamp(x, 0.0, 1.0), std::hypot(a, b).
// Containers for A*: std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> with
//   using Entry = std::tuple<double, double, long long, int>;  // (f = past + eta*h, h, counter, node) as Python's heapq
//   std::map<int, double> / std::set<int> for past_cost / CLOSED.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace lab {

namespace {
// Given helpers (use them if you like).
[[maybe_unused]] constexpr double INF = std::numeric_limits<double>::infinity();

/// Python's float a % b for b > 0 (result in [0, b)).
double py_mod(double a, double b) {
    double m = std::fmod(a, b);
    if (m != 0.0) {
        if ((b < 0) != (m < 0)) m += b;
    } else {
        m = std::copysign(0.0, b);
    }
    return m;
}

/// Python's integer i % k for k > 0 (C++'s % can be negative: -1 % 72 == -1).
[[maybe_unused]] int py_imod(int i, int k) { return ((i % k) + k) % k; }

/// s = i / n as numpy.linspace(0, 1, n + 1)[i]  (i = 0..n; the last value is exactly 1).
[[maybe_unused]] double lin01(int i, int n) { return i == n ? 1.0 : i * (1.0 / n); }
}  // namespace

// ==========================================================================================
// Provided (do not edit): the scenes and small grid helpers
// ==========================================================================================

Scene load_scene(const std::string& name) {
    const sbeg::json all = sbeg::load_json(sbeg::lab_path("../data/scenes.json"));
    if (!all.contains(name)) throw std::invalid_argument("unknown scene: " + name);
    const sbeg::json& s = all[name];
    Scene sc;
    sc.name = name;
    sc.description = s.value("description", std::string());
    sc.L = sbeg::as<Eigen::Vector2d>(s["L"]);
    sc.link_radius = s["link_radius"].get<double>();
    const Eigen::MatrixXd c = sbeg::as_mat(s["circles"]);
    sc.circles = c.size() == 0 ? Eigen::MatrixXd(0, 3) : c;
    if (s.contains("circle_names")) sc.circle_names = s["circle_names"].get<std::vector<std::string>>();
    if (s.contains("floor") && !s["floor"].is_null()) sc.floor = s["floor"].get<double>();
    sc.wrap = s.value("wrap", true);
    const Eigen::Vector2d qs = sbeg::as<Eigen::Vector2d>(s["q_start_deg"]);
    const Eigen::Vector2d qg = sbeg::as<Eigen::Vector2d>(s["q_goal_deg"]);
    sc.q_start = qs * (PI / 180.0);
    sc.q_goal = qg * (PI / 180.0);
    return sc;
}

double wrap_angle(double a) { return py_mod(a + PI, 2 * PI) - PI; }

Eigen::Vector2d wrap_angle(const Eigen::Vector2d& a) { return Eigen::Vector2d(wrap_angle(a(0)), wrap_angle(a(1))); }

// ==========================================================================================
// Part A -- geometry and collision checking (MR Sec. 10.2.1-10.2.2)
// ==========================================================================================

ArmPoints fk_2r(const Eigen::Vector2d& q, const Eigen::Vector2d& L) {
    // TODO: p1 = L1 (cos t1, sin t1); p2 = p1 + L2 (cos(t1 + t2), sin(t1 + t2)); p0 = (0, 0).
    //       return {p0, p1, p2};
    throw std::logic_error("TODO: fk_2r");
}

double point_segment_distance(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    // TODO: Project p onto the line through a and b: t = (p - a).(b - a) / |b - a|^2,
    //       clip t to [0, 1], return |a + t (b - a) - p|. Guard against a == b.
    throw std::logic_error("TODO: point_segment_distance");
}

double capsule_circle_distance(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double r_link,
                               const Eigen::Vector2d& c, double r_obs) {
    // TODO: distance(centre, segment) - r_link - r_obs.
    throw std::logic_error("TODO: capsule_circle_distance");
}

double clearance(const Eigen::Vector2d& q, const Scene& scene) {
    // TODO: Loop over the two links (p0-p1, p1-p2) and every disc (row m of scene.circles: cx, cy, r); also the
    //       floor if scene.floor has a value (if (scene.floor) ... *scene.floor): the floor clearance of a capsule
    //       is min(a_y, b_y) - r_link - floor. Return the minimum (start from
    //       std::numeric_limits<double>::infinity()).
    throw std::logic_error("TODO: clearance");
}

bool config_free(const Eigen::Vector2d& q, const Scene& scene, double margin) {
    // TODO: false outside [-pi, pi] (tolerance 1e-12) on either joint; otherwise clearance(q) > margin.
    throw std::logic_error("TODO: config_free");
}

// ==========================================================================================
// Part B -- C-space grid and edge checking (MR Sec. 10.4)
// ==========================================================================================

Eigen::VectorXd grid_angles(int k) {
    // Given.
    Eigen::VectorXd th(k);
    const double d = 2 * PI / k;
    for (int i = 0; i < k; ++i) th(i) = -PI + i * d;
    return th;
}

Eigen::Vector2d cell_to_q(const Cell& cell, int k) {
    // Given.
    const Eigen::VectorXd th = grid_angles(k);
    return Eigen::Vector2d(th(cell.first), th(cell.second));
}

Cell q_to_cell(const Eigen::Vector2d& q, int k) {
    // TODO: Wrap q to [-pi, pi), divide by the grid step, round to the nearest integer (std::nearbyint, which
    //       rounds halves to even as np.rint), take it modulo k (py_imod). return {i1, i2};
    throw std::logic_error("TODO: q_to_cell");
}

BoolGrid cspace_grid(const Scene& scene, int k) {
    // TODO: Evaluate config_free at every grid configuration; free(i1, i2) (row = joint 1). BoolGrid free(k, k);
    throw std::logic_error("TODO: cspace_grid");
}

Eigen::Vector2d joint_step(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, bool wrap) {
    // TODO: qb - qa, wrapped to [-pi, pi) if wrap (the short way round).
    throw std::logic_error("TODO: joint_step");
}

bool edge_free(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, const Scene& scene, double step, bool wrap) {
    // TODO: d = joint_step(qa, qb); n = ceil(max|d| / step) (at least 1); test q = qa + s d for
    //       s = lin01(i, n), i = 0..n  (= linspace(0, 1, n + 1); wrap q to [-pi, pi) when wrap = true).
    throw std::logic_error("TODO: edge_free");
}

double motion_bound(const Eigen::Vector2d& dq, const Eigen::Vector2d& L) {
    // TODO: (L1 + L2)|dq1| + L2|dq2|.
    throw std::logic_error("TODO: motion_bound");
}

bool edge_free_certified(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, const Scene& scene, double step,
                         bool wrap) {
    // TODO: As edge_free, but each sample must have clearance >= motion_bound(d / n, scene.L) / 2 (and be inside
    //       the joint limits). Explain in your report why this is a guarantee.
    throw std::logic_error("TODO: edge_free_certified");
}

// ==========================================================================================
// Part C -- graph search (MR Algorithm 10.1, Sec. 10.2.4, Sec. 10.4)
// ==========================================================================================

SearchResult astar(int start, int goal, const Neighbors& neighbors, const Heuristic& h, double eta, bool reopen) {
    // TODO: Follow MR Algorithm 10.1. Use a std::priority_queue for OPEN with entries (past + eta*h, h, counter,
    //       node) (see the Entry type at the top of this file) and 'lazy deletion': skip a popped node that is
    //       already CLOSED or whose entry is out of date (f > past[node] + eta*h(node) + 1e-12).
    //       Goal test when a node is popped. Skip neighbours in CLOSED unless reopen = true; with reopen = true,
    //       remove an improved neighbour (tentative cost < old cost - 1e-12) from CLOSED and push it again.
    //       Count every expanded node. On failure: success = false, empty path,
    //       cost = std::numeric_limits<double>::infinity().
    throw std::logic_error("TODO: astar");
}

SearchResult dijkstra(int start, int goal, const Neighbors& neighbors) {
    // TODO: One line: A* with a zero heuristic (a lambda [](int) { return 0.0; }).
    throw std::logic_error("TODO: dijkstra");
}

std::vector<std::pair<Cell, double>> grid_neighbors(const Cell& cell, int k, int connectivity, bool wrap,
                                                    const BoolGrid& free) {
    // TODO: Steps (+-1, 0), (0, +-1) in this order and, for 8-connectivity, the four diagonals (1, 1), (1, -1),
    //       (-1, 1), (-1, -1) (cost sqrt(2) d); std::invalid_argument for another connectivity.
    //       wrap = true: indices modulo k (py_imod); wrap = false: drop neighbours outside 0..k-1.
    //       Skip non-free ones when free.size() > 0.
    throw std::logic_error("TODO: grid_neighbors");
}

double grid_heuristic(const Cell& cell, const Cell& goal, int k, const std::string& kind, bool wrap) {
    // TODO: n_i = |cell_i - goal_i| (with wrap: min(n_i, k - n_i)); then the formula in lab.hpp.
    //       std::invalid_argument for an unknown kind.
    throw std::logic_error("TODO: grid_heuristic");
}

GridSearchResult grid_astar(const BoolGrid& free, const Cell& start, const Cell& goal, int connectivity,
                            const std::string& heuristic, double eta, bool wrap, const EdgeCheck& edge_ok) {
    // TODO: Return failure (cost = infinity, expanded = 0) if start or goal is not free. Number the cells
    //       node = i * k + j (cell = {node / k, node % k}); build a Neighbors lambda from grid_neighbors (and
    //       edge_ok, if it is not empty: `if (!edge_ok || edge_ok(c, nb))`) and a Heuristic lambda from
    //       grid_heuristic, call astar and convert its path back to cells.
    throw std::logic_error("TODO: grid_astar");
}

// ==========================================================================================
// Part D -- the planner
// ==========================================================================================

Eigen::MatrixXd unwrap_path(const Eigen::MatrixXd& qs) {
    // TODO: Start with row 0 of qs; each next row = previous + wrap_angle(q - previous) (per joint).
    throw std::logic_error("TODO: unwrap_path");
}

double path_length(const Eigen::MatrixXd& qs) {
    // TODO: Sum of the norms of consecutive row differences.
    throw std::logic_error("TODO: path_length");
}

PlanResult plan_2r(const Scene& scene, int k, int connectivity, const std::string& heuristic, double eta,
                   bool certified) {
    // TODO: Follow steps 1-4 in lab.hpp; reuse cspace_grid, q_to_cell, cell_to_q, grid_astar and
    //       edge_free_certified / edge_free with step = (2 pi / k) / 4. Unwrap the path if scene.wrap.
    //       On failure return success = false, cost = infinity, an empty (0 x 2) q_path, and the grid in .free.
    throw std::logic_error("TODO: plan_2r");
}

}  // namespace lab
