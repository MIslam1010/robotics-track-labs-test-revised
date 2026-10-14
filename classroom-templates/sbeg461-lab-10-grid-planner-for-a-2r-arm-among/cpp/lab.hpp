// SBEG 461 Lab 10 -- Grid planner for a planar 2R arm among obstacles (MR Sec. 10.2-10.4), C++ track: declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Modern Robotics (MR): configuration q = (theta1, theta2), C-space C, free space C_free,
// C-obstacles C_obs (MR Sec. 10.2.1), distance d(q, B) (MR Sec. 10.2.2), A* search (MR Algorithm 10.1),
// Dijkstra = A* with zero heuristic (MR Sec. 10.2.4.2), grid methods (MR Sec. 10.4).
//
// Parts
//   A. Geometry and collision checking: FK of the 2R arm, capsule-circle and capsule-floor distances,
//      clearance d(q), config_free                                                    MR Sec. 10.2.1-10.2.2
//   B. C-space grid and edge checking: the k x k grid of free cells, a sampled edge check, and a
//      certified (conservative, "grown robot") edge check                              MR Sec. 10.2.2, 10.4
//   C. Graph search: a generic A* with optional node reopening (for inconsistent heuristics),
//      grid neighbours (4/8-connected, with or without wrap-around), heuristics       MR Algorithm 10.1, Sec. 10.4
//   D. The planner: snap start/goal to the grid, search with edge checks, return a continuous joint path
//
// Python -> C++:
//   scene dict                      -> lab::Scene (same field names; floor: std::optional<double>, empty = None)
//   (p0, p1, p2) of fk_2r           -> lab::ArmPoints {p0, p1, p2}
//   grid cell (i1, i2)              -> lab::Cell = std::pair<int, int>   (.first = i1 = joint 1)
//   k x k boolean array `free`      -> lab::BoolGrid = Eigen::Array<bool, Dynamic, Dynamic>   (free(i1, i2))
//   graph nodes of astar            -> int (any non-negative integer; a grid cell (i, j) is node i * k + j)
//   neighbors(node), h(node)        -> std::function (lab::Neighbors, lab::Heuristic); pass a lambda
//   result dict of astar            -> lab::SearchResult;  grid_astar -> lab::GridSearchResult; plan_2r -> lab::PlanResult
//   path None                       -> empty std::vector;  q_path None -> 0 x 2 matrix;  math.inf -> infinity
//   free=None, edge_ok=None         -> empty BoolGrid (default), empty std::function (default)
//   (N, 2) joint path               -> Eigen::MatrixXd with N rows (row = configuration)
//   ValueError                      -> std::invalid_argument
#ifndef SBEG461_LAB10_HPP
#define SBEG461_LAB10_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ==========================================================================================
// Provided (do not edit): the robot, the scenes and small grid helpers
// ==========================================================================================

constexpr double PI = 3.14159265358979323846;
constexpr double ONE_DEG = PI / 180.0;  // = math.radians(1.0)

// planar_2r of the shared course robot models (MR Sec. 4.1 / 8.1; MR Studio "Planar 2R arm")
inline const Eigen::Vector2d L_PLANAR_2R = Eigen::Vector2d(0.6, 0.4);  // link lengths (m)
/// Space screw axes as COLUMNS (6 x 2), as SLIST_PLANAR_2R in Python.
inline const Eigen::MatrixXd SLIST_PLANAR_2R =
    (Eigen::MatrixXd(6, 2) << 0, 0, 0, 0, 1, 1, 0, 0, 0, -0.6, 0, 0).finished();
inline const Eigen::Matrix4d M_PLANAR_2R =
    (Eigen::Matrix4d() << 1, 0, 0, 1.0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1).finished();

/// A scene of data/scenes.json (the Python dict returned by load_scene).
struct Scene {
    std::string name, description;
    Eigen::Vector2d L = Eigen::Vector2d(0.6, 0.4);   ///< link lengths (m)
    double link_radius = 0.0;                        ///< capsule radius of both links (m)
    Eigen::MatrixXd circles = Eigen::MatrixXd(0, 3);  ///< m x 3, one disc per row: cx, cy, r (m)
    std::vector<std::string> circle_names;
    std::optional<double> floor;                     ///< y of a floor line (half-plane y <= floor), or empty
    bool wrap = true;                                ///< true: continuous joints (torus); false: joint stops
    Eigen::Vector2d q_start = Eigen::Vector2d::Zero();  ///< rad
    Eigen::Vector2d q_goal = Eigen::Vector2d::Zero();   ///< rad
};

/// Return the scene `name` ("ward" or "eod") from data/scenes.json (given).
/// q_start, q_goal are converted from degrees to radians.
Scene load_scene(const std::string& name);

/// Map an angle to [-pi, pi) (given; the same arithmetic as Python's (a + pi) % (2 pi) - pi).
double wrap_angle(double a);
/// Element-wise wrap_angle of a joint vector (given).
Eigen::Vector2d wrap_angle(const Eigen::Vector2d& a);

// ==========================================================================================
// Part A -- geometry and collision checking (MR Sec. 10.2.1-10.2.2)
// ==========================================================================================

/// Base (origin), elbow and tip of the planar 2R arm.
struct ArmPoints {
    Eigen::Vector2d p0, p1, p2;
};

/// Joint and tip positions of the planar 2R arm.
/// Returns {p0, p1, p2}: base (origin), elbow and tip.
/// p1 = L1 (c1, s1), p2 = p1 + L2 (c12, s12)  (MR Sec. 4.1, Fig. 10.2).
ArmPoints fk_2r(const Eigen::Vector2d& q, const Eigen::Vector2d& L = L_PLANAR_2R);

/// Euclidean distance from point p to the segment [a, b] (all 2-vectors).
double point_segment_distance(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b);

/// Signed distance between a capsule (segment [a, b] grown by r_link) and a disc (centre c, radius r_obs):
/// > 0 no contact, = 0 contact, < 0 penetration (MR Sec. 10.2.2, sphere form d = ||r - b|| - R - B).
double capsule_circle_distance(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double r_link,
                               const Eigen::Vector2d& c, double r_obs);

/// d(q): the smallest signed distance between the two link capsules and every obstacle
/// (discs in scene.circles and, if scene.floor has a value, the half-plane y <= floor).
/// The base link (link 1) starts at the joint-1 axis; the robot base itself is not an obstacle.
double clearance(const Eigen::Vector2d& q, const Scene& scene);

/// True if q is inside the joint limits [-pi, pi] and clearance(q) > margin (MR: C_free).
bool config_free(const Eigen::Vector2d& q, const Scene& scene, double margin = 0.0);

// ==========================================================================================
// Part B -- C-space grid and edge checking (MR Sec. 10.4)
// ==========================================================================================

using Cell = std::pair<int, int>;                                   ///< grid cell (i1, i2)
using BoolGrid = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;  ///< free(i1, i2), row = joint 1

/// The k grid values of each joint: theta_i = -pi + i * (2 pi / k), i = 0..k-1 (given).
Eigen::VectorXd grid_angles(int k);

/// Grid cell (i1, i2) -> configuration q (rad) (given).
Eigen::Vector2d cell_to_q(const Cell& cell, int k);

/// Configuration q (rad) -> nearest grid cell (i1, i2) (with wrap-around).
Cell q_to_cell(const Eigen::Vector2d& q, int k);

/// Boolean k x k array `free(i1, i2)` = config_free(cell_to_q({i1, i2})). Row index = joint 1.
BoolGrid cspace_grid(const Scene& scene, int k);

/// Joint-space displacement from qa to qb; with wrap = true the shortest way round each joint circle.
Eigen::Vector2d joint_step(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, bool wrap = true);

/// Sampled check of the straight joint-space motion qa -> qb: every sample (spacing <= step in the
/// max-norm, both ends included) must be collision-free. Fast but NOT a guarantee between samples.
bool edge_free(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, const Scene& scene, double step = ONE_DEG,
               bool wrap = true);

/// Upper bound on how far ANY point of the 2R arm moves when the joints move straight by dq:
/// D = (L1 + L2)|dq1| + L2|dq2|  (a point at distance rho from joint 1 moves <= rho |dq1| + ...).
double motion_bound(const Eigen::Vector2d& dq, const Eigen::Vector2d& L = L_PLANAR_2R);

/// Conservative ("grown robot", MR p. 365) check of the straight motion qa -> qb: sample with spacing
/// <= step and require clearance >= D/2 at every sample, where D = motion_bound(sample spacing).
/// Every configuration between two samples is within D/2 of one of them, so true GUARANTEES the whole
/// motion is collision-free. false may be a false alarm (the check is conservative).
bool edge_free_certified(const Eigen::Vector2d& qa, const Eigen::Vector2d& qb, const Scene& scene,
                         double step = ONE_DEG, bool wrap = true);

// ==========================================================================================
// Part C -- graph search (MR Algorithm 10.1, Sec. 10.2.4, Sec. 10.4)
// ==========================================================================================

/// neighbors(node) -> list of (nbr, edge_cost > 0)
using Neighbors = std::function<std::vector<std::pair<int, double>>(int)>;
/// h(node) -> heuristic cost-to-go >= 0
using Heuristic = std::function<double(int)>;

/// Result of astar / dijkstra (the Python dict).
struct SearchResult {
    bool success = false;
    std::vector<int> path;           ///< start .. goal; empty if no path (Python: None)
    double cost = 0.0;               ///< path cost; infinity if no path
    int expanded = 0;                ///< nodes removed from OPEN and expanded (the goal counts when removed)
    std::map<int, int> parent;       ///< parent[start] = -1 (Python: None)
    std::map<int, double> past_cost;
};

/// A* search (MR Algorithm 10.1) on an implicit graph.
///
/// start, goal : nodes (non-negative integers);  neighbors(node) -> list of (nbr, edge_cost > 0);
/// h(node) -> heuristic cost-to-go >= 0;  est_total_cost = past_cost + eta * h  (eta > 1: MR's
/// "suboptimal A*");  reopen = false is MR Algorithm 10.1 exactly (neighbours in CLOSED are skipped);
/// reopen = true moves a CLOSED node back to OPEN when a cheaper path to it is found (needed for optimality
/// when h is admissible but not consistent).
/// The goal test is made when a node is removed from OPEN (MR line 6). Ties in OPEN are broken by the
/// smaller h, then first-in first-out.
///
/// Returns SearchResult{success, path [start..goal] or empty, cost, expanded = number of nodes removed from OPEN
/// and expanded (the goal is counted when it is removed), parent, past_cost}.
SearchResult astar(int start, int goal, const Neighbors& neighbors, const Heuristic& h, double eta = 1.0,
                   bool reopen = false);

/// Dijkstra's algorithm = A* with a zero heuristic (MR Sec. 10.2.4.2).
SearchResult dijkstra(int start, int goal, const Neighbors& neighbors);

/// Neighbours of `cell` on the k x k grid with their costs in rad (grid step d = 2 pi / k):
/// 4-connected: N, S, E, W with cost d;  8-connected: also the diagonals with cost sqrt(2) d (MR p. 372).
/// wrap = true: the C-space is a torus (cell k-1 is next to cell 0, MR p. 360); wrap = false: joint stops.
/// If `free` (k x k) is not empty, only free neighbours are returned. Returns a list of (cell, cost).
/// Throw std::invalid_argument if connectivity is not 4 or 8.
std::vector<std::pair<Cell, double>> grid_neighbors(const Cell& cell, int k, int connectivity = 8, bool wrap = true,
                                                    const BoolGrid& free = BoolGrid());

/// Heuristic cost-to-go (rad) on the grid. Per-axis cell differences n1, n2 (the shorter way round if wrap):
/// "zero" 0;  "manhattan" d (n1 + n2);  "euclidean" d sqrt(n1^2 + n2^2);
/// "octile" d (max + (sqrt 2 - 1) min)  (exact cost-to-go of an empty 8-connected grid).
/// Throw std::invalid_argument for any other kind.
double grid_heuristic(const Cell& cell, const Cell& goal, int k, const std::string& kind = "euclidean",
                      bool wrap = true);

/// edge_ok(cell_a, cell_b) -> bool: a step is used only if it is collision-free.
using EdgeCheck = std::function<bool(const Cell&, const Cell&)>;

/// Result of grid_astar: the SearchResult fields with cells instead of node numbers (parent and
/// past_cost are not returned).
struct GridSearchResult {
    bool success = false;
    std::vector<Cell> path;  ///< start .. goal; empty if no path
    double cost = 0.0;       ///< infinity if no path
    int expanded = 0;
};

/// A* on the C-space grid `free` (k x k) from cell `start` to cell `goal`.
/// edge_ok(cell_a, cell_b) -> bool, optional (empty = no edge check): a step is used only if it is
/// collision-free (MR p. 372). Returns the result of `astar` with the path as a list of cells.
GridSearchResult grid_astar(const BoolGrid& free, const Cell& start, const Cell& goal, int connectivity = 8,
                            const std::string& heuristic = "euclidean", double eta = 1.0, bool wrap = true,
                            const EdgeCheck& edge_ok = EdgeCheck());

// ==========================================================================================
// Part D -- the planner
// ==========================================================================================

/// Make a joint path (N x 2, row = configuration) continuous: replace each jump of about 2 pi
/// (crossing -pi/pi) by the short step.
Eigen::MatrixXd unwrap_path(const Eigen::MatrixXd& qs);

/// Sum of the Euclidean joint-space step lengths of a (continuous) path (N x 2) (rad).
double path_length(const Eigen::MatrixXd& qs);

/// Result of plan_2r (the Python dict).
struct PlanResult {
    bool success = false;
    Eigen::MatrixXd q_path = Eigen::MatrixXd(0, 2);  ///< N x 2, continuous; 0 x 2 if failed (Python: None)
    std::vector<Cell> cells;                        ///< grid cells of the path; empty if failed
    double cost = 0.0;                              ///< grid cost (rad); infinity if failed
    int expanded = 0;
    BoolGrid free;                                  ///< the C-space grid
};

/// Plan a collision-free joint path for the 2R arm of `scene` from scene.q_start to scene.q_goal.
/// 1. build the grid (cspace_grid); 2. snap start and goal to the nearest cells; the snap motions must be
/// collision-free; 3. grid_astar with an edge check (edge_free_certified if certified else edge_free,
/// both with step = grid step / 4); 4. return the continuous path q_start -> cells -> q_goal.
/// Returns PlanResult{success, q_path (N x 2, continuous, starts at q_start and ends at q_goal or an
/// equivalent angle), cells, cost (grid cost, rad), expanded, free}.
PlanResult plan_2r(const Scene& scene, int k = 72, int connectivity = 8, const std::string& heuristic = "euclidean",
                   double eta = 1.0, bool certified = true);

}  // namespace lab

#endif  // SBEG461_LAB10_HPP
