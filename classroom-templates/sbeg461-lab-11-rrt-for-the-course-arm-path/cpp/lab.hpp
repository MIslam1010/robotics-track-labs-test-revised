// SBEG 461 Lab 11 -- RRT for the course arm; path smoothing + time scaling (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided helpers (load_course_arm ... plan_and_scale, marked "given") are in helpers.cpp.
//
// Notation follows Modern Robotics (MR): configuration q in C (6 joint angles), free space C_free, signed
// distance d(q, B) to an obstacle (MR Sec. 10.2.2), RRT (MR Algorithm 10.3, Sec. 10.5.1), RRT* (MR p. 385),
// smoothing by "subdivide and reconnect" (MR Sec. 10.8), straight-line trapezoidal time scaling (MR Sec. 9.2.2,
// Lab 9).
//
// Parts
//   A. Collision model of the course arm: a 7-point skeleton moved by the PoE formula, capsules (segment +
//      radius) against spheres and planes, clearance d(q), config_free, edge_free          MR Sec. 10.2
//   B. Sampling-based planning: sampler with goal bias, nearest node, straight-line local planner (steer),
//      RRT and RRT*                                                                         MR Sec. 10.5.1
//   C. Smoothing and time scaling: path length, greedy and random shortcutting, rest-to-rest trapezoidal
//      time scaling of the via-point path, torque profile and limit check                  MR Sec. 10.8, 9.2
//
// Python -> C++:
//   arm dict of load_course_arm      -> lab::Arm (same field names)
//   scene dict of load_scene         -> lab::Scene (same field names)
//   np.random.Generator (rng)        -> lab::Rng& (given below; lab::Rng rng(seed) replaces np.random.default_rng(seed))
//   a configuration q (6,)           -> Eigen::VectorXd (6)
//   a batch Q (N, 6), a path (k, 6)  -> Eigen::MatrixXd with one ROW per configuration
//   link_points / clearance of a batch -> the separate functions link_points_batch / clearance_batch
//   (N, 7, 3) array of points        -> std::vector<Eigen::MatrixXd>, each 7 x 3 (row k = skeleton point k)
//   parent array (N,)                -> std::vector<int> (parent[0] = -1)
//   result dicts of rrt / rrt_star / plan_and_scale -> lab::RRTResult / lab::RRTStarResult / lab::PipelineResult
//   path None                        -> a 0 x 6 matrix (path.rows() == 0);  first_solution_nodes None -> -1
//   limits=None                      -> an empty matrix (the default argument): use arm.limits
//   tuple returns                    -> std::pair or the small structs below (field names as in Python)
#ifndef SBEG461_LAB11_HPP
#define SBEG461_LAB11_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types only)

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ==========================================================================================
// Types
// ==========================================================================================

/// The course arm (Python: the dict returned by load_course_arm).
struct Arm {
    Eigen::MatrixXd Slist;         ///< 6 x 6 space-frame screw axes (one per COLUMN)
    Eigen::Matrix4d M;             ///< home configuration of the tool frame
    mr::TransformList Mlist;       ///< 7 frames M_{i-1,i} (CoM frames, MR Sec. 8.3.2)
    mr::InertiaList Glist;         ///< 6 spatial inertias G_i
    Eigen::Vector3d g;             ///< gravity in {s} [m/s^2]
    Eigen::MatrixXd limits;        ///< joint position limits [rad], 6 x 2 (min, max)
    Eigen::VectorXd qdot_max;      ///< joint speed limits [rad/s] (6)
    Eigen::VectorXd qddot_max;     ///< joint acceleration limits [rad/s^2] (6)
    Eigen::VectorXd tau_max;       ///< joint torque limits [N m] (6)
    Eigen::VectorXd q_home;        ///< ready pose [rad] (6)
    Eigen::MatrixXd skeleton;      ///< 7 x 3: skeleton point k at home (in {s}), one per ROW
    std::vector<int> skeleton_link;  ///< 7: the link each skeleton point is fixed in (0 = base)
    Eigen::VectorXd link_radius;   ///< 5: capsule radius of segment k-(k+1), k = 1..5 [m]
    Eigen::VectorXd rho;           ///< 6: length of the skeleton beyond joint axis i [m] (motion_bound)
};

/// A scene of data/scenes.json (Python: the dict returned by load_scene).
struct Scene {
    std::string name, description;
    Eigen::MatrixXd spheres = Eigen::MatrixXd(0, 4);  ///< m x 4: cx, cy, cz, R (one sphere per row)
    Eigen::MatrixXd planes = Eigen::MatrixXd(0, 4);   ///< p x 4: nx, ny, nz, d (free side n.x >= d)
    Eigen::VectorXd q_start, q_goal;                  ///< rad (6)
    Eigen::MatrixXd sample_limits;                    ///< 6 x 2: the task sampling box of the planners
};

/// Random numbers for the planners (given; replaces np.random.Generator).
/// lab::Rng rng(seed) is the C++ counterpart of np.random.default_rng(seed). It is built on std::mt19937_64,
/// whose output is fixed by the C++ standard, so the same seed gives the same numbers with every compiler
/// (but NOT the same numbers as NumPy: the tests only check properties of the plans, as in Python).
class Rng {
public:
    explicit Rng(std::uint64_t seed = 0) : gen_(seed) {}
    /// Uniform in [0, 1)  (rng.random()).
    double random() { return static_cast<double>(gen_() >> 11) * (1.0 / 9007199254740992.0); }
    /// Uniform in [lo, hi)  (rng.uniform(lo, hi)).
    double uniform(double lo, double hi) { return lo + (hi - lo) * random(); }
    /// One uniform sample per component: [lo_i, hi_i)  (rng.uniform(lo, hi) with arrays lo, hi).
    Eigen::VectorXd uniform(const Eigen::VectorXd& lo, const Eigen::VectorXd& hi) {
        Eigen::VectorXd out(lo.size());
        for (Eigen::Index i = 0; i < lo.size(); ++i) out(i) = uniform(lo(i), hi(i));
        return out;
    }

private:
    std::mt19937_64 gen_;
};

/// Result of rrt (Python: dict(success, path, nodes, parent, iterations)).
struct RRTResult {
    bool success = false;
    Eigen::MatrixXd path = Eigen::MatrixXd(0, 6);  ///< k x 6, start .. goal; 0 x 6 if no path (Python: None)
    Eigen::MatrixXd nodes;                         ///< N x 6, the tree (row = node)
    std::vector<int> parent;                       ///< N, parent[0] = -1
    int iterations = 0;                            ///< number of samples drawn
};

/// Result of rrt_star (Python: dict(success, path, cost, nodes, parent, goal_index, first_solution_nodes,
/// best_cost_history)).
struct RRTStarResult {
    bool success = false;
    Eigen::MatrixXd path = Eigen::MatrixXd(0, 6);  ///< k x 6; 0 x 6 if no path
    Eigen::VectorXd cost;                          ///< N, cost-to-come of every node
    Eigen::MatrixXd nodes;                         ///< N x 6
    std::vector<int> parent;                       ///< N, parent[0] = -1
    int goal_index = -1;                           ///< index of the goal node, -1 if not reached
    int first_solution_nodes = -1;                 ///< tree size when the goal was first added (-1: None)
    std::vector<std::pair<int, double>> best_cost_history;  ///< (tree size, cost-to-goal) at each improvement
};

/// (T, t_a, v_peak) of a trapezoidal time scaling (Lab 9).
struct TrapezoidParams {
    double T = 0.0, t_a = 0.0, v_peak = 0.0;
};

/// (s, sdot, sddot) sampled at the times t (Lab 9).
struct TimeScaling {
    Eigen::VectorXd s, sd, sdd;
};

/// (v, a, T) of the fastest trapezoidal straight-line motion (Lab 9).
struct TrapezoidLimits {
    double v = 0.0, a = 0.0, T = 0.0;
};

/// (t, Q, Qd, Qdd, seg_T) of time_scale_path: t (K), Q/Qd/Qdd K x n (row = sample), seg_T (number of segments).
struct ScaledPath {
    Eigen::VectorXd t;
    Eigen::MatrixXd Q, Qd, Qdd;
    Eigen::VectorXd seg_T;
};

/// Result of check_limits (Python: dict(vel_ratio, tau_ratio, ok)).
struct LimitCheck {
    Eigen::VectorXd vel_ratio, tau_ratio;  ///< worst ratio per joint (n)
    bool ok = false;
};

/// Result of plan_and_scale (Python: the dict with every intermediate result). If success is false, only
/// `rrt` / `rrt_star` (the plan) is filled.
struct PipelineResult {
    bool success = false;
    std::string planner;       ///< "rrt" or "rrt_star": which of the two plan fields is used
    RRTResult rrt;             ///< the plan if planner == "rrt"
    RRTStarResult rrt_star;    ///< the plan if planner == "rrt_star"
    Eigen::MatrixXd path;      ///< the smoothed via-point path
    ScaledPath traj;           ///< t, Q, Qd, Qdd, seg_T
    Eigen::MatrixXd tau;       ///< K x 6 torques
    LimitCheck limits;
};

// ==========================================================================================
// Provided (given in helpers.cpp; do not edit): robot, scenes, PoE, Lab 9 time-scaling helpers, RNEA
// ==========================================================================================

/// The course arm (numbers identical to the course model in Course resources: Course robot models and
/// hardware safety checklist; data/course_arm.json).
/// payload = true uses a 0.125 kg payload at the tool point (the hardware-run limit: <= 50 % of the rated
/// 0.25 kg, safety checklist rule 4); derated = true applies the hardware-run derating (50 % of rated speed,
/// 60 % of rated torque). Inertial data, ratings and the acceleration limit (4 rad/s^2, as in Lab 9) are
/// ILLUSTRATIVE values.
Arm load_course_arm(bool payload = true, bool derated = true);

/// Scene `name` ("bench_or" or "eod_rig") from data/scenes.json: spheres (m x 4: cx, cy, cz, R), planes
/// (p x 4: nx, ny, nz, d, free side n.x >= d), q_start, q_goal (rad, 6), sample_limits (6 x 2: the task
/// sampling box used by the planners; inside the joint limits except joint 4, lower bound -3.2 < -2.88 rad,
/// so the joint-limit check still rejects some samples), description. Unknown name -> std::invalid_argument.
Scene load_scene(const std::string& name);

/// e^{[S] theta} for one screw axis S = (omega, v), ||omega|| = 1 (Rodrigues' formula, MR Eq. 3.51, and
/// MR Prop. 3.25 for the translation).  (Python: exp6_batch for an array of angles.)
Eigen::Matrix4d exp6(const sbeg::Vector6d& S, double theta);

/// Partial products of exponentials T_i(q) = e^{[S1] q1} ... e^{[Si] qi}, i = 0..6 (T_0 = I), for ONE
/// configuration q (6). A point fixed in link i at home, p_home, is at T_i(q) p_home (MR Eq. 4.14 applied to
/// a truncated chain). Returns 7 matrices.  (Python: poe_frames for a batch, shape (N, 7, 4, 4).)
std::vector<Eigen::Matrix4d> poe_frames(const Arm& arm, const Eigen::VectorXd& q);

/// Distance from the point c to the segment [a, b].
double segment_point_distance(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c);

/// True if every joint of q (6) is inside arm.limits (tolerance 1e-12).
bool within_limits(const Arm& arm, const Eigen::VectorXd& q);
/// within_limits for every row of Q (N x 6).
std::vector<bool> within_limits_batch(const Arm& arm, const Eigen::MatrixXd& Q);

/// Upper bound on how far any skeleton point moves for a straight joint motion dq:
/// D = sum_i rho_i |dq_i|, rho_i = length of the skeleton beyond joint axis i (Lab 10, motion_bound).
double motion_bound(const Arm& arm, const Eigen::VectorXd& dq);

/// (Lab 9) Total time, acceleration time and peak s-speed of a trapezoidal time scaling with s-units
/// v (1/s) and a (1/s^2): T = (a + v^2)/(v a) if v^2/a <= 1 (MR p. 334), else bang-bang T = 2/sqrt(a).
TrapezoidParams trapezoid_params(double v, double a);

/// (Lab 9) Trapezoidal time scaling (MR Eqs. 9.16-9.24); returns (s, sdot, sddot) at the times t
/// (t is clipped to [0, T]).
TimeScaling trapezoid_time_scaling(double v, double a, const Eigen::VectorXd& t);

/// (Lab 9) Fastest trapezoidal straight-line motion: the largest s-units v and a with |D_i| v <= qdot_lim_i and
/// |D_i| a <= qddot_lim_i for every moving joint (MR Eqs. 9.14-9.15). Returns (v, a, T).
TrapezoidLimits trapezoid_from_limits(const Eigen::VectorXd& theta_start, const Eigen::VectorXd& theta_end,
                                      const Eigen::VectorXd& qdot_lim, const Eigen::VectorXd& qddot_lim);

/// Inverse dynamics; uses mr::inverse_dynamics (MR Eqs. 8.50-8.54). Replace it by your own Lab 2 RNEA if you
/// like (same interface).
Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const sbeg::Vector6d& Ftip,
                     const mr::TransformList& Mlist, const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist);

/// (Lab 9) Worst |velocity|/limit and |torque|/limit per joint; ok = all ratios <= 1 + tol.
LimitCheck check_limits(const Arm& arm, const Eigen::MatrixXd& Qd, const Eigen::MatrixXd& Tau, double tol = 1e-9);

/// Convenience pipeline for the experiments: plan (planner "rrt" or "rrt_star", with Rng rng(seed) and the
/// scene's sample_limits), smooth (smooth = "none", "greedy", "random" or "greedy+random"), time-scale,
/// compute torques. Returns every intermediate result. (Python: **kw -> the default planner arguments.)
PipelineResult plan_and_scale(const Arm& arm, const Scene& scene, unsigned seed = 0,
                              const std::string& planner = "rrt", const std::string& smooth = "greedy");

/// Write CSV files for plot_results.py (given): <prefix>_tree.csv (node, parent, q1..q6),
/// <prefix>_path.csv (raw planned path), <prefix>_smooth.csv (smoothed path), <prefix>_traj.csv
/// (t, q1..q6, qd1..qd6, tau1..tau6), <prefix>_tool.csv (t, x, y, z of the tool point). Needs your link_points.
void export_results(const std::string& prefix, const Arm& arm, const PipelineResult& out);

// ==========================================================================================
// Part A -- collision model and clearance (MR Sec. 10.2.1-10.2.2)
// ==========================================================================================

/// Skeleton points of the course arm at configuration q (6). Returns a 7 x 3 matrix (row k = point k).
///
/// arm.skeleton.row(k) is point k at home (in {s}); it is fixed in link arm.skeleton_link[k], so it is at
/// T_link(q) p_k (use poe_frames). Point 6 is the tool point: it equals the position part of
/// FKinSpace(M, Slist, q).
Eigen::MatrixXd link_points(const Arm& arm, const Eigen::VectorXd& q);

/// link_points for every row of Q (N x 6): returns N matrices of size 7 x 3.
std::vector<Eigen::MatrixXd> link_points_batch(const Arm& arm, const Eigen::MatrixXd& Q);

/// Signed clearance d(q) (m): the smallest signed distance between the 5 link capsules (segments k-k+1,
/// k = 1..5, radius arm.link_radius(k-1)) and every obstacle:
///     sphere (c, R):     dist(c, segment) - r_link - R                      (MR Sec. 10.2.2)
///     plane (n, d):      min(n.a, n.b) - d - r_link   (free side n.x >= d)
/// > 0 free, < 0 penetration. The base column (points 0-1) does not move and is not checked.
/// (No obstacles at all: +infinity.)
double clearance(const Arm& arm, const Scene& scene, const Eigen::VectorXd& q);

/// clearance for every row of Q (N x 6): returns a vector of N clearances.
Eigen::VectorXd clearance_batch(const Arm& arm, const Scene& scene, const Eigen::MatrixXd& Q);

/// q in C_free: inside the joint limits and clearance(q) > margin.
bool config_free(const Arm& arm, const Scene& scene, const Eigen::VectorXd& q, double margin = 0.0);

/// Straight joint-space motion qa -> qb (MR Sec. 10.5.1.3, straight-line local planner) is free.
///
/// Sample n + 1 configurations (both ends included) with n = max(1, ceil(max_i |qb_i - qa_i| / step));
/// every sample must be within limits and have clearance > margin. This sampled check is fast but not a
/// guarantee between samples (MR p. 397). certified = true requires clearance > margin + D/2 with
/// D = motion_bound(arm, (qb - qa)/n): then the whole motion is guaranteed free (grown robot, MR p. 365).
/// Evaluate all samples in one call of clearance_batch().
bool edge_free(const Arm& arm, const Scene& scene, const Eigen::VectorXd& qa, const Eigen::VectorXd& qb,
               double step = 0.02, double margin = 0.0, bool certified = false);

// ==========================================================================================
// Part B -- RRT and RRT* (MR Sec. 10.5.1, Algorithm 10.3)
// ==========================================================================================

/// MR Algorithm 10.3 line 3: with probability p_goal return q_goal (goal bias), otherwise a sample from the
/// uniform distribution over the box `limits` (6 x 2). Draw u = rng.random() first; if u < p_goal return a
/// copy of q_goal, else return rng.uniform(limits.col(0), limits.col(1)).
Eigen::VectorXd sample_config(Rng& rng, const Eigen::MatrixXd& limits, const Eigen::VectorXd& q_goal,
                              double p_goal);

/// MR Algorithm 10.3 line 4: index of the row of nodes (N x n) closest to q in Euclidean joint distance
/// (ties: smallest index).
int nearest(const Eigen::MatrixXd& nodes, const Eigen::VectorXd& q);

/// MR Algorithm 10.3 line 5 (straight-line local planner): the point a distance d from q_from on the
/// straight line towards q_to, or q_to itself if it is closer than d.
Eigen::VectorXd steer(const Eigen::VectorXd& q_from, const Eigen::VectorXd& q_to, double d);

/// Follow parent pointers from node i back to the root (node 0, parent -1); return the path root -> i
/// (k x n, one row per node).
Eigen::MatrixXd tree_path(const Eigen::MatrixXd& nodes, const std::vector<int>& parent, int i);

/// Sum of Euclidean joint-space lengths of the straight segments of a via-point path (k x n).
double path_length(const Eigen::MatrixXd& path);

/// Basic RRT, MR Algorithm 10.3, for a kinematic problem in joint space.
///
/// Each iteration: q_samp = sample_config; i = nearest(tree, q_samp); q_new = steer(tree[i], q_samp, d);
/// if the straight motion tree[i] -> q_new is free (edge_free with `step`), add q_new with parent i.
/// Goal set: if the new node is within d of q_goal and the motion q_new -> q_goal is free, add q_goal
/// itself as a node (parent = new node) and return SUCCESS. Skip q_new if it coincides with its parent.
/// Stop with FAILURE when the tree has max_nodes nodes. `limits` (6 x 2) is the sampling box (empty =
/// arm.limits); the collision checks always use the true joint limits arm.limits.
///
/// Returns RRTResult{success, path (k x 6, or 0 x 6), nodes (N x 6), parent (N) with parent[0] = -1,
/// iterations = number of samples drawn}.
RRTResult rrt(const Arm& arm, const Scene& scene, const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_goal,
              Rng& rng, double d = 0.3, double p_goal = 0.1, int max_nodes = 3000, double step = 0.02,
              const Eigen::MatrixXd& limits = Eigen::MatrixXd());

/// RRT* (MR Sec. 10.5.1.4, p. 385) with path length as cost and a fixed neighbourhood radius r_near.
///
/// Like rrt(), but (1) the new node q_new chooses as parent the node x in X_near = {x : ||x - q_new|| <=
/// r_near} (plus the nearest node) with a free motion x -> q_new that minimises cost(x) + ||x - q_new||;
/// (2) every x in X_near that can be reached more cheaply through q_new (free motion) is re-wired to
/// q_new, and the costs of its descendants are updated; (3) the search does NOT stop at the first
/// solution: the goal is added as a node once (when a new node is within d of it with a free motion) and
/// the loop runs until the tree has max_nodes nodes; the goal's cost keeps improving by rewiring.
///
/// Returns RRTStarResult{success, path, cost (N), nodes, parent, goal_index, first_solution_nodes,
/// best_cost_history (list of (tree size, cost-to-goal) each time it improves)}. `limits` as in rrt().
RRTStarResult rrt_star(const Arm& arm, const Scene& scene, const Eigen::VectorXd& q_start,
                       const Eigen::VectorXd& q_goal, Rng& rng, double d = 0.3, double p_goal = 0.1,
                       int max_nodes = 1000, double r_near = 0.6, double step = 0.02,
                       const Eigen::MatrixXd& limits = Eigen::MatrixXd());

// ==========================================================================================
// Part C -- smoothing (MR Sec. 10.8) and time scaling (MR Sec. 9.2, Lab 9)
// ==========================================================================================

/// Deterministic "subdivide and reconnect" (MR Sec. 10.8): from via point i, connect straight to the
/// FARTHEST later via point j whose straight motion is free (try j = last, last-1, ..., i+1); keep j and
/// repeat from j. Returns the shorter via-point path (same first and last points).
Eigen::MatrixXd shortcut_greedy(const Arm& arm, const Scene& scene, const Eigen::MatrixXd& path,
                                double step = 0.02);

/// Point at arc length s along a via-point path: returns (segment index k, point) with the point on
/// segment path[k] -> path[k+1].
std::pair<int, Eigen::VectorXd> point_on_path(const Eigen::MatrixXd& path, double s);

/// Randomised test-and-replace smoothing (MR Sec. 10.8): n_iter times, draw two arc-length values
/// s1 < s2 uniformly on [0, length] (two calls rng.uniform(0, length), sorted); take the points p1, p2 at
/// s1, s2 (point_on_path); if the straight motion p1 -> p2 is free, replace the path between them by
/// that segment, i.e. new path = path[:k1+1] + [p1, p2] + path[k2+1:]. Skip draws with k1 == k2.
Eigen::MatrixXd shortcut_random(const Arm& arm, const Scene& scene, const Eigen::MatrixXd& path, Rng& rng,
                                int n_iter = 100, double step = 0.02);

/// Rest-to-rest time scaling of a via-point path (MR Sec. 9.2 applied segment by segment): each
/// straight segment q_k -> q_{k+1} gets the fastest trapezoidal time scaling allowed by the joint speed
/// and acceleration limits (trapezoid_from_limits), and the robot stops at every via point.
///
/// Sample every segment at t = 0, dt, 2 dt, ... plus its end time; concatenate the segments without
/// repeating the junction points (time is cumulative). Returns ScaledPath{t (K), Q (K x n), Qd (K x n),
/// Qdd (K x n), seg_T (number of segments)}, with Qd = sdot (q_{k+1} - q_k), Qdd = sddot (q_{k+1} - q_k).
ScaledPath time_scale_path(const Eigen::MatrixXd& path, const Eigen::VectorXd& qdot_max,
                           const Eigen::VectorXd& qddot_max, double dt = 0.01);

/// tau_k = M(q_k) qdd_k + c(q_k, qd_k) + g(q_k) along a sampled trajectory (RNEA; Lab 9). Returns K x n.
Eigen::MatrixXd torque_profile(const Arm& arm, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& Qd,
                               const Eigen::MatrixXd& Qdd);

}  // namespace lab

#endif  // SBEG461_LAB11_HPP
