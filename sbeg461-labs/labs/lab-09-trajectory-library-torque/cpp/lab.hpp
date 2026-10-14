// SBEG 461 Lab 9 -- Trajectory library, torque feasibility and time-optimal time scaling (TOPP) on the course arm
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp (14 functions). Do not change these declarations: test_lab.cpp calls them.
// The provided helpers (marked "Provided" or "given") are in helpers.cpp; toppralite (toppralite.hpp) stands in for
// the Python library toppra.
//
// Notation follows Modern Robotics (MR) Ch. 9: a trajectory theta(t) = theta(s(t)) is a path theta(s),
// s in [0, 1], composed with a time scaling s(t), t in [0, T] (MR Sec. 9.1, Eqs. 9.1-9.2); theta' = d theta/ds,
// theta'' = d^2 theta/ds^2. Dynamics in MR list form (MR Sec. 8.3.2): Mlist, Glist, Slist, gravity g.
// Along a path (MR Eqs. 9.32-9.33):  m(s) sdd + c(s) sd^2 + g(s) = tau, and the actuator limits become
// L(s, sd) <= sdd <= U(s, sd) (MR Eqs. 9.36-9.37).
//
// Parts
//   A. Point-to-point time scalings: cubic, quintic, trapezoid       MR Sec. 9.2.2, Eqs. 9.9-9.24
//      + straight-line joint trajectories                             MR Eqs. 9.3-9.5
//   B. Via points: cubic via-point interpolation                      MR Sec. 9.3, Eqs. 9.25-9.29
//      + uniform cubic B-spline                                       MR p. 338; Gotor Sec. 7.1.4
//   C. Torque feasibility and uniform time scaling                    MR Eqs. 8.50-8.54 (RNEA), 9.30-9.31; Sec. 9.7
//   D. Time-optimal time scaling: path dynamics, motion cone,         MR Sec. 9.4, Eqs. 9.32-9.37
//      velocity limit curve, TOPP (toppralite), TOPP vs quintic
//   E. A command file for the course arm (provided)                   course-arm hardware safety checklist
//
// Python -> C++: an array of times t -> Eigen::VectorXd (a scalar t has its own overload, given);
// an (N, n) array -> Eigen::MatrixXd with one ROW per sample; the tuples (s, sdot, sddot), (T, t_a, v_peak),
// (v, a, T), (t, Q, Qd, Qdd), (B, Bd, Bdd), (p, dp, ddp), (m, c, g) and the result dicts -> the small structs below
// (field names as in Python, so `auto [s, sd, sdd] = lab::cubic_time_scaling(T, t);` works); every model dict
// (course arm, MR 2R, elbow exoskeleton, gantry) -> lab::Robot; a path object / trajectory callable -> lab::Path /
// lab::TrajectoryFn (std::function, called as path(s, order) / traj(t, order), one s or t at a time);
// method = 3, 5, "trap" -> lab::CUBIC, lab::QUINTIC, lab::TRAP; an optional argument `x=None` -> an EMPTY
// Eigen::VectorXd (the default); ValueError -> std::invalid_argument; RuntimeError -> std::runtime_error.
#ifndef SBEG461_LAB9_HPP
#define SBEG461_LAB9_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types only)

#include "toppralite.hpp"

#include <string>
#include <utility>
#include <vector>

namespace lab {

constexpr double EPS_M = 1e-9;  ///< |m_i(s)| below this is treated as a zero-inertia point (MR Sec. 9.4.4)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A robot model (Python: the model dicts). Every model fills name, n, Slist, Mlist, Glist and g; the course arm
/// also fills M, qdot_max, tau_max, limits and q_home.
struct Robot {
    std::string name;
    int n = 0;
    Eigen::MatrixXd Slist;                            ///< 6 x n space-frame screw axes (one per COLUMN)
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();  ///< home configuration of the tool frame
    mr::TransformList Mlist;                          ///< n + 1 frames M_{i-1,i} (CoM frames, MR Sec. 8.3.2)
    mr::InertiaList Glist;                            ///< n spatial inertias G_i (angular block first, MR Eq. 8.32)
    Eigen::Vector3d g = Eigen::Vector3d::Zero();      ///< gravity in {s} [m/s^2]
    Eigen::VectorXd qdot_max;                         ///< joint speed limits [rad/s] (n)
    Eigen::VectorXd tau_max;                          ///< joint torque limits [N m] (n)
    Eigen::MatrixXd limits;                           ///< joint position limits [rad], n x 2 (min, max)
    Eigen::VectorXd q_home;                           ///< ready pose [rad] (n)
};

/// The task configurations of data/tasks.json (Python: the dict returned by load_tasks).
struct Tasks {
    Eigen::VectorXd A, B, C, D, W;  ///< task configurations [rad] (6 each)
    Eigen::VectorXd via_times;      ///< via times of A -> W -> B [s]
    Eigen::VectorXd qddot_lim;      ///< joint acceleration limits [rad/s^2] (6)
    double payload_kg = 0.0;
    std::string description;
    /// tasks[name] for name = "A", "B", "C", "D" or "W" (given); other names -> std::invalid_argument.
    const Eigen::VectorXd& config(const std::string& name) const;
};

/// The task of Parts D-E, data/topp_task.json (Python: the dict returned by load_topp_task).
struct ToppTask {
    std::string description;
    Eigen::MatrixXd waypoints;  ///< K x n, one waypoint per row
    double payload_kg = 0, torque_fraction = 0, sim_speed_fraction = 0, hw_speed_fraction = 0,
           first_run_speed_fraction = 0, hw_qddot_max = 0, servo_period_s = 0, joint_margin_deg = 0;
    int grid_N = 0;
};

/// (s, sdot, sddot) sampled at the times t (Python: a tuple of three arrays).
struct TimeScaling {
    Eigen::VectorXd s, sd, sdd;
};

/// (s, sdot, sddot) at one time (Python: a tuple of three scalars).
struct TimeScalingPoint {
    double s = 0.0, sd = 0.0, sdd = 0.0;
};

/// (T, t_a, v_peak) of a trapezoidal time scaling.
struct TrapezoidParams {
    double T = 0.0, t_a = 0.0, v_peak = 0.0;
};

/// (v, a, T) of the fastest trapezoidal straight-line motion.
struct TrapezoidLimits {
    double v = 0.0, a = 0.0, T = 0.0;
};

/// (t, Q, Qd, Qdd): t (N), Q/Qd/Qdd N x n (row k = sample k).
struct Trajectory {
    Eigen::VectorXd t;
    Eigen::MatrixXd Q, Qd, Qdd;
};

/// (B, Bd, Bdd) of a via-point trajectory, each len(t) x m.
struct ViaSamples {
    Eigen::MatrixXd B, Bd, Bdd;
};

/// (p, dp/du, d2p/du2) of a B-spline at one parameter value, each (m).
struct BSplinePoint {
    Eigen::VectorXd p, dp, ddp;
};

/// Result of check_limits (Python: dict(vel_ratio, tau_ratio, ok)).
struct LimitCheck {
    Eigen::VectorXd vel_ratio, tau_ratio;  ///< worst ratio per joint (n)
    bool ok = false;
};

/// Result of min_time_scale (Python: dict(c_vel, c_acc, c_tau, c, binding)).
struct TimeScaleResult {
    double c_vel = 0.0, c_acc = 0.0, c_tau = 0.0, c = 0.0;
    std::pair<std::string, int> binding;  ///< ("velocity", "acceleration" or "torque", joint index 0-based)
};

/// Time-scaling methods of joint_trajectory (Python: method = 3, 5, "trap").
constexpr int CUBIC = 3;
constexpr int QUINTIC = 5;
constexpr int TRAP = 0;

using Path = toppralite::Path;                  ///< path(s, 0) = theta, path(s, 1) = theta', path(s, 2) = theta''
using TrajectoryFn = toppralite::Trajectory;    ///< traj(t, 0) = q, traj(t, 1) = qdot, traj(t, 2) = qddot

/// (m, c, g) of MR Eq. 9.33, each of size n.
struct PathDynamics {
    Eigen::VectorXd m, c, g;
};

/// The result of topp (the Python dict): T, traj(t, order), s_grid and sdot_grid (the phase-plane curve).
struct ToppResult {
    double T = 0;
    TrajectoryFn traj;
    Eigen::VectorXd s_grid, sdot_grid;
};

/// The result of peak_usage (the Python dict).
struct PeakUsage {
    Eigen::VectorXd vel_ratio, tau_ratio;
    double jerk_max = 0;
    Eigen::MatrixXd tau;  ///< (N+1) x n, row = sample
    Eigen::VectorXd t;
};

/// The result of check_command (the Python dict).
struct CommandCheck {
    bool ok = false, within_limits = false;
    double peak_speed_ratio = 0;
    bool speed_ok = false, starts_at_rest = false, ends_at_rest = false;
};

// ---------------------------------------------------------------------------
// Provided helpers (given in helpers.cpp; do not edit)
// ---------------------------------------------------------------------------

/// [w] for a 3-vector (MR Eq. 3.30).
Eigen::Matrix3d skew(const Eigen::Vector3d& w);
/// T^{-1} for T in SE(3).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);
/// [Ad_T] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);
/// [ad_V] (MR Eq. 8.38).
sbeg::Matrix6d ad(const sbeg::Vector6d& V);
/// e^{[S] theta} for a screw axis with ||omega|| = 1 or (omega = 0, ||v|| = 1) (MR Prop. 3.25).
Eigen::Matrix4d matrix_exp6(const sbeg::Vector6d& S, double theta);

/// Recursive Newton-Euler inverse dynamics (MR Eqs. 8.50-8.54), same interface as mr::inverse_dynamics.
/// This is the Lab 2/Lab 4 function; you may paste your own.
Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const sbeg::Vector6d& Ftip,
                     const mr::TransformList& Mlist, const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist);

/// The course arm (from the course robot models in Course resources, numbers identical; data/course_arm.json).
/// payload = true uses the 0.25 kg maximum payload at the tool point. derated = true applies the derating of
/// Parts A-C (50 % of rated speed, 60 % of rated torque). All inertial data and ratings are ILLUSTRATIVE values.
Robot load_course_arm(bool payload = true, bool derated = true);

/// The course arm with a point-mass payload of payload_kg at the tool point and its RATED limits (Part D applies
/// its own limit fractions): G_6 += [Ad_T]^T diag(0, 0, 0, m, m, m) [Ad_T], T = (M_{6,tool})^{-1} (MR Eq. 8.42).
Robot course_arm_model(double payload_kg = 0.0);

/// Task configurations A, B, C, D, W (rad) and via times (s) of Parts A-C (data/tasks.json; illustrative).
Tasks load_tasks();

/// Task data of Parts D-E (data/topp_task.json: waypoints, limit fractions, payload, servo period).
ToppTask load_topp_task(const std::string& name = "topp_task.json");

/// Path of a file in the lab's data folder (Python: DATA / name).
std::string data_path(const std::string& name);

/// Write t, q1..qn, qd1..qdn (SI units) to a CSV file.
void export_csv(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& Q,
                const Eigen::MatrixXd& Qd);

/// Write a command file in the project log format (project starter `hardware.py`): t, cmd_1..cmd_n.
void write_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd);

// ---------------------------------------------------------------------------
// Part A -- point-to-point time scalings (MR Sec. 9.2.2)
// ---------------------------------------------------------------------------

/// Cubic rest-to-rest time scaling s = 3(t/T)^2 - 2(t/T)^3 (MR Eq. 9.9 with a2 = 3/T^2, a3 = -2/T^3).
///
/// Every t is clipped to [0, T]. Returns (s, sdot, sddot), each with the length of t.
/// At t = 0 and t = T return the one-sided values (sddot = +-6/T^2).
TimeScaling cubic_time_scaling(double T, const Eigen::VectorXd& t);
/// Scalar t (given: calls the vector version).
TimeScalingPoint cubic_time_scaling(double T, double t);

/// Quintic rest-to-rest time scaling s = 10 r^3 - 15 r^4 + 6 r^5, r = t/T, with zero end velocities
/// and accelerations (MR Sec. 9.2.2.1, Exercise 9.5; Fig. 9.4). Returns (s, sdot, sddot).
TimeScaling quintic_time_scaling(double T, const Eigen::VectorXd& t);
/// Scalar t (given: calls the vector version).
TimeScalingPoint quintic_time_scaling(double T, double t);

/// Total time and acceleration time of a trapezoidal time scaling with s-units v (1/s) and a (1/s^2).
///
/// Three-stage profile if v^2/a <= 1:  T = (a + v^2)/(v a), t_a = v/a  (MR p. 334).
/// If v^2/a > 1 the cruise speed is never reached (MR Exercise 9.10): bang-bang profile with
/// t_a = T/2 = 1/sqrt(a) and peak speed sqrt(a). Returns (T, t_a, v_peak).
TrapezoidParams trapezoid_params(double v, double a);

/// Trapezoidal time scaling (MR Eqs. 9.16-9.24); bang-bang (triangle) if v^2/a > 1.
/// Returns (s, sdot, sddot) at the times t (clipped to [0, T]).
TimeScaling trapezoid_time_scaling(double v, double a, const Eigen::VectorXd& t);
/// Scalar t (given: calls the vector version).
TimeScalingPoint trapezoid_time_scaling(double v, double a, double t);

/// Provided. Fastest trapezoidal straight-line motion (MR Eqs. 9.14-9.15): the largest v and a (s-units) with
/// |(theta_end - theta_start) v| <= qdot_lim and |(theta_end - theta_start) a| <= qddot_lim, joint by
/// joint (joints that do not move impose no limit). The limits are vectors (one value per joint).
/// Returns (v, a, T); T uses your trapezoid_params.
TrapezoidLimits trapezoid_from_limits(const Eigen::VectorXd& theta_start, const Eigen::VectorXd& theta_end,
                                      const Eigen::VectorXd& qdot_lim, const Eigen::VectorXd& qddot_lim);

/// Straight line in joint space theta(s) = theta_start + s (theta_end - theta_start) (MR Eq. 9.3),
/// time-scaled with method = CUBIC (3), QUINTIC (5) or TRAP (trapezoid; needs v and a and ignores T).
///
/// Returns (t, Q, Qd, Qdd): t (N), Q/Qd/Qdd N x n, with Qd = sdot D and Qdd = sddot D (MR Eqs. 9.4-9.5;
/// the path is straight so d2theta/ds2 = 0). The rows of Q equal mr::joint_trajectory (modern_robotics).
Trajectory joint_trajectory(const Eigen::VectorXd& theta_start, const Eigen::VectorXd& theta_end, double T, int N,
                            int method = QUINTIC, double v = 0.0, double a = 0.0);

// ---------------------------------------------------------------------------
// Part B -- via points (MR Sec. 9.3) and B-splines
// ---------------------------------------------------------------------------

/// Cubic via-point interpolation of ONE coordinate (MR Eqs. 9.26-9.29).
///
/// beta, beta_dot: (k) via positions and velocities; times: (k) increasing via times T_1 = 0 .. T_k.
/// Returns a (k-1) x 4 matrix whose row j is (a_j0, a_j1, a_j2, a_j3) for segment j.
Eigen::MatrixXd via_cubic_coefficients(const Eigen::VectorXd& beta, const Eigen::VectorXd& beta_dot,
                                       const Eigen::VectorXd& times);

/// Provided. Via velocities from positions and times (MR p. 337: "a heuristic could be used").
/// Rule used in this course (Lecture A), per coordinate: zero at the first and last via; at an
/// interior via, the average of the slopes of the two adjacent segments if both slopes have the same
/// strict sign, otherwise zero (the via is a turning point, or one side is flat).
/// points: k x m (one via per row; one coordinate = a k x 1 matrix); returns k x m.
Eigen::MatrixXd heuristic_via_velocities(const Eigen::MatrixXd& points, const Eigen::VectorXd& times);

/// Evaluate the cubic via-point trajectory of all m coordinates at the times t (clipped to [T_1, T_k]).
/// points, velocities: k x m; times: (k). Returns (B, Bd, Bdd), each len(t) x m.
ViaSamples via_trajectory(const Eigen::MatrixXd& points, const Eigen::MatrixXd& velocities,
                          const Eigen::VectorXd& times, const Eigen::VectorXd& t);

/// Provided. The four uniform cubic B-spline blending functions at local parameter u in [0, 1]:
/// b0 = (1-u)^3/6, b1 = (3u^3 - 6u^2 + 4)/6, b2 = (-3u^3 + 3u^2 + 3u + 1)/6, b3 = u^3/6.
Eigen::Vector4d bspline_basis(double u);
/// Provided. Vector u: returns 4 x len(u) (column k = basis at u(k)).
Eigen::MatrixXd bspline_basis(const Eigen::VectorXd& u);

/// Uniform cubic B-spline with control points ctrl (n x m, one point per row), n >= 4, at global parameter
/// u in [0, n-3] (segment i = floor(u) uses P_i..P_{i+3}, local parameter w = u - i; clamp i to n - 4).
/// p = sum_k b_k(w) P_{i+k} with b = bspline_basis(w); dp/du and d2p/du2 use the derivatives of the b_k.
/// Returns (p, dp/du, d2p/du2), each (m). The curve is C2 and lies in the convex hull of the
/// control points (MR p. 338).
BSplinePoint uniform_cubic_bspline(const Eigen::MatrixXd& ctrl, double u);

// ---------------------------------------------------------------------------
// Part C -- torque feasibility (MR Eqs. 9.30-9.31) and uniform time scaling
// ---------------------------------------------------------------------------

/// Joint torques tau_k = M(q_k) qdd_k + c(q_k, qd_k) + g(q_k) along a sampled trajectory, by the RNEA.
/// robot: any model (course arm from load_course_arm, ...); Q, Qd, Qdd: N x n. Returns N x n.
Eigen::MatrixXd torque_profile(const Robot& robot, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& Qd,
                               const Eigen::MatrixXd& Qdd);

/// Provided. Worst |velocity|/limit and |torque|/limit per joint along a trajectory.
/// Returns (vel_ratio (n), tau_ratio (n), ok); ok if every ratio <= 1 + tol.
LimitCheck check_limits(const Robot& robot, const Eigen::MatrixXd& Qd, const Eigen::MatrixXd& Tau,
                        double tol = 1e-9);

/// Smallest uniform stretch factor c > 0 that makes a sampled trajectory feasible.
///
/// Stretching time by c (theta_c(t) = theta(t/c)) divides velocities by c and accelerations by c^2, and
/// because c(theta, thetadot) is quadratic in thetadot (MR Sec. 9.4, c = thetadot^T Gamma thetadot)
///     tau_c = g(theta) + d / c^2,   d = tau(Q, Qd, Qdd) - g(theta)     (MR Sec. 9.7, Hollerbach 1984).
/// Velocity limits give c >= max |qd| / qdot_max (robot.qdot_max); if qddot_max is given (non-empty),
/// acceleration limits give c >= sqrt(max |qdd| / qddot_max). Torque limits (robot.tau_max, symmetric) give,
/// sample by sample and joint by joint,
///     d > 0:  c^2 >= d / (tau_max - g),      d < 0:  c^2 >= -d / (tau_max + g),
/// which needs |g| < tau_max (otherwise the robot cannot even hold the configuration: throw
/// std::invalid_argument).
///
/// Returns (c_vel, c_acc, c_tau, c = the largest of the three, binding = ("velocity", "acceleration" or "torque",
/// joint index 0-based)); c_acc = 0 if qddot_max is empty. Ties go to velocity, then acceleration.
/// For a rest-to-rest trajectory computed with T = 1, c is the minimum feasible duration in seconds.
TimeScaleResult min_time_scale(const Robot& robot, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& Qd,
                               const Eigen::MatrixXd& Qdd, const Eigen::VectorXd& qddot_max = Eigen::VectorXd());

// ---------------------------------------------------------------------------
// Part D -- time-optimal time scaling (MR Sec. 9.4)
// ---------------------------------------------------------------------------

/// Provided. theta(s) = q0 + s (q1 - q0), s in [0, 1] (MR Eq. 9.3). (Python: StraightLinePath(q0, q1).)
Path straight_line_path(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1);

/// Provided. MR Fig. 9.9: x = xc + R cos(pi s), y = yc - R sin(pi s), s in [0, 1].  (Python: SemicirclePath.)
Path semicircle_path(double R = 0.3, double xc = 0.0, double yc = 0.0);

/// Provided. C2 cubic-spline path through the joint-space waypoints (K x n, one waypoint per ROW), s_k = k/(K-1),
/// with theta'' = 0 at the ends ("natural" spline, as toppra.SplineInterpolator(bc_type="natural")).
Path waypoint_path(const Eigen::MatrixXd& waypoints);

/// Provided. The MR 2R arm (MR Fig. 8.1, Eqs. 8.9-8.10): point masses at the link ends, gravity g along -y.
Robot mr_2r_model(double L1 = 1.0, double L2 = 1.0, double m1 = 1.0, double m2 = 1.0, double g = 0.0);

/// Provided. Elbow exoskeleton + forearm + 1 kg bottle, vertical plane, theta = 0 hanging down. Lumped link:
/// m = 3.0 kg, sum(m r) = 0.698 kg m, inertia about the elbow 0.19734 kg m^2 (illustrative values).
Robot elbow_exo_model();

/// Provided. X-Y gantry (two prismatic joints, horizontal plane) of a humanitarian-demining sensor head.
/// Joint 1 (x) moves Mx in total; joint 2 (y) moves my. Illustrative values.
Robot gantry_model(double Mx = 40.0, double my = 15.0);

/// Provided. The RNEA (rnea above) for a model; gravity = false sets g = 0.
Eigen::VectorXd inverse_dynamics(const Robot& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& ddq, bool gravity = true);

/// Return (m, c, g) of MR Eq. 9.33 at path parameter s.
/// Use three calls of inverse_dynamics (no new dynamics code):
///   g(s) = ID(theta, 0, 0)            with gravity,
///   m(s) = ID(theta, 0, theta')       without gravity  (= M theta'),
///   c(s) = ID(theta, theta', theta'') without gravity  (= M theta'' + theta'^T Gamma theta').
PathDynamics path_dynamics(const Robot& model, const Path& path, double s);

/// (L, U) of MR Eqs. 9.36-9.37 at one state: the smallest and largest sdd that satisfy every
/// tau_min_i <= m_i sdd + c_i sdot^2 + g_i <= tau_max_i.
/// If m_i < 0 the two bounds swap. If |m_i| < EPS_M (zero-inertia point) joint i gives no bound on
/// sdd, but if c_i sdot^2 + g_i lies outside [tau_min_i, tau_max_i] the state is inadmissible: return
/// (L, U) = (+inf, -inf). The state is admissible when L <= U.
std::pair<double, double> accel_limits(const Eigen::VectorXd& m, const Eigen::VectorXd& c, const Eigen::VectorXd& g,
                                       double sdot, const Eigen::VectorXd& tau_min, const Eigen::VectorXd& tau_max);

/// sdot_lim(s): the largest sdot with L(s, sdot) <= U(s, sdot) (the velocity limit curve, MR Sec. 9.4.1),
/// combined with the kinematic limit |theta'_i| sdot <= qdot_max_i if qdot_max is given (non-empty).
/// Bisection on sdot in [0, sdot_max] (assumes one admissible interval [0, sdot_lim], MR Sec. 9.4.4).
/// Returns sdot_max if no limit is found below it, and 0 if the state (s, 0) is inadmissible.
double velocity_limit(const Robot& model, const Path& path, double s, const Eigen::VectorXd& tau_min,
                      const Eigen::VectorXd& tau_max, const Eigen::VectorXd& qdot_max = Eigen::VectorXd(),
                      double sdot_max = 100.0, double tol = 1e-9);

/// Time-optimal time scaling with toppra (TOPP-RA; C++: toppralite), the same problem as MR Sec. 9.4.
/// Constraints: joint torques tau_min <= tau <= tau_max (default, i.e. empty tau_min: -tau_max) via
/// inverse_dynamics, and optionally |qdot| <= qdot_max, |qddot| <= qddot_max (non-empty vectors). Start and end
/// at rest. Grid: N + 1 equally spaced points on [0, 1]. Throws std::runtime_error if no feasible time scaling
/// exists.
ToppResult topp(const Robot& model, const Path& path, const Eigen::VectorXd& tau_max,
                const Eigen::VectorXd& qdot_max = Eigen::VectorXd(), const Eigen::VectorXd& qddot_max = Eigen::VectorXd(),
                int N = 200, const Eigen::VectorXd& tau_min = Eigen::VectorXd());

/// Provided. theta(s(t)) with your quintic s(t), as a trajectory traj(t, order) (MR Eqs. 9.1-9.2).
/// Sampled on t in [0, 1], quintic_traj(path, 1.0) gives the Q, Qd, Qdd that min_time_scale needs for the
/// TOPP-vs-quintic comparison of Part D.
TrajectoryFn quintic_traj(const Path& path, double T);

/// Provided. Peak |qdot|/qdot_max and |tau|/tau_max per joint, and the peak |jerk| (finite differences of
/// qddot) along a trajectory traj(t, order) sampled at N+1 instants on [0, T].
PeakUsage peak_usage(const Robot& model, const TrajectoryFn& traj, double T, const Eigen::VectorXd& tau_max,
                     const Eigen::VectorXd& qdot_max, int N = 1000);

// ---------------------------------------------------------------------------
// Part E -- a command file for the course arm (provided)
// ---------------------------------------------------------------------------

/// Provided. Sample a trajectory for the servo stream. `slow` >= 1 stretches time uniformly
/// (q_cmd(t) = q(t / slow)), which divides all speeds by `slow` and accelerations by slow^2.
/// Returns (t, q): t = 0, dt, ..., slow*T (the last sample is exactly slow*T), q of size len(t) x n (row = sample).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> command_table(const TrajectoryFn& traj, double T, double dt = 0.02,
                                                          double slow = 1.0);

/// Provided. Safety check of a command table before it goes to the arm (course-arm hardware safety checklist,
/// rules 1 and 3; Course resources):
///   * within_limits: every q stays at least `margin_deg` (checklist: 10 deg) inside the joint limits;
///   * peak_speed_ratio: max over joints and samples of |dq/dt| / (speed_fraction * qdot_max)
///     (finite differences; qdot_max = the RATED speeds of course_arm_model);  speed_ok = ratio <= 1 + tol;
///   * starts_at_rest / ends_at_rest: first and last steps are smaller than 1e-3 rad;
///   * ok = all of the above.
CommandCheck check_command(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, const Robot& model,
                           double speed_fraction, double margin_deg = 10.0, double tol = 0.01);

}  // namespace lab

#endif  // SBEG461_LAB9_HPP
