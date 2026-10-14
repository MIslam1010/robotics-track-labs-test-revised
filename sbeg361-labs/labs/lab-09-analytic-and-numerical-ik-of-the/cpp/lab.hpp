// SBEG 361 Lab 9 -- Analytic and numerical inverse kinematics of the course arm; all-solutions search
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Chapter 6, and Murray-Li-Sastry (MLS) Ch. 3 Sec. 3:
//   forward kinematics   T(theta) = e^[S1]th1 ... e^[S6]th6 M                      (MR Eq. 4.14, p. 140)
//   2R planar IK         beta = acos((L1^2+L2^2-x^2-y^2)/(2 L1 L2)),  alpha = acos((x^2+y^2+L1^2-L2^2)/(2 L1 r)),
//                        gamma = atan2(y, x);  righty (th1, th2) = (gamma-alpha, pi-beta),
//                        lefty (gamma+alpha, beta-pi)                                (MR pp. 218-219)
//   Paden-Kahan          PK1  e^[xi th] p = q;  PK2  e^[xi1 th1] e^[xi2 th2] p = q;  PK3 |q - e^[xi th] p| = delta
//                        (MLS pp. 99-103, Eqs. 3.17-3.29; Pardos-Gotor Sec. 4.3, pp. 59-63)
//   Newton-Raphson       theta_{k+1} = theta_k - (dg/dtheta)^-1 g(theta_k)            (MR Sec. 6.2.1, p. 225)
//   body-twist IK        [V_b] = log(T_sb(theta_i)^-1 T_sd),  theta_{i+1} = theta_i + J_b^+(theta_i) V_b
//                        stop when |omega_b| <= eps_omega and |v_b| <= eps_v        (MR Sec. 6.2.2, pp. 228-229)
//   space-twist IK       V_s = [Ad_Tsb] V_b,  theta_{i+1} = theta_i + J_s^+ V_s        (MR p. 229)
// Twists are (omega, v), angular part first.  MLS and Pardos-Gotor write twists linear-first.
//
// Python -> C++: ValueError -> std::invalid_argument; screw lists Slist / Blist -> Eigen::MatrixXd (6 x n, one axis
// per column, as in Python); joint vectors -> Eigen::VectorXd; a list of joint vectors -> std::vector<VectorXd>;
// a tuple (theta1, theta2) -> std::pair<double, double>; `None` -> an empty std::optional; tuples of 3+ values and
// the iteration-table dicts -> the small structs below (fields named as in Python); the callables g, dg of
// newton_raphson -> lab::ResidualFunction / lab::JacobianFunction (a scalar problem uses vectors of length 1).
#ifndef SBEG361_LAB9_HPP
#define SBEG361_LAB9_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Same threshold as modern_robotics.NearZero.
constexpr double TOL = 1e-6;

/// Geometry of the UR-type course arm (Python: info["geometry"]) [m].
struct Geometry {
    double W1 = 0.0, W2 = 0.0, L1 = 0.0, L2 = 0.0, H1 = 0.0, H2 = 0.0;
    double tool = 0.0;  ///< tool length along z_b (0 if absent)
};

/// A robot model of data/robots.json (Python: the tuple (Slist, M, info)).
struct Robot {
    Eigen::MatrixXd Slist;    ///< 6 x n space screw axes (columns)
    Eigen::Matrix4d M;        ///< home configuration of {b}
    std::string name;         ///< info["name"]
    std::string joint_types;  ///< info["joint_types"], e.g. "RRRRRR"
    Eigen::MatrixXd limits;   ///< n x 2 joint limits [lower, upper] (rad)
    Eigen::VectorXd q_home;   ///< ready pose (rad)
    bool has_geometry = false;  ///< true for the UR-type arms (course_arm, desktop_6r)
    Geometry geometry;        ///< info["geometry"] (only if has_geometry)
};

/// One row of data/biopsy_targets.csv (Python: dict {id, entry, axis, note}).
struct BiopsyTarget {
    std::string id;
    Eigen::Vector3d entry;  ///< skin entry point in {s} [m]
    Eigen::Vector3d axis;   ///< unit needle direction (pointing into the phantom)
    std::string note;
};

/// g(theta) -> residual vector (Python: float or array).
using ResidualFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
/// dg(theta) -> derivative / Jacobian matrix (Python: float or 2-D array).
using JacobianFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

/// One entry (k, theta_k, g(theta_k)) of the Newton-Raphson history.
struct NewtonStep {
    int k = 0;
    Eigen::VectorXd theta;
    Eigen::VectorXd g;
};

/// Result of newton_raphson (Python: the tuple (theta, history)).
struct NewtonResult {
    Eigen::VectorXd theta;
    std::vector<NewtonStep> history;
};

/// One row of the IK iteration table (Python: dict {i, theta, V_b or V_s, w_err, v_err}).
struct IKRow {
    int i = 0;
    Eigen::VectorXd theta;  ///< theta^i (a copy)
    sbeg::Vector6d V;       ///< V_b (ik_newton_body) or V_s (ik_newton_space)
    double w_err = 0.0;     ///< |omega|
    double v_err = 0.0;     ///< |v|
};

/// Result of ik_newton_body / ik_newton_space (Python: the tuple (theta, success, table)).
struct IKTableResult {
    Eigen::VectorXd theta;
    bool success = false;
    std::vector<IKRow> table;
};

/// Result of all_solutions_search (Python: the tuple (solutions, counts, n_converged)).
struct SearchResult {
    std::vector<Eigen::VectorXd> solutions;
    std::vector<int> counts;
    int n_converged = 0;
};

// ---------------------------------------------------------------------------
// Helpers (complete; from Labs 2-7) -- given in starter.cpp
// ---------------------------------------------------------------------------

/// 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// omega from [omega].
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);

/// 4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V);

/// V = (omega, v) from a 4x4 matrix [V].
sbeg::Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);

/// T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// 6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);

/// Rotation matrix Rot(omega_hat, theta) by Rodrigues' formula (MR Eq. 3.51); the axis is normalised.
Eigen::Matrix3d rot(const Eigen::Vector3d& axis, double angle);

/// Rodrigues' formula e^[w]theta (MR Eq. 3.51).
Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat);

/// [w]theta = log R, theta in [0, pi] (MR Sec. 3.2.3.3).
Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R);

/// e^[S]theta for a twist matrix (MR Eqs. 3.88-3.89).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

/// [S]theta = log T (MR Sec. 3.3.3.2).
Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T);

/// Space-form PoE forward kinematics (MR Eq. 4.14).
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body-form PoE forward kinematics (MR Eq. 4.16).
Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16).
Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M);

/// Space Jacobian J_s(theta) (MR Eq. 5.11) -- your Lab 7 function.
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body Jacobian J_b(theta) (MR Eq. 5.18) -- your Lab 7 function.
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// Moore-Penrose pseudoinverse, the same numbers as np.linalg.pinv(A) (SVD, rcond 1e-15).  (C++ only.)
Eigen::MatrixXd pinv(const Eigen::MatrixXd& A);

/// Wrap an angle to [-pi, pi)  (Python: (theta + pi) % (2 pi) - pi).
double wrap_to_pi(double theta);

/// Wrap every entry of a joint vector to [-pi, pi).
Eigen::VectorXd wrap_to_pi(const Eigen::VectorXd& theta);

/// (rotation error angle [rad], position error [m]) between two poses.
std::pair<double, double> pose_error(const Eigen::Matrix4d& T, const Eigen::Matrix4d& T_d);

/// True if every joint value lies inside its [lower, upper] limit (minus a margin); limits is n x 2.
bool within_limits(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits, double margin = 0.0);

/// A model of data/robots.json (numbers identical to the course robot models).
/// An empty path means the lab's data/robots.json.
Robot load_robot(const std::string& name, const std::string& path = "");

/// The rows of data/biopsy_targets.csv. An empty path means the lab's data file.
std::vector<BiopsyTarget> load_biopsy_targets(const std::string& path = "");

/// n_starts x n matrix of initial guesses drawn uniformly in [-pi, pi) with a C++ random generator
/// (std::mt19937 seeded with `seed`; the same numbers on every compiler). For your own experiments (results sheet); the auto-checks use the
/// NumPy starts stored in cpp_cases.json.  (C++ only: replaces rng.uniform(-pi, pi, (n_starts, n)).)
Eigen::MatrixXd random_starts(int n_starts, int n, unsigned seed);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- Paden-Kahan subproblems
// ---------------------------------------------------------------------------

/// Paden-Kahan subproblem 1 (MLS p. 99-100, Eqs. 3.17-3.19): find theta with e^[xi theta] p = q,
/// xi = zero-pitch unit twist with axis direction omega through the point r.
///
/// Returns theta in (-pi, pi].  Throws std::invalid_argument if no solution exists (omega.u != omega.v or
/// |u'| != |v'|, tolerance 1e-6 m).  If p lies on the axis (u' = 0), every theta works: return 0.0.
double pk1(const Eigen::Vector3d& omega, const Eigen::Vector3d& r, const Eigen::Vector3d& p,
           const Eigen::Vector3d& q);

/// Paden-Kahan subproblem 2 (MLS pp. 100-102, Eqs. 3.20-3.25): find (theta1, theta2) with
/// e^[xi1 theta1] e^[xi2 theta2] p = q; the two axes (directions omega1, omega2) intersect at r
/// and are not parallel.
///
/// Returns a list of 0, 1 or 2 pairs (theta1, theta2).
std::vector<std::pair<double, double>> pk2(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2,
                                           const Eigen::Vector3d& r, const Eigen::Vector3d& p,
                                           const Eigen::Vector3d& q);

/// Paden-Kahan subproblem 3 (MLS pp. 102-103, Eqs. 3.26-3.29): find theta with
/// |q - e^[xi theta] p| = delta.
///
/// Returns a list of 0, 1 or 2 angles in [-pi, pi).
std::vector<double> pk3(const Eigen::Vector3d& omega, const Eigen::Vector3d& r, const Eigen::Vector3d& p,
                        const Eigen::Vector3d& q, double delta);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- analytic IK
// ---------------------------------------------------------------------------

/// Planar 2R inverse kinematics (MR pp. 218-219).
///
/// Returns {} if (x, y) is outside the annulus L1-L2 <= r <= L1+L2, otherwise
/// {righty, lefty} = {(gamma - alpha, pi - beta), (gamma + alpha, beta - pi)} (one pair if both coincide).
std::vector<std::pair<double, double>> ik_2r(double x, double y, double L1, double L2);

/// All analytic IK solutions of the UR-type course arm (MR Example 4.5 layout).
///
/// geom: W1, W2, L1, L2, H1, H2 [m] and tool [m] (tool along z_b).
/// Returns a list of up to 8 joint vectors (6 entries each), each wrapped to [-pi, pi).
/// Steps (lab handout, "Derivation of the course-arm IK"):
///   1. wrist point p5 = p - (W2 + tool) z_b (intersection of axes 5 and 6);
///   2. theta1 from  -sin(th1) p5x + cos(th1) p5y = W1   (2 solutions, shoulder left/right);
///   3. theta5 from  cos(th5) = -sin(th1) z_bx + cos(th1) z_by   (2 solutions, wrist flip);
///   4. theta6 from row 2 of R' = Rot(z,-th1) R, which equals (s5 c6, -s5 s6, c5);
///   5. theta234 from Rot(y, th234) = R' (Rot(z,-th5) Rot(y,th6) R_M)^T;
///   6. (theta2, theta3) from planar 2R IK of the joint-4 point; theta4 = theta234 - theta2 - theta3.
/// The wrist singularity sin(th5) = 0 (axes 4 and 6 parallel) is handled by setting theta6 = 0.
std::vector<Eigen::VectorXd> ik_course_arm(const Eigen::Matrix4d& T, const Geometry& geom);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- Newton-Raphson
// ---------------------------------------------------------------------------

/// Newton-Raphson root finding for g(theta) = 0 (MR Sec. 6.2.1), theta a vector (length 1 for a scalar problem).
///
/// g(theta) -> residual vector, dg(theta) -> derivative / Jacobian matrix.
/// Iteration theta_{k+1} = theta_k - dg(theta_k)^-1 g(theta_k) (linear solve if dg is square; pinv if not).
/// Stops when |g(theta_k)| <= tol or after max_iter steps.
/// Returns {theta, history} with history = [(k, theta_k, g(theta_k)), ...] starting at k = 0.
NewtonResult newton_raphson(const ResidualFunction& g, const JacobianFunction& dg, const Eigen::VectorXd& theta0,
                            double tol = 1e-10, int max_iter = 50);

/// Body-twist Newton-Raphson IK (MR Sec. 6.2.2, p. 228-229; same stopping rule and default
/// iteration limit as modern_robotics.IKinBody).
///
/// Returns {theta, success, table}; table rows {i, theta (copy), V = V_b, w_err = |omega_b|, v_err = |v_b|}
/// for i = 0, 1, ..., last.
IKTableResult ik_newton_body(const Eigen::MatrixXd& Blist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_sd,
                             const Eigen::VectorXd& theta0, double eomg = 1e-3, double ev = 1e-4, int max_iter = 20);

/// Space-twist Newton-Raphson IK (MR p. 229: V_s = [Ad_Tsb] V_b, J_s), as modern_robotics.IKinSpace.
///
/// Returns {theta, success, table} with rows {i, theta, V = V_s, w_err, v_err}.
IKTableResult ik_newton_space(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_sd,
                              const Eigen::VectorXd& theta0, double eomg = 1e-3, double ev = 1e-4,
                              int max_iter = 20);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- all-solutions search
// ---------------------------------------------------------------------------

/// Remove duplicates from a list of joint vectors; two solutions are equal if every wrapped
/// joint difference is below tol (rad).  Returns wrapped solutions in first-seen order.
std::vector<Eigen::VectorXd> unique_solutions(const std::vector<Eigen::VectorXd>& sols, double tol = 1e-4);

/// Multi-start numerical IK: run ik_newton_body from every row of `starts` (n_starts x n initial guesses,
/// uniform in [-pi, pi)^n; Python draws them inside the function with np.random.default_rng(seed)).
///
/// Returns {solutions, counts, n_converged}: distinct converged solutions (wrapped, tolerance 1e-4),
/// how many starts converged to each one (its basin size), and the total number of converged starts.
SearchResult all_solutions_search(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M,
                                  const Eigen::Matrix4d& T_sd, const Eigen::MatrixXd& starts, double eomg = 1e-9,
                                  double ev = 1e-9, int max_iter = 60);

/// Return the feasible solution closest to theta_now, or an empty std::optional if no solution is feasible.
///
/// A revolute joint angle and the same angle +- 2 pi give the same pose, so for every joint of a
/// solution consider the representatives s_j + 2 pi k (k = -1, 0, 1) that lie inside [lower, upper]
/// and keep the one closest to theta_now[j]; a solution is infeasible if some joint has none.
/// Among the feasible solutions return the one with the smallest Euclidean distance to theta_now.
/// limits is n x 2 (columns lower, upper).
std::optional<Eigen::VectorXd> select_solution(const std::vector<Eigen::VectorXd>& sols,
                                               const Eigen::VectorXd& theta_now, const Eigen::MatrixXd& limits);

// ---------------------------------------------------------------------------
// Checkpoint 5 -- biopsy needle guide
// ---------------------------------------------------------------------------

/// Desired pose T_sd of the needle-guide frame {b} for a biopsy trajectory.
///
/// z_b = unit vector along the needle direction `axis` (pointing into the patient); the guide tip is
/// `standoff` metres before the skin entry point: p = entry - standoff * z_b; x_b is the unit
/// projection of x_hint onto the plane normal to z_b (if x_hint is parallel to z_b use (0, 1, 0)),
/// y_b = z_b x x_b.
Eigen::Matrix4d needle_guide_pose(const Eigen::Vector3d& entry, const Eigen::Vector3d& axis, double standoff,
                                  const Eigen::Vector3d& x_hint = Eigen::Vector3d(1.0, 0.0, 0.0));

}  // namespace lab

#endif  // SBEG361_LAB9_HPP
