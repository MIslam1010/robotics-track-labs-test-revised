// SBEG 361 Lab 8 -- Statics, manipulability maps of an assistive arm, workspace design (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 5.2-5.4:
//   statics (power balance tau^T thetadot = F^T V):  tau = J^T(theta) F,  F = (m, f) moment first      (MR Eq. 5.26)
//       tau = J_b^T F_b = J_s^T F_s;  wrench frame change F_s = [Ad_{T_bs}]^T F_b                     (MR Eq. 3.98)
//       n = 6 and J invertible:  F = J^-T tau;  Null(J^T) = wrenches resisted with zero joint torque    (MR pp. 188-189)
//   singularity: rank J(theta) < min(6, n); rank J_s = rank J_b                                           (MR Sec. 5.3)
//   manipulability ellipsoid  qdot^T A^-1 qdot = 1, A = J J^T, semi-axes sqrt(lambda_i(A))               (MR Eq. 5.27)
//   force ellipsoid  f^T B^-1 f = 1, B = (J J^T)^-1, semi-axes 1/sqrt(lambda_i(A)), same axes             (MR p. 197)
//   measures  mu1 = sqrt(lmax/lmin) >= 1,  mu2 = lmax/lmin = mu1^2,  mu3 = sqrt(det A)                    (MR pp. 196-197)
//   planar 2R point Jacobian (tip velocity in the plane)                                                  (MR Eq. 5.1)
// Wrenches and twists are 6-vectors with the angular part first.  Units: m, kg, s, rad, N, N m.
//
// Python -> C++: ValueError -> std::invalid_argument; Jacobians, screw lists and grids -> Eigen::MatrixXd
// (Slist: 6 x n, one axis per COLUMN as in Python; grids: row = y value, column = x value, as in Python);
// joint vectors and lists of numbers -> Eigen::VectorXd; an (N, 2) / (N, n) array -> Eigen::MatrixXd (row = sample);
// a tuple -> std::pair or a small struct (fields named as in Python); a dict -> a struct with the same keys;
// `None` for limits -> an empty matrix (Eigen::MatrixXd()); `None` returned -> an empty std::optional;
// np.nan / np.inf -> std::numeric_limits<double>::quiet_NaN() / infinity() (test with std::isnan / std::isinf).
#ifndef SBEG361_LAB8_HPP
#define SBEG361_LAB8_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Same threshold as modern_robotics.NearZero.
constexpr double TOL = 1e-6;

/// A robot model of data/robots.json (Python: the tuple (Slist, M, info)).
struct Robot {
    Eigen::MatrixXd Slist;     ///< 6 x n space screw axes (columns)
    Eigen::Matrix4d M;         ///< home configuration of {b}
    std::string name;          ///< info["name"]
    std::string joint_types;   ///< info["joint_types"], e.g. "RRRRRR"
    Eigen::MatrixXd limits;    ///< info["limits"]: n x 2 joint limits (rad); empty if null in the file
    Eigen::VectorXd q_home;    ///< info["q_home"]; empty if null in the file
    Eigen::VectorXd tau_max;   ///< info["tau_max"] (rated torques, N m); empty if null in the file
    Eigen::VectorXd L;         ///< info["L"] (link lengths of the planar arms, m); empty if absent
};

/// (semi_axes, directions) of an ellipsoid: directions.col(i) is the unit axis of length semi_axes(i).
struct Ellipsoid {
    Eigen::VectorXd semi_axes;
    Eigen::MatrixXd directions;
};

/// dict(mu1, mu2, mu3) of manipulability_measures.
struct Measures {
    double mu1;
    double mu2;
    double mu3;
};

/// dict(rank, sigma_min, lost_twists, null_motions) of singularity_report.
struct SingularityReport {
    int rank;
    double sigma_min;
    Eigen::MatrixXd lost_twists;   ///< m x (m - rank)
    Eigen::MatrixXd null_motions;  ///< n x (n - rank)
};

/// One row of design_search: dict(ratio, L1, L2, b, worst, fraction).
struct DesignRow {
    double ratio;
    double L1;
    double L2;
    double b;
    double worst;
    double fraction;
};

/// (rows, best) of design_search; `best` is empty (Python: None) if no design is feasible.
struct DesignSearchResult {
    std::vector<DesignRow> rows;
    std::optional<DesignRow> best;
};

/// dict(mu1, mu3, sigma_min) of measures_along_path: one value per row of Q.
struct PathMeasures {
    Eigen::VectorXd mu1;
    Eigen::VectorXd mu3;
    Eigen::VectorXd sigma_min;
};

// ---------------------------------------------------------------------------
// Helpers (complete; from Labs 2-7) -- given in starter.cpp
// ---------------------------------------------------------------------------

/// 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// 4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V);

/// T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// 6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);

/// e^[S]theta for a twist matrix (MR Eqs. 3.51, 3.88-3.89).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

/// Space-form PoE forward kinematics (MR Eq. 4.14).
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16).
Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M);

/// Space Jacobian J_s(theta) (MR Eq. 5.11) -- your Lab 7 function.
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body Jacobian J_b(theta) (MR Eq. 5.18) -- your Lab 7 function.
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// 2 x 2 tip-velocity Jacobian of a planar 2R arm (MR Eq. 5.1).
Eigen::Matrix2d jacobian_2r(const Eigen::Vector2d& theta, double L1, double L2);

/// Analytic IK of a planar 2R arm with the base at the origin (law of cosines), given complete.
/// One branch only: elbow=+1 gives theta2 >= 0, elbow=-1 gives theta2 <= 0.  Returns an empty optional
/// (Python: None) if the point is out of reach.  Not the same function as Lab 9's ik_2r(x, y, L1, L2), which
/// returns both branches; keep both names in your library.
std::optional<Eigen::Vector2d> ik_2r_given(double x, double y, double L1, double L2, int elbow = 1);

/// True if every joint is inside [low, high] (limits n x 2; an empty matrix means no limits).
bool within_limits(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits);

/// Model `name` of data/robots.json (numbers = Course resources: course robot models).
/// An empty path means the lab's data/robots.json.
Robot load_robot(const std::string& name, const std::string& path = "");

/// N x 2 therapy targets (table coordinates, m) from data/therapy_targets.csv (empty path = the lab's file).
Eigen::MatrixXd load_targets(const std::string& path = "");

/// (t, Q) from data/feeding_path.csv; Q is N x 7 (rad) (empty path = the lab's file).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> load_feeding_csv(const std::string& path = "");

/// "all" -> J; "angular" -> J_omega (top 3 rows); "linear" -> J_v (bottom 3 rows) (MR p. 196).
/// Any other string: std::invalid_argument.
Eigen::MatrixXd select_rows(const Eigen::MatrixXd& J, const std::string& rows = "all");

/// Value of a measure of J: "sigma_min" (smallest singular value), "mu1", "mu2" or "mu3"
/// (via manipulability_measures); any other string: std::invalid_argument.  (Python: _measure.)
double measure_of(const Eigen::MatrixXd& J, const std::string& measure);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- statics
// ---------------------------------------------------------------------------

/// Joint torques/forces that balance an end-effector wrench: tau = J^T F (MR Eq. 5.26).
/// J and F must be in the same frame (J_s with F_s, or J_b with F_b).  For a planar arm J may be
/// the 2 x n point Jacobian and F the planar tip force (MR Eq. 5.3).
Eigen::VectorXd joint_torques(const Eigen::MatrixXd& J, const Eigen::VectorXd& F);

/// End-effector wrench produced by joint torques at equilibrium: F = J^-T tau (MR Eq. 5.4, p. 173; general case
/// Sec. 5.2, p. 188).  Only well posed for a square, non-singular J: throw std::invalid_argument otherwise
/// (singular: smallest singular value < tol * max(1, largest)).
Eigen::VectorXd wrench_from_torques(const Eigen::MatrixXd& J, const Eigen::VectorXd& tau, double tol = 1e-9);

/// F_s = [Ad_{T_bs}]^T F_b (MR Eq. 3.98).
sbeg::Vector6d body_to_space_wrench(const sbeg::Vector6d& F_b, const Eigen::Matrix4d& T_sb);

/// Wrench F_s = (p x f, f) that the robot must apply to hold a point mass m at point p (in {s}):
/// f = -m g (the opposite of the load's weight), moment about the origin of {s} = p x f.
sbeg::Vector6d holding_wrench_space(double m, const Eigen::Vector3d& p,
                                    const Eigen::Vector3d& g = Eigen::Vector3d(0.0, 0.0, -9.81));

/// Orthonormal basis (6 x k) of Null(J^T): the wrenches the robot resists with zero joint torque
/// and cannot actively generate (MR p. 189).  Columns are left singular vectors of J.
Eigen::MatrixXd wrench_null_space(const Eigen::MatrixXd& J, double tol = 1e-6);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- ellipsoids and measures
// ---------------------------------------------------------------------------

/// (semi_axes, directions) of the manipulability ellipsoid qdot^T (J J^T)^-1 qdot = 1 (MR Eq. 5.27).
/// semi_axes = sqrt(lambda_i(A)), A = J J^T, sorted from longest to shortest; directions.col(i) is the
/// unit eigenvector of A for semi_axes(i) (sign arbitrary).
Ellipsoid manipulability_ellipsoid(const Eigen::MatrixXd& J);

/// (semi_axes, directions) of the force ellipsoid f^T (J J^T) f = 1 (MR p. 197).
/// Same directions and ORDER as manipulability_ellipsoid(J); semi_axes = 1/sqrt(lambda_i(A))
/// (infinity where lambda_i <= tol): the longest velocity axis is the shortest force axis.
Ellipsoid force_ellipsoid(const Eigen::MatrixXd& J, double tol = 1e-12);

/// (mu1, mu2, mu3) of A = J J^T (MR pp. 196-197).
/// mu1 = sqrt(lmax/lmin) (isotropy, 1 is best), mu2 = lmax/lmin (condition number of A),
/// mu3 = sqrt(det A) = product of the semi-axes (volume measure, larger is better).
/// At a singularity mu1 = mu2 = infinity and mu3 = 0.
Measures manipulability_measures(const Eigen::MatrixXd& J, double tol = 1e-12);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- singularities
// ---------------------------------------------------------------------------

/// (rank, sigma_min, lost_twists, null_motions) of a Jacobian (MR Sec. 5.3).
///   rank          number of singular values > tol
///   sigma_min     smallest of the min(m, n) singular values
///   lost_twists   m x (m - rank) orthonormal basis of twists the end-effector cannot produce
///                 (= Null(J^T), the wrenches resisted with zero torque)
///   null_motions  n x (n - rank) orthonormal basis of joint velocities that give zero twist
SingularityReport singularity_report(const Eigen::MatrixXd& J, double tol = 1e-6);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- maps and design (rehab_2r)
// ---------------------------------------------------------------------------

/// Grid (ys.size() x xs.size()) of a measure of the 2R point Jacobian over table points (x, y).
/// The base is at `base`; each point is solved with ik_2r_given(x - bx, y - by, ...).  Entries are NaN
/// where the point is out of reach or the IK solution violates `limits`.  measure: "mu1", "mu2",
/// "mu3" or "sigma_min".
Eigen::MatrixXd manipulability_map_2r(double L1, double L2, const Eigen::VectorXd& xs, const Eigen::VectorXd& ys,
                                      const std::string& measure = "mu3", int elbow = 1,
                                      const Eigen::MatrixXd& limits = Eigen::MatrixXd(),
                                      const Eigen::Vector2d& base = Eigen::Vector2d(0.0, 0.0));

/// (worst, fraction) over a task region (points: N x 2, one (x, y) per row): worst = min of mu3 / sigma_min, or
/// max of mu1 / mu2, over the reachable points; fraction = share of points reachable within limits (NaN worst if
/// none).
std::pair<double, double> region_score(double L1, double L2, const Eigen::MatrixXd& points,
                                       const std::string& measure = "mu3", int elbow = 1,
                                       const Eigen::MatrixXd& limits = Eigen::MatrixXd(),
                                       const Eigen::Vector2d& base = Eigen::Vector2d(0.0, 0.0));

/// Grid search of a 2R design: L1 = ratio * total, L2 = total - L1, base at (0, -b).
/// Returns (rows, best): rows = one DesignRow(ratio, L1, L2, b, worst, fraction) per design in loop order
/// (ratios outer, base_offsets inner); best = the feasible row (fraction == 1) with the best worst
/// case (largest for mu3 / sigma_min, smallest for mu1 / mu2; the first one on a tie), or empty if no row is
/// feasible.
DesignSearchResult design_search(double total, const Eigen::VectorXd& ratios, const Eigen::VectorXd& base_offsets,
                                 const Eigen::MatrixXd& points, const Eigen::MatrixXd& limits = Eigen::MatrixXd(),
                                 int elbow = 1, const std::string& measure = "mu3");

// ---------------------------------------------------------------------------
// Checkpoint 5 -- spatial arms
// ---------------------------------------------------------------------------

/// (mu1, mu3, sigma_min) vectors (one value per row of Q) of the BODY Jacobian rows
/// `rows` ("linear" = J_v, "angular" = J_omega, "all") along a joint path Q (N x n) (MR p. 196).
PathMeasures measures_along_path(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Q,
                                 const std::string& rows = "linear");

/// Grid (gj.size() x gi.size()) of a measure of the body Jacobian rows `rows` when joints i and j
/// (0-based) sweep the values gi and gj and the other joints stay at q0.
Eigen::MatrixXd joint_space_map(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::VectorXd& q0,
                                int i, int j, const Eigen::VectorXd& gi, const Eigen::VectorXd& gj,
                                const std::string& measure = "sigma_min", const std::string& rows = "all");

}  // namespace lab

#endif  // SBEG361_LAB8_HPP
