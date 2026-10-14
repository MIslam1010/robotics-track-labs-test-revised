// SBEG 46X Lab 1 -- Software remote centre of motion (RCM): constrained resolved-rate control of the course arm
// holding a laparoscope (C++ track): declarations.
//
// The same functions as starter.py (module rcm.py of your course library sbeg46x_medrob), in namespace `lab`, with
// Eigen types instead of NumPy arrays.  Write your code in starter.cpp.  Do not change these declarations:
// test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR):
//   space-form PoE  T_sb(theta) = e^[S1]th1 ... e^[S6]th6 M                       (MR Eq. 4.14)
//   space Jacobian  V_s = J_s(theta) thetadot,  V_s = (w_s, v_s)                  (MR Eq. 5.11)
//   velocity of a point p fixed to the moving body: pdot = v_s + w_s x p = [-[p]  I] V_s   (from MR Eq. 3.73, p. 97)
//   inverse velocity kinematics thetadot = J^+ V_d                                 (MR Eq. 6.7, p. 231)
//   damped least squares  A^# = A^T (A A^T + lam^2 I)^-1
// Twists V = (omega, v): angular part first.
//
// Software RCM (this lab).  The tool frame {b} is at the laparoscope tip, z_b along the shaft towards the tip.  The
// trocar port p_port is fixed in {s}.  The shaft point closest to the port is p_c = p_tip - s z, s = z.(p_tip - p_port)
// (the insertion depth beyond the port).  The RCM constraint asks that the material point of the shaft at p_c moves
// only ALONG the shaft:  N^T (v_s + w_s x p_c) = k_rcm N^T (p_port - p_c),  N = [n1 n2] perpendicular to z.
// Together with the tip task  v_tip = pdot_d + k_p (p_d - p_tip)  this gives 5 equations for 6 joint rates;
// the remaining freedom (roll about the shaft, mixed with the arm's self-motion) is resolved by the minimum-norm
// (damped) pseudo-inverse.
//
// Python -> C++: a dict -> a small struct with the same field names; a tuple -> std::pair (auto [a, b] = ...);
// ValueError -> std::invalid_argument; R_sv=None -> the default argument Eigen::Matrix3d::Identity().
// Units: m, rad, s.
#ifndef SBEG46X_LAB1_HPP
#define SBEG46X_LAB1_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// 3 x 2 matrix (the basis N = [n1 n2] perpendicular to the shaft).
using Matrix32d = Eigen::Matrix<double, 3, 2>;

// ===========================================================================================================
// Provided (complete): robot model (course_models.cpp = course_models.py) and helpers (helpers.cpp)
// ===========================================================================================================

/// Straight teaching laparoscope length (m, illustrative; a clinical laparoscope is ~0.3 m or longer).
constexpr double TOOL_LENGTH = 0.12;

/// Course arm PoE model with a straight tool (Python: the dict returned by course_arm_kinematics).
struct Robot {
    Eigen::MatrixXd Slist;     ///< 6 x 6 space screw axes, COLUMNS S_i = (w_i, v_i)    (MR Example 4.5 form)
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();  ///< home pose of the tool frame {b} (at the tool tip)
    Eigen::MatrixXd limits;    ///< 6 x 2 software joint limits (rad) = hardware limits minus the 10 deg margin
    Eigen::VectorXd qdot_max;  ///< 6 joint-speed limits (rad/s) = 25 % of the rated 160 deg/s (first-run rule)
    double tool = TOOL_LENGTH; ///< the tool length (m)
};

/// The course arm (low-cost desktop UR-type 6R, myCobot-280 class) holding a straight tool of length `tool`
/// along z_b, so {b} sits at the tool tip.  (Provided, course_models.cpp.)
Robot course_arm_kinematics(double tool = TOOL_LENGTH);

/// One segment of the tip path, written in the port frame {p} (see make_trajectory).
struct PathSegment {
    std::string type;                              ///< "line" or "circle"
    Eigen::Vector3d to = Eigen::Vector3d::Zero();  ///< line: end point in {p} (m)
    double radius = 0.0;                           ///< circle: radius (m)
    double turns = 0.0;                            ///< circle: number of turns
    double T = 0.0;                                ///< duration (s)
};

/// The lab task (Python: the dict returned by load_task; task["gains"]["kp"] -> task.gains.kp, ...).
struct Task {
    Eigen::Vector3d p_port = Eigen::Vector3d::Zero();  ///< trocar port in {s} (m)
    Eigen::VectorXd theta0;            ///< start configuration (rad)
    std::vector<PathSegment> path;     ///< the tip path
    double dt = 0.005;                 ///< simulation step (s)
    struct Gains {
        double kp = 5.0, k_rcm = 10.0, lam = 0.0;
    } gains;
    struct Acceptance {
        double max_tip_error_mm = 0.5, max_rcm_error_mm = 0.5;
    } acceptance;
};

/// Load the lab task (port, start configuration, tip path) from data/lab1_task.json (empty path) or `path`.
/// (Provided.)
Task load_task(const std::string& path = "");

/// Versions and backends of your lab environment (Checkpoint 0): compiler, C++ standard, Eigen, kinematics
/// backend.  (Provided.)
std::map<std::string, std::string> environment_report();

/// [p], the 3 x 3 skew-symmetric matrix with [p] x = p x x (MR Eq. 3.30).  (Provided.)
Eigen::Matrix3d skew(const Eigen::Vector3d& p);

/// Minimum-jerk time scaling s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 and ds/dtau (MR Sec. 9.2.2, quintic, p. 332);
/// tau is clipped to [0, 1].  Returns {s, ds}.  (Provided.)
std::pair<double, double> min_jerk(double tau);

/// Rotation R_sp of a 'port frame' {p}: z_p = initial shaft direction, x_p horizontal.  (Provided.)
Eigen::Matrix3d port_frame(const Eigen::Vector3d& z_axis);

/// traj(t) -> {p_d, pdot_d} in {s}.
using TrajectoryFn = std::function<std::pair<Eigen::Vector3d, Eigen::Vector3d>(double)>;

/// Desired tip path from the task file.  (Provided.)
/// The path is written in the port frame {p} (origin at the port, z_p along the initial shaft direction z0, in m):
///   line   (to, T)          straight line, minimum-jerk timing
///   circle (radius, turns, T)  circle about the current point's z_p-axis projection (centre = start - r x_p)
/// Returns {traj, T_total} with traj(t) -> {p_d, pdot_d} in {s}.
std::pair<TrajectoryFn, double> make_trajectory(const Task& task, const Eigen::Vector3d& p_tip0,
                                                const Eigen::Vector3d& z0);

// ===========================================================================================================
// Kinematics backend (given in starter.cpp: mrcpp, the equivalent of the Python modern_robotics fallback;
// you may replace the two bodies by your own SBEG 361 C++ functions, 361 Lab 5 FK and 361 Lab 7 Jacobian)
// ===========================================================================================================

/// Name of the active kinematics backend (Python: KIN_BACKEND).
std::string kinematics_backend();

/// T = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14).
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Space Jacobian J_s(theta), 6 x n (MR Eq. 5.11).
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

// ===========================================================================================================
// Part A: geometry of the tool and the port
// ===========================================================================================================

/// T_sb(theta) of the laparoscope tip frame (MR Eq. 4.14), using the kinematics backend above.
Eigen::Matrix4d tool_pose(const Robot& robot, const Eigen::VectorXd& theta);

/// Tip position p_tip and unit shaft direction z (towards the tip) from T_sb.  Returns {p_tip, z}.
std::pair<Eigen::Vector3d, Eigen::Vector3d> shaft(const Eigen::Matrix4d& T);

/// Port geometry of the shaft (Python: the dict returned by rcm_geometry).
struct RcmGeometry {
    double depth = 0.0;                             ///< s = z.(p_tip - p_port), insertion depth beyond the port (m)
    Eigen::Vector3d p_c = Eigen::Vector3d::Zero();  ///< p_tip - s z, the shaft point closest to the port
    Eigen::Vector3d e = Eigen::Vector3d::Zero();    ///< p_port - p_c, the RCM error vector (perpendicular to z)
    double dist = 0.0;                              ///< |e|, the distance from the port to the shaft line (m)
};

/// Port geometry of the shaft for the tool pose T (see RcmGeometry).
RcmGeometry rcm_geometry(const Eigen::Matrix4d& T, const Eigen::Vector3d& p_port);

/// 3 x 2 matrix N = [n1 n2] of orthonormal vectors perpendicular to the unit vector z (right-handed n1, n2, z).
Matrix32d perp_basis(const Eigen::Vector3d& z);

// ===========================================================================================================
// Part B: Jacobians
// ===========================================================================================================

/// 3 x n Jacobian of the point p of the moving body (p in {s} coordinates):  pdot = [-[p]  I] J_s thetadot.
Eigen::MatrixXd point_jacobian(const Eigen::MatrixXd& Js, const Eigen::Vector3d& p);

/// The `info` of rcm_jacobian (Python: dict(T=T_sb, N=N, geometry=rcm_geometry(T, p_port))).
struct RcmInfo {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Matrix32d N = Matrix32d::Zero();
    RcmGeometry geometry;
};

/// Stacked 5 x 6 task Jacobian of the software RCM.
/// Rows 1-3: tip velocity  J_tip = [-[p_tip] I] J_s.
/// Rows 4-5: lateral velocity of the shaft point at the port  N^T J_c,  J_c = [-[p_c] I] J_s,  N = perp_basis(z).
/// Returns {A, info}.
std::pair<Eigen::MatrixXd, RcmInfo> rcm_jacobian(const Robot& robot, const Eigen::VectorXd& theta,
                                                 const Eigen::Vector3d& p_port);

/// Damped least-squares inverse A^T (A A^T + lam^2 I)^-1; lam = 0 gives the pseudo-inverse A^+ (MR Eq. 6.7).
Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& A, double lam = 0.0);

// ===========================================================================================================
// Part C: controllers
// ===========================================================================================================

/// Software-RCM resolved-rate law (6 joint rates):
///     thetadot = A^#(lam) [ pdot_d + kp (p_d - p_tip) ;  k_rcm N^T e ],   e = p_port - p_c.
Eigen::VectorXd rcm_rates(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::Vector3d& p_port,
                          const Eigen::Vector3d& p_d, const Eigen::Vector3d& pdot_d, double kp = 5.0,
                          double k_rcm = 10.0, double lam = 0.0);

/// Naive resolved rate WITHOUT the RCM rows: thetadot = J_tip^#(lam) [pdot_d + kp (p_d - p_tip)].
Eigen::VectorXd tip_only_rates(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::Vector3d& p_d,
                               const Eigen::Vector3d& pdot_d, double kp = 5.0, double lam = 0.0);

/// Uniformly scale qdot so that |qdot_i| <= qdot_max_i for every joint (keeps the direction of motion).
Eigen::VectorXd scale_rates(const Eigen::VectorXd& qdot, const Eigen::VectorXd& qdot_max);

/// Result of simulate (Python: dict of arrays), one row / entry per step k = 0..K.
struct SimLog {
    Eigen::VectorXd t;        ///< K+1
    Eigen::MatrixXd theta;    ///< (K+1) x 6
    Eigen::MatrixXd p_tip;    ///< (K+1) x 3
    Eigen::MatrixXd p_d;      ///< (K+1) x 3
    Eigen::VectorXd tip_err;  ///< K+1, |p_d - p_tip| (m)
    Eigen::VectorXd rcm_err;  ///< K+1, rcm_geometry(...).dist (m)
    Eigen::VectorXd depth;    ///< K+1, rcm_geometry(...).depth (m)
};

/// Simulate the tip path of `task` with forward-Euler integration theta <- theta + dt * thetadot.
///
/// controller = "rcm" (rcm_rates) or "tip" (tip_only_rates); anything else: throw std::invalid_argument.
/// Joint rates are limited with scale_rates (robot.qdot_max).  K = round(T_total / dt).
SimLog simulate(const Robot& robot, const Task& task, const std::string& controller = "rcm", double dt = 0.005,
                double kp = 5.0, double k_rcm = 10.0, double lam = 0.0);

/// Write a SimLog as CSV for plotting (columns t, theta_1..theta_6, p_tip_x..z, p_d_x..z, tip_err, rcm_err, depth).
/// (Provided.)
void save_log_csv(const std::string& path, const SimLog& log);

// ===========================================================================================================
// Part D: fulcrum effect and motion scaling
// ===========================================================================================================

/// Hand-held instrument pivoting at the port: lateral tip motion / lateral handle motion = -depth / (L - depth)
/// (minus sign: the tip moves opposite to the hand).  L = shaft length from handle to tip.
double fulcrum_ratio(double L, double depth);

/// Motion scaling: tip displacement in {s} = scale * R_sv d_leader, where d_leader is the leader (hand)
/// displacement in the view (camera) frame {v} and R_sv the camera orientation in {s} (identity by default).
Eigen::Vector3d scaled_tip_command(const Eigen::Vector3d& d_leader, double scale,
                                   const Eigen::Matrix3d& R_sv = Eigen::Matrix3d::Identity());

}  // namespace lab

#endif  // SBEG46X_LAB1_HPP
