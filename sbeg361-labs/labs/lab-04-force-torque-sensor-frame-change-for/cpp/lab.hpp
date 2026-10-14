// SBEG 361 Lab 4 -- Exponential coordinates on SE(3), wrenches and force-torque sensor frame changes for a
// laparoscopic tool and an EOD gripper (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.3.3 and 3.4:
//   S = (omega, v), [S] in se(3)          screw axis, angular part FIRST          (MR Eq. 3.85)
//   e^{[S]theta} = [[e^{[w]theta}, G(theta) v], [0, 1]]                         (MR Prop. 3.25, Eq. 3.88)
//   G(theta) = I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2            (MR Eq. 3.87)
//   e^{[S]theta} = [[I, v theta], [0, 1]]  if omega = 0, |v| = 1                  (MR Eq. 3.89)
//   log: v = G^{-1}(theta) p,  G^{-1} = I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2   (MR Eqs. 3.91-3.92)
//   F = (m, f)                             wrench, MOMENT FIRST                   (MR Eq. 3.93)
//   V_b^T F_b = V_a^T F_a                  power is frame independent              (MR Eq. 3.94)
//   F_b = [Ad_{T_ab}]^T F_a                change of frame of a wrench             (MR Eq. 3.95, Prop. 3.27)
//   wrench screw (Poinsot): h = m.f/|f|^2, q = f x m/|f|^2   (MLS Eqs. 2.69-2.70, p. 65; MLS orders (f, tau))
//
// Force-torque sensor convention (as in MR Example 3.28, pp. 108-109): the sensor frame {f} reports the total
// wrench of the loads acting on everything distal to the sensor (gravity, contact), plus a constant electrical
// offset ("bias"), all expressed in {f}.  Static or slow motion only.
//
// Python -> C++: ValueError -> std::invalid_argument; a tuple -> std::pair or a small struct (fields named as in
// the Python docstring); a 6-vector -> sbeg::Vector6d; a list of 4x4 poses -> std::vector<Eigen::Matrix4d>;
// a K x 6 array of wrenches -> Eigen::MatrixXd (one wrench per ROW); None -> an empty std::optional.
#ifndef SBEG361_LAB4_HPP
#define SBEG361_LAB4_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/labio.hpp>  // sbeg::json for load_json

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using sbeg::Matrix6d;
using sbeg::Vector6d;

constexpr double TOL = 1e-6;  ///< same threshold as modern_robotics.NearZero
inline const Eigen::Vector3d G_S{0.0, 0.0, -9.81};  ///< gravity in {s} (m/s^2)

// ---------------------------------------------------------------------------
// Helpers from Labs 1-3 (given, complete -- you may use them)
// ---------------------------------------------------------------------------

/// [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);
/// omega from [omega].
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);
/// Rodrigues' formula (MR Eq. 3.51); input [omega]theta.
Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat);
/// Matrix logarithm on SO(3) (MR Sec. 3.2.3.3); returns [omega]theta with theta in [0, pi].
Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R);
/// T = (R, p) as a 4x4 matrix.
Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);
/// (R, p) of a 4x4 transform.
std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T);
/// T^-1 (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);
/// [V] (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const Vector6d& V);
/// V from [V].
Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);
/// [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
Matrix6d adjoint(const Eigen::Matrix4d& T);

// ---------------------------------------------------------------------------
// Part A -- exp and log on SE(3)
// ---------------------------------------------------------------------------

/// T = e^{[S]theta} for the 4x4 matrix [S]theta (MR Prop. 3.25, Eqs. 3.88-3.89).
/// Works for any element of se(3): theta = |omega theta| if omega != 0, otherwise the pure translation case
/// (Eq. 3.89).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

/// [S]theta = log(T) (MR Sec. 3.3.3.2 algorithm, Eqs. 3.91-3.92), theta in [0, pi].
Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T);

/// Unit screw axis S and distance theta that carry T1 to T2 (returned as the pair (S, theta)).
///   frame = "space":  T2 = e^{[S]theta} T1   (S expressed in {s}; log of T2 T1^-1)
///   frame = "body":   T2 = T1 e^{[S]theta}   (S expressed in the moving frame at T1; log of T1^-1 T2)
/// S is normalised as in MR Def. 3.24: |omega| = 1, or omega = 0 and |v| = 1.  For T1 = T2 the function
/// returns (zeros(6), 0.0).  Throws std::invalid_argument for an unknown frame.
std::pair<Vector6d, double> screw_between(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2,
                                          const std::string& frame = "space");

/// Pose a fraction s in [0, 1] of the way along the constant screw motion from T1 to T2:
/// T(s) = e^{[S] s theta} T1 (MR Sec. 3.3.3; this is the straight line in exponential coordinates).
Eigen::Matrix4d screw_path(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2, double s);

// ---------------------------------------------------------------------------
// Part B -- wrenches
// ---------------------------------------------------------------------------

/// Wrench F = (r x f, f) of a force f acting at point r, both in the same frame (MR Eq. 3.93).
Vector6d point_force_wrench(const Eigen::Vector3d& f, const Eigen::Vector3d& r);

/// F_b = [Ad_{T_ab}]^T F_a (MR Eq. 3.95 / Prop. 3.27): the same wrench, represented in {b}.
Vector6d change_wrench_frame(const Vector6d& F_a, const Eigen::Matrix4d& T_ab);

/// Instantaneous power V^T F (MR Eq. 3.94); V and F must be expressed in the same frame.
double power(const Vector6d& V, const Vector6d& F);

/// Screw (Poinsot) coordinates of a wrench (Python: the tuple (h, q, s_hat, magnitude)).
struct WrenchScrew {
    double h = 0.0;                    ///< pitch (m); +infinity for a pure moment
    std::optional<Eigen::Vector3d> q;  ///< point of the axis closest to the origin; empty (Python None) if f = 0
    Eigen::Vector3d s_hat = Eigen::Vector3d::Zero();  ///< unit direction of the axis
    double magnitude = 0.0;            ///< |f|, or |m| for a pure moment
};

/// Screw (Poinsot) coordinates of a wrench F = (m, f) (MLS Thm. 2.17, Eqs. 2.69-2.71, p. 65).
///   f != 0: h = m.f/|f|^2, q = f x m/|f|^2 (point of the axis closest to the origin), s_hat = f/|f|,
///           magnitude |f|.  Check: m = q x f + h f (MLS Eq. 2.68 with q x w = -w x q).
///   f == 0: pure moment, h = inf (std::numeric_limits<double>::infinity()), q empty, s_hat = m/|m|,
///           magnitude |m|.
/// Throws std::invalid_argument for F = 0.
WrenchScrew wrench_to_screw(const Vector6d& F);

/// Gravity wrench on a body of mass `mass` whose centre of mass is at r_com in {f}, expressed in {f}, when the
/// orientation of {f} in {s} is R_sf and gravity is g_s in {s}:  f_f = mass R_sf^T g_s,  m_f = r_com x f_f.
Vector6d gravity_wrench(double mass, const Eigen::Vector3d& r_com, const Eigen::Matrix3d& R_sf,
                        const Eigen::Vector3d& g_s = G_S);

/// Result of estimate_payload (Python: the tuple (mass, r_com, bias)).
struct PayloadEstimate {
    double mass = 0.0;                                  ///< kg
    Eigen::Vector3d r_com = Eigen::Vector3d::Zero();    ///< centre of mass in {f} (m)
    Vector6d bias = Vector6d::Zero();                   ///< sensor bias (b_m, b_f)
};

/// Least-squares mass, centre of mass and sensor bias from K >= 3 static readings.
///
/// Model for reading k (sensor orientation R_k = R_sf,k; F_list holds one wrench per ROW, K x 6):
///     F_k = b + ( c x (R_k^T g_s),  mass R_k^T g_s ),   c = mass * r_com
/// which is LINEAR in x = (mass, c, b) (10 unknowns):
///     f-rows:  f_k = b_f + mass gamma_k            (gamma_k = R_k^T g_s)
///     m-rows:  m_k = b_m - [gamma_k] c
/// Returns {mass, r_com, bias} with bias = (b_m, b_f).  Use as  auto [m, r, b] = lab::estimate_payload(Rs, Fs);
/// Throws std::invalid_argument if R_list.size() != F_list.rows() or F_list does not have 6 columns.
PayloadEstimate estimate_payload(const std::vector<Eigen::Matrix3d>& R_list, const Eigen::MatrixXd& F_list,
                                 const Eigen::Vector3d& g_s = G_S);

/// Contact wrench in {f}: reading minus bias minus the tool's own gravity wrench.
Vector6d compensate(const Vector6d& F_meas, double mass, const Eigen::Vector3d& r_com, const Eigen::Matrix3d& R_sf,
                    const Vector6d& bias, const Eigen::Vector3d& g_s = G_S);

// ---------------------------------------------------------------------------
// Loaders and analysis (given)
// ---------------------------------------------------------------------------

/// Contents of a pose-wrench CSV file (Python: the tuple (t or None, Ts, Fs)).
struct PoseWrenchData {
    Eigen::VectorXd t;                  ///< time stamps (s); size 0 if the file has no "t" column (Python None)
    std::vector<Eigen::Matrix4d> Ts;    ///< T_sf of every row
    Eigen::MatrixXd Fs;                 ///< K x 6 wrenches (mx, my, mz, fx, fy, fz), one per row
};

/// Read a CSV with columns [t,] R11..R33, px, py, pz, mx, my, mz, fx, fy, fz.
PoseWrenchData load_pose_wrench_csv(const std::string& path);

/// Read a JSON file (nlohmann::json; e.g. sbeg::as<Eigen::Matrix4d>(d["lift"]["T_sf"]) gives a 4x4 matrix).
sbeg::json load_json(const std::string& path);

/// Compensate every reading and express it at the tool tip: returns the K x 6 tip wrenches F_t (one per row).
Eigen::MatrixXd tip_forces(const std::vector<Eigen::Matrix4d>& Ts, const Eigen::MatrixXd& Fs, double mass,
                           const Eigen::Vector3d& r_com, const Vector6d& bias, const Eigen::Matrix4d& T_ft,
                           const Eigen::Vector3d& g_s = G_S);

}  // namespace lab

#endif  // SBEG361_LAB4_HPP
