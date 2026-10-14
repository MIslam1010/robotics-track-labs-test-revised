// SBEG 361 Lab 5 -- Forward kinematics of the course arm by the product of exponentials (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. The complete helpers (Labs 1-3 helpers, URDF reader, log and offset readers) are
// in helpers.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// You may NOT call mrcpp (the C++ port of modern_robotics) or Eigen's .inverse() / .exp() inside the functions you
// write -- they are what the tests compare against. The helpers below may be used freely.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Chapter 4:
//   space form   T(theta) = e^[S1]th1 ... e^[Sn]thn M        (MR Eq. 4.14)
//   body form    T(theta) = M e^[B1]th1 ... e^[Bn]thn        (MR Eq. 4.16), B_i = [Ad_{M^-1}] S_i
//   revolute     S = (omega, -omega x q), |omega| = 1;  prismatic  S = (0, v), |v| = 1   (MR Sec. 4.1.1)
//   exp on se(3) MR Prop. 3.25, Eqs. 3.88-3.89
//   URDF         joint origin = pose of the child-link frame in the parent-link frame at q = 0;
//                rpy = roll about x, pitch about y, yaw about z, all about FIXED axes:
//                R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll)   (MR Sec. 4.2)
// Hardware log format (identical to the project's hwcompare.py):
//   t, cmd_1..cmd_n, meas_1..meas_n[, stop]   -- t in s, angles in SERVO DEGREES.
//   Model angles:  q_model = q_servo - servo_zero_offset   (course convention; offsets in data/servo_offsets.json).
//
// Python -> C++: ValueError -> std::invalid_argument; a screw axis (6-vector) -> sbeg::Vector6d; Slist / Blist
// (6 x n, one axis per COLUMN) -> Eigen::MatrixXd; theta -> Eigen::VectorXd; an (N, n) array of joint vectors ->
// Eigen::MatrixXd with one ROW per sample; a dict / tuple result -> a small struct with the same field names;
// a keyword argument that may be None -> std::optional (std::nullopt = None).
#ifndef SBEG361_LAB5_HPP
#define SBEG361_LAB5_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

constexpr double TOL = 1e-6;  ///< same threshold as modern_robotics.NearZero

// ---------------------------------------------------------------------------
// Helpers from Labs 1-3 (complete, in helpers.cpp)
// ---------------------------------------------------------------------------

/// [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// [V] = [[omega] v; 0 0] (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V);

/// T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);

/// Rotation about a coordinate axis "x", "y" or "z" (MR Eq. 3.16 family).
Eigen::Matrix3d rot_axis(const std::string& axis, double angle);

// ---------------------------------------------------------------------------
// Step 1 -- screw axes and exp
// ---------------------------------------------------------------------------

/// Screw axis of a one-DOF joint at the zero configuration (MR Sec. 4.1.1).
/// Revolute: give the unit axis `omega` and any point `q` on the axis -> S = (omega, -omega x q).
/// Prismatic: give only the unit direction `v` -> S = (0, v):  screw_axis(std::nullopt, std::nullopt, v).
/// Throws std::invalid_argument if the direction is not a unit vector or the arguments are inconsistent.
sbeg::Vector6d screw_axis(const std::optional<Eigen::Vector3d>& omega = std::nullopt,
                          const std::optional<Eigen::Vector3d>& q = std::nullopt,
                          const std::optional<Eigen::Vector3d>& v = std::nullopt);

/// e^{[S] theta} for a screw axis S (MR Prop. 3.25, Eq. 3.88; Eq. 3.89 if omega = 0).
/// Same result as Lab 3's matrix_exp6(vec_to_se3(S) * theta); here the axis and the angle are passed separately.
Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta);

// ---------------------------------------------------------------------------
// Step 2 -- PoE forward kinematics
// ---------------------------------------------------------------------------

/// Space-form PoE: T = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14). Slist is 6 x n.
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body-form PoE: T = M e^[B1]th1 ... e^[Bn]thn (MR Eq. 4.16). Blist is 6 x n.
Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// B_i = [Ad_{M^-1}] S_i for every column (MR Eq. 4.16).
Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M);

/// (M, Slist) of the course arm from its geometry (UR-type layout of MR Example 4.5).
///
/// Zero configuration: arm stretched along +x_s, {s} on the joint-1 axis at the base, z_s up.
/// Joint axes at home (direction, one point on the axis):
///   1: +z through (0, 0, 0)            2: +y through (0, 0, H1)
///   3: +y through (L1, 0, H1)          4: +y through (L1+L2, 0, H1)
///   5: -z through (L1+L2, W1, 0)       6: +y through (L1+L2, 0, H1-H2)
/// {b} at home: origin (L1+L2, W1+W2, H1-H2), x_b = -x_s, y_b = z_s, z_b = y_s;
/// a straight tool of length `tool` moves {b} along its own z_b.
/// Returns the pair (M, Slist):  auto [M, S] = lab::course_arm_screws(...);
std::pair<Eigen::Matrix4d, Eigen::MatrixXd> course_arm_screws(double W1, double W2, double L1, double L2, double H1,
                                                              double H2, double tool = 0.0);

// ---------------------------------------------------------------------------
// Step 3 -- URDF -> PoE
// ---------------------------------------------------------------------------

/// One <joint> element of a URDF file (Python: a dict with the same keys).
struct UrdfJoint {
    std::string name, type, parent, child;
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();     ///< <origin xyz>, default zero
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();     ///< <origin rpy>, default zero
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();   ///< <axis xyz>, default (1, 0, 0)
    std::optional<std::pair<double, double>> limit;    ///< (lower, upper), or std::nullopt (Python None)
};

/// Parse the <joint> elements of a URDF file (complete helper, in helpers.cpp).
/// Missing <origin>/<axis> take the URDF defaults (zero, and (1, 0, 0)).
std::vector<UrdfJoint> read_urdf_joints(const std::string& path);

/// URDF roll-pitch-yaw (fixed axes x, y, z): R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll).  rpy = (roll, pitch, yaw).
Eigen::Matrix3d rpy_to_rot(const Eigen::Vector3d& rpy);

/// Joints from the root link to `tip_link`, in order from the base (complete helper, in helpers.cpp).
/// Throws std::invalid_argument if no joint leads to `tip_link`.
std::vector<UrdfJoint> urdf_chain(const std::vector<UrdfJoint>& joints, const std::string& tip_link);

/// Result of poe_from_urdf (Python: the tuple (M, Slist, names, limits)).
struct PoeModel {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();  ///< T_s,tip at zero
    Eigen::MatrixXd Slist;                            ///< 6 x n
    std::vector<std::string> names;                   ///< n joint names
    Eigen::MatrixXd limits;                           ///< n x 2 (lower, upper), +-infinity where the URDF has none
};

/// Space-form PoE model of the chain that ends at `tip_link`.
///
/// Walk the chain from the base with all joint values zero, composing the joint origins:
/// T_s,child = T_s,parent * (rpy_to_rot(rpy), xyz).  For each movable joint the axis in {s} is
/// omega = R_s,child * axis (axis is given in the child frame), the joint frame origin lies on the
/// axis, so S = (omega, -omega x p) (revolute/continuous) or (0, R axis) (prismatic).
/// Fixed joints only move the frame.  M = T_s,tip at zero.
/// Throws std::invalid_argument for any other joint type.
PoeModel poe_from_urdf(const std::vector<UrdfJoint>& joints, const std::string& tip_link);

// ---------------------------------------------------------------------------
// Step 4 -- simulation vs hardware
// ---------------------------------------------------------------------------

/// A joint log (Python: dict with keys t, cmd, meas, stop). Angles in servo degrees.
struct JointLog {
    Eigen::VectorXd t;     ///< N time stamps (s)
    Eigen::MatrixXd cmd;   ///< N x n commanded angles
    Eigen::MatrixXd meas;  ///< N x n measured angles
    Eigen::VectorXd stop;  ///< N stop flags; EMPTY (size 0) if the log has no stop column (Python None)
};

/// Read a joint log 't, cmd_i, meas_i[, stop]' (servo degrees) (complete helper, in helpers.cpp).
JointLog load_joint_log(const std::string& path);

/// Servo zero offsets in degrees from a calibration JSON file (complete helper, in helpers.cpp).
Eigen::VectorXd load_offsets(const std::string& path);

/// Model joint angles in RADIANS: q_model = deg2rad(q_servo - offset), for one joint vector [n] ...
Eigen::VectorXd servo_to_model(const Eigen::VectorXd& q_servo_deg, const Eigen::VectorXd& offsets_deg);

/// ... and for N joint vectors, one per ROW [N x n] (the offset is subtracted from every row).
Eigen::MatrixXd servo_to_model(const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg);

/// Positions in {s} (m) of the tool point p_tool (given in {b}) for each row of Q (radians). Returns N x 3.
Eigen::MatrixXd tool_points(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                            const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

/// Result of compare_log (Python: a dict with the same keys).
struct LogComparison {
    Eigen::VectorXd joint_rms_deg;   ///< [n] RMS of e = meas - cmd (degrees)
    Eigen::VectorXd joint_max_deg;   ///< [n] max |e|
    Eigen::VectorXd joint_mean_deg;  ///< [n] mean of e
    Eigen::VectorXd task_err_mm;     ///< [N'] |p(meas) - p(cmd)| of the tool point
    double task_rms_mm = 0.0;
    double task_max_mm = 0.0;
    int n_samples = 0;               ///< N' (samples used)
};

/// Commanded vs measured motion, using only the samples BEFORE the stop command (all samples if there is no
/// stop column or no stop = 1 sample).
LogComparison compare_log(const JointLog& log, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                          const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

/// Distance (mm) between the FK-predicted tool point and the externally measured one (rows of p_meas_mm, N x 3),
/// per pose. Returns [N].
Eigen::VectorXd tip_errors_mm(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                              const Eigen::MatrixXd& p_meas_mm,
                              const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

/// Zero-offset error delta (degrees) of ONE joint (0-based index) that best explains the measured tool points:
/// minimise sum_k |p(q_k + delta e_joint) - p_meas_k|^2 over |delta| <= search_deg (golden-section search).
/// Returns (delta_deg, rms_after_mm).
std::pair<double, double> fit_joint_offset(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& p_meas_mm, int joint,
                                           const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero(),
                                           double search_deg = 5.0, double tol_deg = 1e-4);

}  // namespace lab

#endif  // SBEG361_LAB5_HPP
