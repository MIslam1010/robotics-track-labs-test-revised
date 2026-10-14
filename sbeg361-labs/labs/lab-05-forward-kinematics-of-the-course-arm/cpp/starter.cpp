// SBEG 361 Lab 5 -- Forward kinematics of the course arm by the product of exponentials (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 5 --cpp
//
// You may NOT call mrcpp (the C++ port of modern_robotics) or Eigen's .inverse() inside these functions -- the
// tests compare your answers with mrcpp and with numbers computed by the Python reference. The complete helpers
// in helpers.cpp (vec_to_so3, vec_to_se3, trans_inv, adjoint, rot_axis, read_urdf_joints, urdf_chain,
// load_joint_log, load_offsets) may be used freely. These functions join your sbeg361_kinematics library.
//
// Eigen reminders:  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();   T.topLeftCorner<3, 3>() = R;
//   T.block<3, 1>(0, 3) = p;  (the position column)    S.head<3>(), S.tail<3>()  (omega and v of a 6-vector)
//   sbeg::Vector6d S;  S << w, v;   (stack two 3-vectors)    w.cross(q), w.norm(), Slist.col(i), Q.row(k)
//   A * B (matrix product), std::sin, std::cos, std::sqrt, std::abs.  No M_PI: write your own constexpr PI.
#include "lab.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Step 1 -- screw axes and exp
// ---------------------------------------------------------------------------

sbeg::Vector6d screw_axis(const std::optional<Eigen::Vector3d>& omega, const std::optional<Eigen::Vector3d>& q,
                          const std::optional<Eigen::Vector3d>& v) {
    // TODO: revolute (omega given): check |omega| = 1 (tolerance TOL) and return (omega, -omega x q);
    //       prismatic (only v given): return (0, 0, 0, v); throw std::invalid_argument for bad input
    //       (non-unit direction, omega without q, omega and v together, nothing given).
    //       `if (omega)` tests whether an optional holds a value; `*omega` is the Eigen::Vector3d inside.
    throw std::logic_error("TODO: screw_axis");
}

Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta) {
    // TODO: omega ~ 0 (norm < TOL) -> Eq. 3.89 (I, v*theta). Otherwise R by Rodrigues (Eq. 3.51) and
    //       p = G(theta) v with G from Eq. 3.87
    throw std::logic_error("TODO: screw_exp");
}

// ---------------------------------------------------------------------------
// Step 2 -- PoE forward kinematics
// ---------------------------------------------------------------------------

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    // TODO: multiply e^[S_i]theta_i from i = 1 to n (left to right), then M on the RIGHT (Eq. 4.14)
    throw std::logic_error("TODO: fk_space");
}

Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    // TODO: start from M, then multiply e^[B_i]theta_i on the RIGHT for i = 1..n (Eq. 4.16)
    throw std::logic_error("TODO: fk_body");
}

Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M) {
    // TODO: one line with adjoint() and trans_inv()
    throw std::logic_error("TODO: body_screws");
}

std::pair<Eigen::Matrix4d, Eigen::MatrixXd> course_arm_screws(double W1, double W2, double L1, double L2, double H1,
                                                              double H2, double tool) {
    // TODO: list the six (omega_i, q_i) pairs from the docstring in lab.hpp, build Slist (6 x 6) column by column
    //       with screw_axis, then M; tool: M * Trans(0, 0, tool).  return {M, Slist};
    throw std::logic_error("TODO: course_arm_screws");
}

// ---------------------------------------------------------------------------
// Step 3 -- URDF -> PoE
// ---------------------------------------------------------------------------

Eigen::Matrix3d rpy_to_rot(const Eigen::Vector3d& rpy) {
    // TODO: use rot_axis; the ORDER matters (fixed axes x, then y, then z)
    throw std::logic_error("TODO: rpy_to_rot");
}

PoeModel poe_from_urdf(const std::vector<UrdfJoint>& joints, const std::string& tip_link) {
    // TODO: loop over urdf_chain(joints, tip_link); compose T = T * (rpy_to_rot(rpy), xyz); skip "fixed" after
    //       composing; omega = R * axis (normalised). Fill out.M, out.Slist (6 x n), out.names and out.limits
    //       (n x 2; std::numeric_limits<double>::infinity() where j.limit is empty). Collect the axes in a
    //       std::vector<sbeg::Vector6d> first, then copy them into the matrix once n is known.
    throw std::logic_error("TODO: poe_from_urdf");
}

// ---------------------------------------------------------------------------
// Step 4 -- simulation vs hardware
// ---------------------------------------------------------------------------

Eigen::VectorXd servo_to_model(const Eigen::VectorXd& q_servo_deg, const Eigen::VectorXd& offsets_deg) {
    // TODO: subtract the offset (degrees), then convert to radians
    throw std::logic_error("TODO: servo_to_model");
}

Eigen::MatrixXd servo_to_model(const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg) {
    // TODO: the same for every ROW: q_servo_deg.rowwise() - offsets_deg.transpose()
    throw std::logic_error("TODO: servo_to_model (matrix)");
}

Eigen::MatrixXd tool_points(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                            const Eigen::Vector3d& p_tool) {
    // TODO: homogeneous point (p_tool, 1) as an Eigen::Vector4d; one fk_space call per row of Q
    //       (Eigen::VectorXd q = Q.row(k).transpose();); return an N x 3 matrix
    throw std::logic_error("TODO: tool_points");
}

LogComparison compare_log(const JointLog& log, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                          const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool) {
    // TODO: find k0 (first sample with stop > 0.5; log.stop.size() == 0 means no stop column), e = meas - cmd on
    //       the first k0 rows (.topRows(k0)), convert both to model radians, tool_points for both, distance in mm.
    //       Column statistics: e.array().square().colwise().sum(), e.cwiseAbs().colwise().maxCoeff(), ...
    throw std::logic_error("TODO: compare_log");
}

Eigen::VectorXd tip_errors_mm(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                              const Eigen::MatrixXd& p_meas_mm, const Eigen::Vector3d& p_tool) {
    // TODO: 1e3 * tool_points(...) minus the measured points, row-wise norm (.rowwise().norm())
    throw std::logic_error("TODO: tip_errors_mm");
}

std::pair<double, double> fit_joint_offset(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& p_meas_mm, int joint,
                                           const Eigen::Vector3d& p_tool, double search_deg, double tol_deg) {
    // TODO: cost(delta) = mean squared tip error with column `joint` of a copy of Q increased by deg2rad(delta)
    //       (a lambda: auto cost = [&](double dd) { ... };); golden-section search on [-search_deg, search_deg]
    //       until the interval is shorter than tol_deg.  return {delta_deg, rms_after_mm};
    throw std::logic_error("TODO: fit_joint_offset");
}

}  // namespace lab
