// SBEG 46X Lab 1 -- Software remote centre of motion (RCM): constrained resolved-rate control of the course arm
// holding a laparoscope (C++ STARTER).
//
// Fill in every function marked  // TODO  (13 functions; replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 1 --cpp
// The provided helpers (load_task, skew, min_jerk, port_frame, make_trajectory, save_log_csv, environment_report)
// are in helpers.cpp and the robot model in course_models.cpp: use them, do not edit them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR); see lab.hpp for the equations of the software RCM.
//
// Eigen reminders:  T.block<3, 1>(0, 3) (position column of T), T.block<3, 1>(0, 2) (z_b), a.dot(b), a.cross(b),
//   v.norm(), A.transpose(), A * B (matrix product), A.inverse(), Eigen::Matrix3d::Identity(),
//   Eigen::MatrixXd::Identity(n, n), v.cwiseAbs(), (a.array() / b.array()).maxCoeff().
//   Stack blocks with the comma initialiser:  Eigen::MatrixXd A(5, 6);  A << top_3x6, bottom_2x6;
//   Columns side by side:  Matrix32d N;  N << n1, n2;
//   Never write `auto C = A * B;` (auto keeps an unevaluated expression): write the type, Eigen::MatrixXd C = A * B;
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>  // the kinematics backend below and mr::pinv (= np.linalg.pinv)

#include <cmath>
#include <stdexcept>
#include <string>

namespace lab {

// ---------------------------------------------------------------------------
// Kinematics backend (given)
// FK and space Jacobians: mrcpp, the equivalent of the Python modern_robotics fallback.  If you did the SBEG 361
// labs in C++, replace the two bodies by your own functions (361 Lab 5 FK, 361 Lab 7 space Jacobian) and change
// the name returned by kinematics_backend(); test_361_library_matches_modern_robotics checks them.
// ---------------------------------------------------------------------------

std::string kinematics_backend() { return "mrcpp (fallback)"; }

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    return mr::fkin_space(M, Slist, theta);  // fallback: mrcpp  (MR Eq. 4.14)
}

Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    return mr::jacobian_space(Slist, theta);  // fallback: mrcpp  (MR Eq. 5.11)
}

// ---------------------------------------------------------------------------
// Part A: geometry of the tool and the port
// ---------------------------------------------------------------------------

Eigen::Matrix4d tool_pose(const Robot& robot, const Eigen::VectorXd& theta) {
    // TODO: call fk_space(M, Slist, theta) from the kinematics backend; return the 4 x 4 matrix.
    throw std::logic_error("TODO: tool_pose");
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> shaft(const Eigen::Matrix4d& T) {
    // TODO: p_tip is the position column of T, z is the third column of R (z_b).  return {p_tip, z};
    throw std::logic_error("TODO: shaft");
}

RcmGeometry rcm_geometry(const Eigen::Matrix4d& T, const Eigen::Vector3d& p_port) {
    // TODO: s = z . (p_tip - p_port);  p_c = p_tip - s z;  e = p_port - p_c;  dist = |e|.
    //       Fill an RcmGeometry g (g.depth = s, g.p_c, g.e, g.dist) and return it.
    throw std::logic_error("TODO: rcm_geometry");
}

Matrix32d perp_basis(const Eigen::Vector3d& z) {
    // TODO: pick a helper vector a not parallel to z, n1 = a x z / |a x z|, n2 = z x n1.
    throw std::logic_error("TODO: perp_basis");
}

// ---------------------------------------------------------------------------
// Part B: Jacobians
// ---------------------------------------------------------------------------

Eigen::MatrixXd point_jacobian(const Eigen::MatrixXd& Js, const Eigen::Vector3d& p) {
    // TODO: pdot = v_s + w_s x p = -[p] w_s + v_s, so J_p = [-[p]  I] J_s (use skew()).
    throw std::logic_error("TODO: point_jacobian");
}

std::pair<Eigen::MatrixXd, RcmInfo> rcm_jacobian(const Robot& robot, const Eigen::VectorXd& theta,
                                                 const Eigen::Vector3d& p_port) {
    // TODO: J_s from jacobian_space; stack point_jacobian at p_tip (3 rows) and N^T point_jacobian at p_c (2 rows).
    //       Fill an RcmInfo (info.T, info.N, info.geometry) and return {A, info}.
    throw std::logic_error("TODO: rcm_jacobian");
}

Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& A, double lam) {
    // TODO: lam == 0: mr::pinv(A) (same cut-off as np.linalg.pinv); otherwise A^T (A A^T + lam^2 I)^-1.
    throw std::logic_error("TODO: dls_inverse");
}

// ---------------------------------------------------------------------------
// Part C: controllers
// ---------------------------------------------------------------------------

Eigen::VectorXd rcm_rates(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::Vector3d& p_port,
                          const Eigen::Vector3d& p_d, const Eigen::Vector3d& pdot_d, double kp, double k_rcm,
                          double lam) {
    // TODO: b = [pdot_d + kp (p_d - p_tip),  k_rcm N^T e]  (5-vector);  thetadot = dls_inverse(A, lam) * b.
    throw std::logic_error("TODO: rcm_rates");
}

Eigen::VectorXd tip_only_rates(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::Vector3d& p_d,
                               const Eigen::Vector3d& pdot_d, double kp, double lam) {
    // TODO: same as rcm_rates but with only the 3 tip rows (no RCM rows).
    throw std::logic_error("TODO: tip_only_rates");
}

Eigen::VectorXd scale_rates(const Eigen::VectorXd& qdot, const Eigen::VectorXd& qdot_max) {
    // TODO: ratio = max_i |qdot_i| / qdot_max_i; divide qdot by ratio only if ratio > 1.
    throw std::logic_error("TODO: scale_rates");
}

SimLog simulate(const Robot& robot, const Task& task, const std::string& controller, double dt, double kp,
                double k_rcm, double lam) {
    // TODO: throw std::invalid_argument for a controller other than "rcm" / "tip".
    //       auto [traj, T_total] = make_trajectory(task, p_tip0, z0) with the start pose; K = std::lround(T_total / dt);
    //       size the SimLog fields (K + 1 rows); loop k = 0..K: read traj(t), log tip error, RCM distance and depth,
    //       then (except at k == K) theta += dt * scale_rates(qdot, robot.qdot_max).
    throw std::logic_error("TODO: simulate");
}

// ---------------------------------------------------------------------------
// Part D: fulcrum effect and motion scaling
// ---------------------------------------------------------------------------

double fulcrum_ratio(double L, double depth) {
    // TODO: similar triangles about the port: tip moves depth/(L - depth) times the hand, in the opposite direction.
    throw std::logic_error("TODO: fulcrum_ratio");
}

Eigen::Vector3d scaled_tip_command(const Eigen::Vector3d& d_leader, double scale, const Eigen::Matrix3d& R_sv) {
    // TODO: return scale * R_sv * d_leader (R_sv is the identity when not given).
    throw std::logic_error("TODO: scaled_tip_command");
}

}  // namespace lab
