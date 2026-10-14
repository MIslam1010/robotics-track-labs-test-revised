// SBEG 461 Lab 6 -- computed torque vs PID with model error on the course arm; task-space control (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 6 --cpp
//
// Provided (helpers.cpp, do not edit): load_config, arm_model, make_case, tau_limits, time_grid, quintic_reference,
// cartesian_reference, rk4_step, diag_gain (Python _diag), Plant, simulate_arm, and the constructors of
// JointController and TaskController.  As in Python you may use mrcpp (the C++ port of modern_robotics):
// mr::inverse_dynamics, mr::mass_matrix, mr::gravity_forces, mr::jacobian_body, mr::fkin_body, mr::trans_inv,
// mr::matrix_log6, mr::se3_to_vec, mr::adjoint, mr::ad.
//
// Model fields: model.Mlist, model.Glist, model.Slist, model.Blist, model.M, model.g, model.b (friction vector).
// Eigen reminders: element-wise product a.cwiseProduct(b), quotient a.cwiseQuotient(b); M.diagonal();
//   M.row(k).transpose() is sample k as a column vector; (qd - q).rowwise().norm(); tau.cwiseAbs().colwise().maxCoeff();
//   J.partialPivLu().solve(b) (= np.linalg.solve); J.inverse(); mr::Vector6d::Zero(); Eigen::VectorXd::Zero(n).
//   Gains: diag_gain(K, n) turns a scalar or vector gain into a length-n vector (as _diag in Python).
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>

namespace lab {

// ==========================================================================================
// Part A -- gains
// ==========================================================================================

std::pair<double, double> ct_gains(double zeta, double wn) {
    // TODO: kp = wn^2, kd = 2 zeta wn;  return {kp, kd};
    throw std::logic_error("TODO: ct_gains");
}

double ct_ki_max(double kp, double kd) {
    // TODO: Routh condition for s^3 + kd s^2 + kp s + ki
    throw std::logic_error("TODO: ct_ki_max");
}

PIDGains decentralized_gains(const Robot& model, const Eigen::VectorXd& q_ref, double zeta, double wn,
                             double ki_fraction) {
    // TODO: diagonal of mr::mass_matrix at q_ref, then the Lab 5 pd_gains rule per joint;  return {Kp, Ki, Kd};
    throw std::logic_error("TODO: decentralized_gains");
}

// ==========================================================================================
// Part B -- joint-space control laws
// ==========================================================================================

Eigen::VectorXd model_inverse_dynamics(const Robot& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                       const Eigen::VectorXd& ddq) {
    // TODO: mr::inverse_dynamics of the model (Ftip = mr::Vector6d::Zero()) plus model.b * dq (element-wise)
    throw std::logic_error("TODO: model_inverse_dynamics");
}

Eigen::VectorXd pid_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                           const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Gain& Kp, const Gain& Ki,
                           const Gain& Kd) {
    // TODO: e = qd - q, e_dot = dqd - dq; use diag_gain() so scalar and vector gains both work
    throw std::logic_error("TODO: pid_torque");
}

Eigen::VectorXd pd_gravity_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                                  const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Gain& Kp,
                                  const Gain& Ki, const Gain& Kd, const Robot& model) {
    // TODO: pid_torque(...) + mr::gravity_forces of the MODEL at the measured q
    throw std::logic_error("TODO: pd_gravity_torque");
}

Eigen::VectorXd feedforward_pd_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                      const Eigen::VectorXd& eint, const Eigen::VectorXd& qd,
                                      const Eigen::VectorXd& dqd, const Eigen::VectorXd& ddqd, const Gain& Kp,
                                      const Gain& Ki, const Gain& Kd, const Robot& model) {
    // TODO: model_inverse_dynamics at the REFERENCE (qd, dqd, ddqd) + pid_torque(...)
    throw std::logic_error("TODO: feedforward_pd_torque");
}

Eigen::VectorXd computed_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                                const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Eigen::VectorXd& ddqd,
                                const Gain& kp, const Gain& ki, const Gain& kd, const Robot& model) {
    // TODO: commanded acceleration a = ddqd + Kp e + Ki eint + Kd e_dot, then model_inverse_dynamics(model, q, dq, a)
    throw std::logic_error("TODO: computed_torque");
}

double error_energy(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& qd, const Gain& Kp,
                    const Robot& model) {
    // TODO: 0.5 e^T Kp e + 0.5 dq^T M(q) dq with mr::mass_matrix of the model
    throw std::logic_error("TODO: error_energy");
}

void JointController::reset() {
    // TODO: set the integral state eint to zeros (length = number of joints = qd.cols())
    throw std::logic_error("TODO: JointController::reset");
}

Eigen::VectorXd JointController::operator()(int k, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    // TODO: update the integral (as in MR Fig. 11.20), then call the control law named by `law`
    //       (sample k of the reference: qd.row(k).transpose(); the model: *model)
    throw std::logic_error("TODO: JointController::operator()");
}

Metrics run_metrics(const Eigen::VectorXd& t, const Eigen::MatrixXd& qd, const Eigen::MatrixXd& q,
                    const Eigen::MatrixXd& tau) {
    // TODO: norms over joints with .rowwise().norm(); trapezoidal rule for the effort
    throw std::logic_error("TODO: run_metrics");
}

// ==========================================================================================
// Part C -- task-space computed torque (MR Sec. 11.4.3)
// ==========================================================================================

Eigen::MatrixXd body_jacobian_dot(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    // TODO: loop over columns i and j > i; mr::ad(J.col(i)) * J.col(j) * dq(j)
    throw std::logic_error("TODO: body_jacobian_dot");
}

std::pair<Vector6d, Vector6d> task_errors(const Eigen::Matrix4d& X, const Vector6d& V, const Eigen::Matrix4d& Xd,
                                          const Vector6d& Vd) {
    // TODO: mr::trans_inv, mr::matrix_log6, mr::se3_to_vec, mr::adjoint;  return {Xe, Ve};
    throw std::logic_error("TODO: task_errors");
}

Vector6d task_feedforward_accel(const Eigen::Matrix4d& X, const Vector6d& V, const Eigen::Matrix4d& Xd,
                                const Vector6d& Vd, const Vector6d& dVd) {
    // TODO: A = X^-1 Xd; W = Ad_A Vd; return Ad_A dVd - ad_V W
    throw std::logic_error("TODO: task_feedforward_accel");
}

std::pair<Matrix6d, Vector6d> model_task_space_dynamics(const Robot& model, const Eigen::VectorXd& q,
                                                        const Eigen::VectorXd& dq) {
    // TODO: J = mr::jacobian_body, M = mr::mass_matrix, h = model_inverse_dynamics(model, q, dq, 0);
    //       return {Lambda, eta};
    throw std::logic_error("TODO: model_task_space_dynamics");
}

Eigen::VectorXd task_space_ct(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Vector6d& Xint,
                              const Eigen::Matrix4d& Xd, const Vector6d& Vd, const Vector6d& dVd, const Gain& kp,
                              const Gain& ki, const Gain& kd, const Robot& model) {
    // TODO: X, J, V; errors; a; joint acceleration J.partialPivLu().solve(a - Jdot dq); one RNEA call
    throw std::logic_error("TODO: task_space_ct");
}

void TaskController::reset() {
    // TODO: set the integral state Xint to zeros
    throw std::logic_error("TODO: TaskController::reset");
}

Eigen::VectorXd TaskController::operator()(int k, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    // TODO: update the integral (as in MR Fig. 11.20), then call the control law task_space_ct
    //       (sample k: Xd[k], Vd.row(k).transpose(), dVd.row(k).transpose())
    throw std::logic_error("TODO: TaskController::operator()");
}

}  // namespace lab
