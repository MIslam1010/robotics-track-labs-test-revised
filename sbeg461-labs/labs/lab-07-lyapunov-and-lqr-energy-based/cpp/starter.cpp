// SBEG 461 Lab 7 -- Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 7 --cpp
//
// Provided (helpers.cpp, do not edit): load_config, plant_2r, plant_course_arm, scaled_model, mass_matrix,
// coriolis_vector, gravity_torque, potential_energy, forward_dynamics, pd_gains_from_inertia, is_hurwitz, time_grid,
// rk4_step, simulate, and care_solve (the Riccati solver; in Python scipy.linalg.solve_continuous_are).
// Robot fields: robot.n, robot.b (friction vector), robot.q_home, robot.tau_max, robot.Mlist, robot.Glist, ...
// Eigen reminders: Eigen::VectorXd::Zero(n), Eigen::MatrixXd::Identity(n, n); x.dot(y); A.transpose();
//   A.partialPivLu().solve(b) (= np.linalg.solve); A.inverse(); v.asDiagonal(); v.cwiseProduct(w), v.cwiseInverse();
//   blocks: A.topRightCorner(n, n), A.bottomLeftCorner(n, n), A.bottomRows(n); stacking: Eigen::VectorXd x(2 * n);
//   x << a, b;  M.row(k).transpose() is sample k as a column vector; Frobenius norm A.norm().
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lab {

// ==========================================================================================
// Part A -- passivity and the Lyapunov function of PD + gravity compensation
// ==========================================================================================

std::vector<Eigen::MatrixXd> mass_matrix_partials(const Robot& robot, const Eigen::VectorXd& q, double h) {
    // TODO: for each k: e = h * unit vector k; dM[k] = (mass_matrix(robot, q + e) - mass_matrix(robot, q - e)) / (2 h)
    // TODO: return a std::vector of n matrices (push_back in the order k = 0, 1, ..., n - 1)
    throw std::logic_error("TODO: mass_matrix_partials");
}

Eigen::MatrixXd coriolis_matrix(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq, double h) {
    // TODO: dM = mass_matrix_partials(robot, q, h); dM[k](i, j) = d m_ij / d q_k
    // TODO: Gamma_ijk = 0.5 * (dM[k](i, j) + dM[j](i, k) - dM[i](j, k))   (three nested loops)
    // TODO: C(i, j) = sum_k Gamma_ijk dq(k)
    throw std::logic_error("TODO: coriolis_matrix");
}

Eigen::MatrixXd mass_matrix_dot(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq, double h) {
    // TODO: sum over k of mass_matrix_partials(...)[k] * dq(k)
    throw std::logic_error("TODO: mass_matrix_dot");
}

double stored_energy(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    // TODO: kinetic 1/2 dq^T M dq (mass_matrix) + potential_energy(robot, q) (provided)
    throw std::logic_error("TODO: stored_energy");
}

Eigen::VectorXd pd_gravity_setpoint_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                           const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp,
                                           const Eigen::MatrixXd& Kd, const Robot& model) {
    // TODO: Kp * (qd - q) - Kd * dq + gravity_torque(model, q)   (the MODEL's gravity, at the measured q)
    throw std::logic_error("TODO: pd_gravity_setpoint_torque");
}

double lyapunov_energy(const Robot& plant, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                       const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp) {
    // TODO: e = qd - q;  0.5 e^T Kp e + 0.5 dq^T M(q) dq with mass_matrix(plant, q)
    throw std::logic_error("TODO: lyapunov_energy");
}

double lyapunov_rate(const Robot& plant, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                     const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp, const Eigen::VectorXd& tau) {
    // TODO: ddq = forward_dynamics(plant, q, dq, tau)
    // TODO: -dq^T Kp e + dq^T M ddq + 0.5 dq^T mass_matrix_dot(plant, q, dq) dq
    throw std::logic_error("TODO: lyapunov_rate");
}

Eigen::VectorXd equilibrium_offset(const Robot& plant, const Robot& model, const Eigen::VectorXd& qd,
                                   const Eigen::MatrixXd& Kp, int iters, double tol, double h) {
    // TODO: F(q) = Kp * (qd - q) + gravity_torque(model, q) - gravity_torque(plant, q)   (a lambda is convenient)
    // TODO: Newton: q = qd; repeat iters times: if |F(q)| < tol return q; J.col(k) = (F(q + h e_k) - F(q - h e_k)) / (2h);
    // TODO:         q = q - J.partialPivLu().solve(F(q));  after the loop: throw std::invalid_argument if |F(q)| > 1e-9
    throw std::logic_error("TODO: equilibrium_offset");
}

// ==========================================================================================
// Part B -- LQR for the linearised arm
// ==========================================================================================

Linearization linearize(const Robot& plant, const Eigen::VectorXd& q_eq, double h) {
    // TODO: G.col(k) = (gravity_torque(plant, q_eq + h e_k) - gravity_torque(plant, q_eq - h e_k)) / (2h)
    // TODO: Minv = mass_matrix(plant, q_eq).inverse();  A = [[0, I], [-Minv G, -Minv diag(b)]],  B = [[0], [Minv]]
    // TODO: return {A, B, gravity_torque(plant, q_eq)}  (Linearization fields A, B, u_eq)
    throw std::logic_error("TODO: linearize");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bryson_weights(const Eigen::VectorXd& x_max, const Eigen::VectorXd& u_max) {
    // TODO: {diag(1 / x_max^2), diag(1 / u_max^2)}  (Eigen::MatrixXd(v.asDiagonal()))
    throw std::logic_error("TODO: bryson_weights");
}

Eigen::MatrixXd lyapunov_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Q) {
    // TODO: build the n^2 x n^2 matrix L = I kron A^T + A^T kron I for the COLUMN-stacked vec(P)
    //       (vec index of P(i, j) is j n + i; (A^T P)_ij = sum_k A(k, i) P(k, j), (P A)_ij = sum_k P(i, k) A(k, j))
    // TODO: solve L vec(P) = -vec(Q), unstack, return 0.5 (P + P^T)
    throw std::logic_error("TODO: lyapunov_solve");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> lqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
    // TODO: P = care_solve(A, B, Q, R) (provided; symmetrise it);  K = R^-1 B^T P (R.partialPivLu().solve(...))
    // TODO: return {K, P}
    throw std::logic_error("TODO: lqr");
}

Eigen::MatrixXd pd_state_feedback(const Robot& model, const Eigen::VectorXd& q_eq, const Eigen::MatrixXd& Kp,
                                  const Eigen::MatrixXd& Kd, double h) {
    // TODO: G = dg~/dq at q_eq by central differences (gravity_torque(model, ...)), as in linearize
    // TODO: K0 (n x 2n) = [Kp - G, Kd]   (K0 << Kp - G, Kd;)
    throw std::logic_error("TODO: pd_state_feedback");
}

KleinmanResult kleinman_lqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                            const Eigen::MatrixXd& R, const Eigen::MatrixXd& K0, int iters, double tol) {
    // TODO: if !is_hurwitz(A - B K0): throw std::invalid_argument
    // TODO: for k = 1..iters: P = lyapunov_solve(A - B K, Q + K^T R K); K = R^-1 B^T P;
    //       stop (return {K, P, k}) when (P - P_old).norm() <= tol * max(1, P.norm())
    throw std::logic_error("TODO: kleinman_lqr");
}

double lqr_cost(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& K, const Eigen::VectorXd& x0) {
    // TODO: Acl = A - B K; throw std::invalid_argument if !is_hurwitz(Acl)
    // TODO: P_K = lyapunov_solve(Acl, Q + K^T R K);  return x0^T P_K x0
    throw std::logic_error("TODO: lqr_cost");
}

Eigen::VectorXd state_feedback_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& q_eq,
                                      const Eigen::VectorXd& u_eq, const Eigen::MatrixXd& K) {
    // TODO: x = (q - q_eq, dq);  return u_eq - K x
    throw std::logic_error("TODO: state_feedback_torque");
}

double run_cost(const SimResult& sim, const Eigen::VectorXd& q_eq, const Eigen::VectorXd& u_eq,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
    // TODO: for each sample k: x_k = (sim.q.row(k)^T - q_eq, sim.dq.row(k)^T), u_k = sim.tau.row(k)^T - u_eq,
    //       w_k = x_k^T Q x_k + u_k^T R u_k;  trapezoid: sum 0.5 (w_k + w_{k+1}) (t_{k+1} - t_k)
    throw std::logic_error("TODO: run_cost");
}

}  // namespace lab
