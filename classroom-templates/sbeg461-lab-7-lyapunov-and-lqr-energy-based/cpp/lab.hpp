// SBEG 461 Lab 7 -- Lyapunov stability and passivity of PD + gravity compensation; LQR (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided Python helpers (load_config, plant_2r, plant_course_arm, scaled_model, mass_matrix, coriolis_vector,
// gravity_torque, potential_energy, forward_dynamics, pd_gains_from_inertia, is_hurwitz, time_grid, rk4_step,
// simulate) are given in helpers.cpp, together with care_solve (the Riccati solver that replaces
// scipy.linalg.solve_continuous_are); the course arm is in course_arm.hpp / course_arm.cpp (do not edit them).
//
// Notation follows Modern Robotics (MR): tau = M(theta) theta'' + C(theta, theta') theta' + g(theta) (MR Sec. 8.1.2,
// Eqs. 8.18-8.19) plus viscous friction B theta' in the plant; theta_e = theta_d - theta; in code q = theta.
// "model" is the CONTROLLER's model (M~, g~); "plant" is the simulated robot.  Units: SI (s, rad, N m, J).
// Linear systems follow Astrom-Murray Sec. 6.3: x' = A x + B u, J = int (x^T Q x + u^T R u) dt (Eq. 6.26),
// algebraic Riccati equation (Eq. 6.27).
//
// Python -> C++: ValueError -> std::invalid_argument; a robot dict -> lab::Robot; a tuple -> std::pair or a small
// struct with the Python names; an (n, n, n) array -> std::vector<Eigen::MatrixXd> (element k = dM/dq_k);
// an (N, n) array -> Eigen::MatrixXd (row k = sample k); the result dict of simulate -> lab::SimResult;
// a gain matrix np.diag(...) -> Eigen::MatrixXd.
#ifndef SBEG461_LAB07_HPP
#define SBEG461_LAB07_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/labio.hpp>  // sbeg::json (the configuration file)
#include <sbeg/mrcpp.hpp>

#include "course_arm.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ==========================================================================================
// Provided (do not edit; in helpers.cpp): configuration, robots, dynamics, simulator, Riccati solver
// ==========================================================================================

/// Parameters of the closed-form planar 2R dynamics (Python robot["closed_form"]).
struct ClosedForm2R {
    double L1 = 0, L2 = 0, m1 = 0, m2 = 0, lc1 = 0, lc2 = 0, I1 = 0, I2 = 0, g0 = 0;
};

/// The Python robot dict: the MR lists (name, n, Slist, M, Mlist, Glist, g, tau_max, q_home, ... of
/// course_arm::Robot) plus key ("planar_2r" or "course_arm"), the viscous friction vector b and, for the 2R,
/// the closed-form parameters (has_closed_form = true).
struct Robot : course_arm::Robot {
    std::string key;
    Eigen::VectorXd b;           ///< viscous friction b_j (N m s/rad)
    bool has_closed_form = false;
    ClosedForm2R cf;
};

/// data/lab07_config.json (illustrative values; see make_data.py), as JSON: cfg["dt"].get<double>(),
/// sbeg::as_vec(cfg["planar_2r"]["q_eq"]), ...
sbeg::json load_config();

/// The planar 2R plant (L = 0.6, 0.4 m, uniform rods m = 2, 1 kg, gravity -y) with the configured viscous
/// friction (friction = false: b = 0).
Robot plant_2r(const sbeg::json& cfg, bool friction = true);
/// The course arm (no payload) with the configured viscous friction.
Robot plant_course_arm(const sbeg::json& cfg);
/// A copy of `robot` whose masses and inertias are multiplied by mass_scale (a wrong controller model).
Robot scaled_model(const Robot& robot, double mass_scale);

/// M(q): closed form for the planar 2R, mr::mass_matrix otherwise.
Eigen::MatrixXd mass_matrix(const Robot& robot, const Eigen::VectorXd& q);
/// c(q, dq) = C(q, dq) dq: closed form for the 2R, mr::vel_quadratic_forces otherwise.
Eigen::VectorXd coriolis_vector(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq);
/// g(q) = dP/dq: closed form for the 2R (gravity along -y), mr::gravity_forces otherwise.
Eigen::VectorXd gravity_torque(const Robot& robot, const Eigen::VectorXd& q);
/// P(q) = -sum_i m_i g^T p_i(q), p_i = position of the CoM of link i (zero level: the origin of {s}).
double potential_energy(const Robot& robot, const Eigen::VectorXd& q);
/// q'' = M(q)^-1 (tau - c(q, dq) - g(q) - b .* dq) (MR Eq. 8.1 with viscous friction, MR Eq. 11.20).
Eigen::VectorXd forward_dynamics(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& tau);

/// Diagonal PD gains from the diagonal of M(q) (Lab 5 rule per joint).
struct PDGains {
    Eigen::MatrixXd Kp;  ///< diag(M_jj wn^2)
    Eigen::MatrixXd Kd;  ///< diag(2 zeta wn M_jj - b_j)
};
PDGains pd_gains_from_inertia(const Robot& robot, const Eigen::VectorXd& q, double zeta, double wn);

/// True if every eigenvalue of A has a negative real part (x' = A x is asymptotically stable).
bool is_hurwitz(const Eigen::MatrixXd& A);

Eigen::VectorXd time_grid(double T, double dt);
/// One RK4 step of x' = f(x) (the input is held over the step).
Eigen::VectorXd rk4_step(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x,
                         double h);

/// policy(t, q, dq) -> tau
using Policy = std::function<Eigen::VectorXd(double, const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/// The result dict of simulate: t (N+1), q, dq, tau (N+1) x n, row k = sample k (the last row of tau is the
/// torque the controller would apply at T).
struct SimResult {
    Eigen::VectorXd t;
    Eigen::MatrixXd q, dq, tau;
};
/// Closed loop: tau_k = policy(t_k, q_k, dq_k) held over [t_k, t_k + dt]; one RK4 step of the plant per sample.
SimResult simulate(const Robot& plant, const Policy& policy, const Eigen::VectorXd& q0, const Eigen::VectorXd& dq0,
                   double T, double dt);

/// Stabilising solution P of the algebraic Riccati equation A^T P + P A - P B R^-1 B^T P + Q = 0 (Astrom-Murray
/// Eq. 6.27), by the matrix-sign-function method on the Hamiltonian matrix [[A, -B R^-1 B^T], [-Q, -A^T]].
/// The C++ counterpart of scipy.linalg.solve_continuous_are (agrees with it to about 1e-12 relative on this lab's
/// systems).  Throws std::invalid_argument for wrong sizes or if the iteration does not converge.
Eigen::MatrixXd care_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R);

// ==========================================================================================
// Part A -- passivity of the manipulator and the Lyapunov function of PD + gravity compensation (you write these)
// ==========================================================================================

/// dM[k] = dM/dq_k at q by central differences of mass_matrix (step h).  Returns n matrices n x n.
std::vector<Eigen::MatrixXd> mass_matrix_partials(const Robot& robot, const Eigen::VectorXd& q, double h = 1e-6);

/// Coriolis matrix C(q, dq) with c_ij = sum_k Gamma_ijk dq_k (MR Eq. 8.19), Christoffel symbols
/// Gamma_ijk = 1/2 (dm_ij/dq_k + dm_ik/dq_j - dm_jk/dq_i) (MR Eq. 8.17) from mass_matrix_partials.
Eigen::MatrixXd coriolis_matrix(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                double h = 1e-6);

/// M'(q, dq) = dM/dt = sum_k (dM/dq_k) dq_k.
Eigen::MatrixXd mass_matrix_dot(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                double h = 1e-6);

/// Total mechanical energy E = 1/2 dq^T M(q) dq + P(q) (kinetic energy MR Eq. 8.14 + potential energy).
double stored_energy(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq);

/// PD setpoint control with gravity compensation (MR Eq. 11.38 with K_i = 0 and dqd = 0):
/// tau = Kp (qd - q) - Kd dq + g~(q).  Kp, Kd are n x n gain matrices; g~ from the MODEL.
Eigen::VectorXd pd_gravity_setpoint_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                           const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp,
                                           const Eigen::MatrixXd& Kd, const Robot& model);

/// Error energy V = 1/2 theta_e^T Kp theta_e + 1/2 dq^T M(q) dq (MR Eq. 11.41), M of the PLANT.
double lyapunov_energy(const Robot& plant, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                       const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp);

/// dV/dt along the plant for the applied torque tau, from the first line of MR Eq. 11.42:
/// V' = -dq^T Kp theta_e + dq^T M(q) q'' + 1/2 dq^T M'(q, dq) dq, with q'' = forward_dynamics(plant, q, dq, tau).
double lyapunov_rate(const Robot& plant, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                     const Eigen::VectorXd& qd, const Eigen::MatrixXd& Kp, const Eigen::VectorXd& tau);

/// Rest configuration q_ss of PD + g~ when the model gravity is wrong: solve
/// F(q) = Kp (qd - q) + g~(q) - g(q) = 0 by Newton's method from q = qd (Jacobian by central differences, step h).
/// Throws std::invalid_argument if |F| > 1e-9 after `iters` iterations.
Eigen::VectorXd equilibrium_offset(const Robot& plant, const Robot& model, const Eigen::VectorXd& qd,
                                   const Eigen::MatrixXd& Kp, int iters = 50, double tol = 1e-12, double h = 1e-6);

// ==========================================================================================
// Part B -- LQR for the linearised arm (you write these)
// ==========================================================================================

/// The Python tuple (A, B, u_eq) of linearize.
struct Linearization {
    Eigen::MatrixXd A;     ///< 2n x 2n
    Eigen::MatrixXd B;     ///< 2n x n
    Eigen::VectorXd u_eq;  ///< n
};
/// Linearise the plant about the equilibrium (q_eq, dq = 0, u_eq = g(q_eq)) (Tedrake Sec. 3.4.1):
/// x = (q - q_eq, dq), u = tau - u_eq, x' = A x + B u with
/// A = [[0, I], [-M^-1 dg/dq, -M^-1 diag(b)]], B = [[0], [M^-1]] (M, dg/dq at q_eq; dg/dq by central differences).
Linearization linearize(const Robot& plant, const Eigen::VectorXd& q_eq, double h = 1e-6);

/// Bryson's rule (Astrom-Murray Exercise 6.15): {Q, R} = {diag(1 / x_max^2), diag(1 / u_max^2)}.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bryson_weights(const Eigen::VectorXd& x_max, const Eigen::VectorXd& u_max);

/// Solve the Lyapunov equation A^T P + P A + Q = 0 for P with the Kronecker (vec) form
/// (I kron A^T + A^T kron I) vec(P) = -vec(Q) (vec stacks COLUMNS).  Returns the symmetric part of P.
Eigen::MatrixXd lyapunov_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Q);

/// Infinite-horizon LQR (Astrom-Murray Eqs. 6.26-6.27): P from the algebraic Riccati equation (care_solve),
/// K = R^-1 B^T P.  Returns {K, P}.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> lqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

/// The PD + gravity-compensation law as linear state feedback about (q_eq, 0): K0 = [Kp - dg~/dq(q_eq), Kd]
/// (dg~/dq by central differences, step h), so that tau - g~(q_eq) = -K0 x to first order.  Returns n x 2n.
Eigen::MatrixXd pd_state_feedback(const Robot& model, const Eigen::VectorXd& q_eq, const Eigen::MatrixXd& Kp,
                                  const Eigen::MatrixXd& Kd, double h = 1e-6);

/// The Python tuple (K, P, iterations) of kleinman_lqr.
struct KleinmanResult {
    Eigen::MatrixXd K;
    Eigen::MatrixXd P;
    int iterations = 0;
};
/// Kleinman's iteration (Newton's method for the Riccati equation) from a STABILISING gain K0:
/// repeat A_k = A - B K_k; solve A_k^T P_k + P_k A_k + Q + K_k^T R K_k = 0 (lyapunov_solve); K_{k+1} = R^-1 B^T P_k,
/// until ||P_k - P_{k-1}||_F <= tol max(1, ||P_k||_F).  Throws std::invalid_argument if A - B K0 is not Hurwitz.
KleinmanResult kleinman_lqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                            const Eigen::MatrixXd& R, const Eigen::MatrixXd& K0, int iters = 50, double tol = 1e-10);

/// Cost J = int_0^inf (x^T Q x + u^T R u) dt of u = -K x from x(0) = x0 on x' = A x + B u: J = x0^T P_K x0 with
/// (A - BK)^T P_K + P_K (A - BK) + Q + K^T R K = 0.  Throws std::invalid_argument if A - BK is not Hurwitz.
double lqr_cost(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& K, const Eigen::VectorXd& x0);

/// tau = u_eq - K x with x = (q - q_eq, dq) (the LQR or PD law applied to the nonlinear arm).
Eigen::VectorXd state_feedback_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& q_eq,
                                      const Eigen::VectorXd& u_eq, const Eigen::MatrixXd& K);

/// Quadratic cost of a simulated run, int (x^T Q x + u^T R u) dt by the trapezoidal rule over sim.t, with
/// x_k = (q_k - q_eq, dq_k) and u_k = tau_k - u_eq.
double run_cost(const SimResult& sim, const Eigen::VectorXd& q_eq, const Eigen::VectorXd& u_eq,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

}  // namespace lab

#endif  // SBEG461_LAB07_HPP
