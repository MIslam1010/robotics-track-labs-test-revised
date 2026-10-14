// SBEG 461 Lab 8 -- Impedance control on a haptic device and a simulated rehabilitation robot (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp, with the full documentation).
// Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 8 --cpp
//
// Rules (as in starter.py): handle_mass and gravity_stiffness may call mrcpp (mr::mass_matrix, mr::gravity_forces:
// the C++ port of modern_robotics); everything else uses Eigen and the provided helpers of helpers.cpp
// (rehab_fk, rehab_jacobian, rehab_jacobian_dot, rehab_dynamics, rk4_step, expm, find_peaks, linear_fit, ...).
// Notation follows Modern Robotics (MR) Ch. 11 (see lab.hpp).
//
// Eigen reminders:  Eigen::VectorXd v = Eigen::VectorXd::Zero(n);  v(k) = ...;  v.size();  v.head(3), v.tail(3);
//   Eigen::MatrixXd A = Eigen::MatrixXd::Zero(r, c);  A.topLeftCorner(2, 2) = ...;  A.block(0, 2, 2, 1) = ...;
//   A.row(3) = row_vector;  a * b.transpose() (= np.outer(a, b));  M.inverse(), M.partialPivLu().solve(b)
//   (= np.linalg.solve);  x.array().square() (= x**2 element by element);  std::nearbyint(y) (= Python round:
//   halves go to the even integer);  std::signbit(y);  std::deque<double> for a FIFO buffer (push_back, front,
//   pop_front).  Loop indices: Eigen::Index (a signed integer).  Write 0.5, not 1/2 (integer division).
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

namespace lab {

namespace {
[[maybe_unused]] constexpr double PI = 3.14159265358979323846;  // for fit_free_decay (M_PI is not standard C++)
}

// ---------------------------------------------------------------------------
// Part A -- the haptic device as a 1-DOF plant
// ---------------------------------------------------------------------------

double handle_mass(const HapkitModel& model) {
    // TODO: I_pivot = mr::mass_matrix(Eigen::VectorXd::Zero(1), model.Mlist, model.Glist, model.Slist)(0, 0);
    //       m_h = (I_pivot + model.params.reflected_rotor_inertia) / rh^2
    throw std::logic_error("TODO: handle_mass");
}

double gravity_stiffness(const HapkitModel& model) {
    // TODO: central difference of mr::gravity_forces(theta, model.g, Mlist, Glist, Slist)(0) at theta = +-h
    //       (h = 1e-6; theta as Eigen::VectorXd::Constant(1, h)), divided by 2h and by rh^2
    throw std::logic_error("TODO: gravity_stiffness");
}

Eigen::VectorXd handle_force_to_current(const Eigen::VectorXd& F, const HapkitModel& model, double k_t) {
    // TODO: tau = F rh, tau_m = tau / ratio, I = tau_m / k_t (element by element)
    throw std::logic_error("TODO: handle_force_to_current");
}

FreeDecayFit fit_free_decay(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double min_fraction) {
    // TODO: (see the documentation in lab.hpp and lab_handout.md)
    //   1. dt = t(1) - t(0); zero crossings i where std::signbit(x(i)) != std::signbit(x(i + 1));
    //      T_guess = 2 * mean spacing of their times if there are more than 2, else (t_end - t_0) / 4
    //   2. peaks = find_peaks(x, std::max(1, static_cast<int>(0.5 * T_guess / dt))); keep x(i) > 0, then keep
    //      x(i) >= min_fraction * x(first kept peak)
    //   3. with k = 0, 1, 2, ... (peak number): slope of ln(peak) and slope of the peak times (linear_fit)
    //   4. delta = -slope; zeta = delta / sqrt(4 pi^2 + delta^2); wn = (2 pi / T_d) / sqrt(1 - zeta^2)
    throw std::logic_error("TODO: fit_free_decay");
}

Identification identify_device(const Eigen::VectorXd& Ks, const std::vector<FreeDecayFit>& fits) {
    // TODO: linear_fit(Ks, wn^2) -> m_h = 1 / slope, k_g = intercept * m_h; b_app = 2 zeta wn m_h (one per test);
    //       linear_fit(Ks, b_app) -> b_h = intercept, tau_eff = -slope
    throw std::logic_error("TODO: identify_device");
}

// ---------------------------------------------------------------------------
// Part B -- rendering springs, dampers and walls with a sampled-data loop
// ---------------------------------------------------------------------------

double spring_damper_force(double x, double v, double K, double B, double x0, double v0) {
    // TODO: F = -K (x - x0) - B (v - v0)
    throw std::logic_error("TODO: spring_damper_force");
}

double wall_force(double x, double v, double K, double B, double x_wall, int side) {
    // TODO: 0 outside (side * (x - x_wall) <= 0); inside -K (x - x_wall) - B v, clipped so that side * F <= 0
    throw std::logic_error("TODO: wall_force");
}

DeviceSim simulate_device(const Device& dev, const ForceLaw& law, double T_end, double x0, double v0,
                          const ForceLaw& f_hand, int delay_steps, bool quantize, int substeps) {
    // TODO: exactly as documented in lab.hpp.  n = std::llround(T_end / dev.T) samples after t = 0 (n + 1 in all),
    //   t(k) = k * T.  Resize every vector of the result to n + 1 (saturated: .assign(n + 1, false)).
    //   Per sample: quantise (std::nearbyint), backward-difference velocity, law, saturation, delay buffer
    //   (a std::deque<double> pre-filled with delay_steps zeros: push_back the new command, the applied force is
    //   front(), then pop_front()), then `substeps` rk4_step calls of the plant with the force held.
    //   f_hand may be empty: use `f_hand ? f_hand(t_k, x, v) : 0.0`.
    throw std::logic_error("TODO: simulate_device");
}

PushTest static_push_test(const Device& dev, double K, double B, double F_load, double T_end) {
    // TODO: simulate_device with law = spring_damper_force(x, v, K, B) and f_hand = F_load (lambdas);
    //       x_ss = mean of x over t >= T_end - 0.2; K_rendered = F_load / x_ss; error_pct = 100 (K_rendered - K) / K
    throw std::logic_error("TODO: static_push_test");
}

Eigen::MatrixXd closed_loop_matrix(double m, double b, double k_g, double K, double B, double T, int delay_steps) {
    // TODO: augmented 3 x 3 matrix [[A_c, B_c], [0, 0]] -> E = expm(Aug * T); A_d = E top-left 2 x 2, B_d = E(0..1, 2).
    //   c_k = crow * z_k with crow(0) = -K - B/T, crow(2) = B/T.  d = 0: the top two rows are [A_d, 0] + B_d crow;
    //   d > 0: A_d, B_d in column 2 + d, row 3 = crow, shift c_{k-j} -> slot j + 1.  Always A(2, 0) = 1.
    throw std::logic_error("TODO: closed_loop_matrix");
}

double max_stable_K(double m, double b, double k_g, double B, double T, int delay_steps, double K_hi, double tol) {
    // TODO: bisection on [0, K_hi] with spectral_radius(closed_loop_matrix(...)) < 1 (return K_hi if stable there)
    throw std::logic_error("TODO: max_stable_K");
}

double first_order_K_limit(double b, double B, double T, int delay_steps) {
    // TODO: (b + B) / (T/2 + d T)
    throw std::logic_error("TODO: first_order_K_limit");
}

Eigen::VectorXd device_energy(const Device& dev, const DeviceSim& sim, double K, double x0) {
    // TODO: E = 1/2 m_h v^2 + 1/2 K (x - x0)^2 + 1/2 k_g x^2 at every sample
    throw std::logic_error("TODO: device_energy");
}

// ---------------------------------------------------------------------------
// Part C -- the rehabilitation robot: apparent inertia, impedance and admittance control
// ---------------------------------------------------------------------------

Eigen::Matrix2d task_inertia(const Eigen::Vector2d& q) {
    // TODO: Lambda = J^-T M J^-1 with M from rehab_dynamics(q, 0).first and J = rehab_jacobian(q)
    throw std::logic_error("TODO: task_inertia");
}

double apparent_mass(const Eigen::Vector2d& q, const Eigen::Vector2d& u) {
    // TODO: normalise u; m_u = 1 / (u^T Lambda^-1 u)
    throw std::logic_error("TODO: apparent_mass");
}

Eigen::Vector2d impedance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                         const Eigen::Vector2d& p_d, const Eigen::Vector2d& v_d,
                                         const Eigen::Matrix2d& K, const Eigen::Matrix2d& B) {
    // TODO: p = rehab_fk(q), v = J dq; tau = J^T (K (p_d - p) + B (v_d - v))
    throw std::logic_error("TODO: impedance_control_torque");
}

Eigen::Vector2d admittance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                          const Eigen::Vector2d& f_meas, const Eigen::Vector2d& p0,
                                          const Eigen::Matrix2d& M_d, const Eigen::Matrix2d& B_d,
                                          const Eigen::Matrix2d& K_d) {
    // TODO: a_d = M_d^-1 (f_meas - B_d v - K_d (p - p0)); qdd = J^-1 (a_d - J' dq) (rehab_jacobian_dot);
    //       tau = M qdd + c - J^T f_meas with (M, c) = rehab_dynamics(q, dq)
    throw std::logic_error("TODO: admittance_control_torque");
}

}  // namespace lab
