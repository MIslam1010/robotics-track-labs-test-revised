// SBEG 461 Lab 5 -- Error dynamics and PID control of a single joint (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Modern Robotics (MR) Ch. 11: joint error  theta_e = theta_d - theta  (MR p. 407), unit error
// response (MR Sec. 11.2.1), standard second-order form  e'' + 2 zeta wn e' + wn^2 e = 0  (MR Eq. 11.8).
// In code q = theta, qd = theta_d, e = theta_e.  Units: SI (s, rad, rad/s, N m).
//
// Parts
//   A. Error dynamics: zeta and wn, characteristic roots and stability, the unit error response, overshoot and
//      2 % settling time, measured metrics of a simulated response            MR Sec. 11.2, pp. 407-415
//   B. Motion control with velocity inputs: P, PI and feedforward + PI control of a joint whose servo is in
//      velocity mode (stepper or velocity-mode amplifier)                    MR Sec. 11.3.1, Eqs. 11.11-11.15
//   C. PID control with torque inputs: gain design for a target zeta and wn, the integral-gain bound, the PD
//      steady-state error under gravity, a discrete PID controller (MR Fig. 11.15 pseudocode) with torque
//      saturation and anti-windup, and a single-joint simulator of  tau = M q'' + b q' + g(q)  (MR Eq. 11.21)
//                                                                            MR Sec. 11.4.1.1, pp. 424-429
//   D. The ROS 2 node (ros2/ folder) reuses JointPID unchanged; nothing to write here.
//
// Python -> C++: a plant dict -> lab::Plant; a time series (NumPy 1-D array) -> Eigen::VectorXd; polynomial
// coefficients -> Eigen::VectorXd (highest power first); complex roots -> Eigen::VectorXcd; a tuple of two values
// -> std::pair; a dict result -> a small struct with the same field names; None -> an empty std::optional
// (std::nullopt); the callable f(x) of rk4_step -> lab::StateFunction (a lambda works); ValueError ->
// std::invalid_argument.
#ifndef SBEG461_LAB5_HPP
#define SBEG461_LAB5_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <complex>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace lab {

// ==========================================================================================
// Provided (complete in starter.cpp, do not edit): plants, reference trajectories, RK4 integrator
// ==========================================================================================

/// A single-joint plant of data/joint_plants.json:  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) with
/// g(q) = g_cos cos q + g_sin sin q  (Python: the dict returned by load_plant, same field names).
struct Plant {
    std::string name;
    double M = 0.0;                   ///< inertia about the joint axis [kg m^2]
    double b = 0.0;                   ///< viscous friction [N m s/rad]
    double g_cos = 0.0;               ///< gravity coefficients of g(q) [N m]
    double g_sin = 0.0;
    std::optional<double> tau_max;    ///< actuator torque limit [N m]; empty = no limit (Python None)
    std::optional<double> qdot_max;   ///< speed limit [rad/s]; empty = none
    Eigen::Vector2d q_limits = Eigen::Vector2d::Zero();  ///< (lower, upper) [rad]
    Eigen::VectorXd locked_q;         ///< course-arm plants: the locked joint angles q_home (empty otherwise)
    int joint_index = 0;              ///< course-arm plants: joint number 1..6 (0 otherwise)
    double payload = 0.0;             ///< course-arm plants: payload [kg]
    std::string description;
    std::string source;
};

/// A plant from data/joint_plants.json. Names: arm_j1, arm_j2 (course arm, illustrative inertias, derated
/// limits), knee (prosthesis), eod_wrist (EOD arm), mr_link (MR Sec. 11.4.1, p. 424).
/// Throws std::invalid_argument for an unknown name.
Plant load_plant(const std::string& name);

/// g(q) = g_cos cos q + g_sin sin q (N m); the torque needed to hold the joint still at q.
double plant_gravity(const Plant& plant, double q);

/// State derivative of x = (q, q') under torque tau:  q'' = (tau - b q' - g(q)) / M  (g = 0 if gravity is false).
Eigen::Vector2d plant_rhs(const Plant& plant, const Eigen::Vector2d& x, double tau, bool gravity = true);

/// Right-hand side f(x) of  x' = f(x)  (the input is held constant over the step).
using StateFunction = std::function<Eigen::Vector2d(const Eigen::Vector2d&)>;

/// One classical Runge-Kutta step of x' = f(x).
Eigen::Vector2d rk4_step(const StateFunction& f, const Eigen::Vector2d& x, double h);

/// 0, dt, 2 dt, ..., T (inclusive, rounded so that the last sample is T): n = round(T / dt), t_k = k dt.
Eigen::VectorXd time_grid(double T, double dt);

/// Setpoint reference: qd = q0 before t_step, q1 from t_step on; qd' = 0.  Returns (qd, qd_dot).
std::pair<Eigen::VectorXd, Eigen::VectorXd> step_reference(const Eigen::VectorXd& t, double q0, double q1,
                                                           double t_step = 0.0);

/// Constant-velocity reference qd = q0 + rate t, qd' = rate (MR Fig. 11.8).  Returns (qd, qd_dot).
std::pair<Eigen::VectorXd, Eigen::VectorXd> ramp_reference(const Eigen::VectorXd& t, double q0, double rate);

/// The knee swing-phase reference of data/knee_swing.csv (Python: the tuple t, qd, qd', qd'').
struct KneeSwing {
    Eigen::VectorXd t, qd, qd_dot, qd_ddot;
};

/// Illustrative knee swing-phase reference (data/knee_swing.csv): t, qd, qd', qd'' (1 ms grid).
KneeSwing load_knee_swing();

// ==========================================================================================
// Part A -- error dynamics (MR Sec. 11.2)
// ==========================================================================================

/// Damping ratio and natural frequency of  e'' + a1 e' + a0 e = 0  compared with MR Eq. 11.8:
/// a1 = 2 zeta wn, a0 = wn^2.  Requires a0 > 0 (else throw std::invalid_argument).  Returns (zeta, wn).
std::pair<double, double> second_order_params(double a1, double a0);

/// Roots of the characteristic polynomial  c0 s^p + c1 s^(p-1) + ... + cp  (MR Eq. 11.3), coefficients
/// highest power first (numpy.roots convention).  Returns the p complex roots (any order).
Eigen::VectorXcd char_roots(const Eigen::VectorXd& coeffs);

/// True if every root of the characteristic polynomial has a negative real part (MR p. 410): Re(s) < -tol.
bool is_stable(const Eigen::VectorXd& coeffs, double tol = 1e-12);

/// Unit error response e(t) of MR Eq. 11.8 with e(0) = 1, e'(0) = 0 (MR p. 413):
///   zeta > 1:  c1 e^{s1 t} + c2 e^{s2 t},  s1,2 = -zeta wn +- wn sqrt(zeta^2 - 1),
///              c1 = 1/2 + zeta / (2 sqrt(zeta^2 - 1)),  c2 = 1/2 - zeta / (2 sqrt(zeta^2 - 1));
///   zeta = 1:  (1 + wn t) e^{-wn t};
///   zeta < 1:  (cos wd t + zeta / sqrt(1 - zeta^2) sin wd t) e^{-zeta wn t},  wd = wn sqrt(1 - zeta^2).
/// Treat |zeta - 1| < 1e-9 as critical damping.  One value per entry of t.
Eigen::VectorXd unit_error_response(double zeta, double wn, const Eigen::VectorXd& t);

/// Percent overshoot of the second-order unit error response, exp(-pi zeta / sqrt(1 - zeta^2)) x 100
/// for 0 <= zeta < 1 and 0 for zeta >= 1 (MR p. 415).
double overshoot_from_zeta(double zeta);

/// Inverse of overshoot_from_zeta for 0 < overshoot < 100 %:  zeta = -ln(OS) / sqrt(pi^2 + ln^2(OS)),
/// OS = overshoot_pct / 100.
double zeta_from_overshoot(double overshoot_pct);

/// Result of step_metrics (Python: dict(ess=..., overshoot=..., settling_time=...)).
struct StepMetrics {
    double ess = 0.0;            ///< steady-state error
    double overshoot = 0.0;      ///< percent
    double settling_time = 0.0;  ///< s; NaN (std::numeric_limits<double>::quiet_NaN()) if never settled
};

/// Measured metrics of an error response e(t) (MR Sec. 11.2.1, p. 408):
///   ess            steady-state error (default, std::nullopt: the last sample e[-1]);
///   overshoot      |e_min - ess| / |e(0) - ess| x 100 if the response crosses ess, else 0, where e_min is
///                  the extreme value on the far side of ess (use the normalised d = (e - ess)/(e(0) - ess):
///                  overshoot = max(0, -min d) x 100);
///   settling_time  the first sample time T such that |e(t) - ess| <= band |e(0) - ess| for all t >= T
///                  (NaN if the last sample is still outside the band).
/// If e(0) == ess: overshoot 0 and settling_time t[0].
StepMetrics step_metrics(const Eigen::VectorXd& t, const Eigen::VectorXd& e, std::optional<double> ess = std::nullopt,
                         double band = 0.02);

// ==========================================================================================
// Part B -- motion control with velocity inputs (MR Sec. 11.3.1)
// ==========================================================================================

/// Gains of the velocity-input PI controller (MR Eq. 11.13) giving the error dynamics
/// e'' + Kp e' + Ki e = 0 (MR Eq. 11.14) the target zeta and wn:  Kp = 2 zeta wn, Ki = wn^2.  Returns (Kp, Ki).
std::pair<double, double> pi_velocity_gains(double zeta, double wn);

/// Result of simulate_velocity_joint (Python: dict(q, u, e)), one entry per sample of t.
struct VelocitySim {
    Eigen::VectorXd q, u, e;
};

/// Discrete-time simulation of a joint with a velocity input (the servo realises the commanded velocity
/// exactly, MR Sec. 11.3).  At every sample k (dt = t[1] - t[0]):
///     e_k = qd_k - q_k ;  E_k = E_{k-1} + e_k dt  (E_{-1} = 0) ;
///     u_k = [qd_dot_k if feedforward] + kp e_k + ki E_k            (MR Eqs. 11.12, 11.13, 11.15)
///     u_k is clipped to +-qdot_max if given; while clipped (|u| at the limit and the error has the sign of u)
///     the integral is NOT updated (E_k = E_{k-1}; simple anti-windup);
///     q_{k+1} = q_k + u_k dt.
/// Returns (q, u, e), each of the length of t (u and e at every sample; q[0] = q0).
VelocitySim simulate_velocity_joint(const Eigen::VectorXd& t, const Eigen::VectorXd& qd, const Eigen::VectorXd& qd_dot,
                                    double kp, double ki = 0.0, double q0 = 0.0, bool feedforward = false,
                                    std::optional<double> qdot_max = std::nullopt);

// ==========================================================================================
// Part C -- PID control with torque inputs (MR Sec. 11.4.1.1)
// ==========================================================================================

/// PD gains for the setpoint error dynamics  M e'' + (b + Kd) e' + Kp e = 0  (MR Eq. 11.25) with the
/// target zeta and wn (MR Eq. 11.26):  Kp = M wn^2,  Kd = 2 zeta wn M - b.  Returns (Kp, Kd).
std::pair<double, double> pd_gains(double M, double b, double zeta, double wn);

/// Upper bound on Ki for stability of the PID third-order error dynamics (MR Eq. 11.30, p. 427):
/// 0 < Ki < (b + Kd) Kp / M.
double ki_max(double M, double b, double kp, double kd);

/// Coefficients (highest power first) of MR Eq. 11.30:  s^3 + (b + Kd)/M s^2 + Kp/M s + Ki/M.
/// With ki = 0 return the PD polynomial  s^2 + (b + Kd)/M s + Kp/M  (MR Eq. 11.26).
Eigen::VectorXd pid_char_poly(double M, double b, double kp, double ki, double kd);

/// Steady-state error of PD setpoint control under gravity (MR Eq. 11.27): at rest Kp e = g(q) with
/// q = qd - e.  Solve  f(e) = Kp e - g(qd - e) = 0  for e by Newton's method from e = g(qd)/Kp
/// (f'(e) = Kp + g'(qd - e), g'(q) = -g_cos sin q + g_sin cos q).  Returns e (rad).
double pd_gravity_error(const Plant& plant, double kp, double qd);

/// The P, I and D contributions of the last update (Python: the dict terms with keys P, I, D).
struct PidTerms {
    double P = 0.0, I = 0.0, D = 0.0;
};

/// Discrete PID joint controller (MR Eq. 11.23, pseudocode MR Fig. 11.15) with an optional torque limit
/// and integrator anti-windup (MR p. 427; Astrom-Murray Sec. 10.4).
///
///     lab::JointPID ctrl(kp, ki, kd, dt, tau_max = std::nullopt, anti_windup = true);
///     ctrl.reset(q0);                      // error integral = 0, previous angle = q0
///     double tau = ctrl.update(qd, qd_dot, q, qdot = std::nullopt, tau_ff = 0.0);
///
/// update() at every servo cycle:
///     if qdot is empty: qdot = (q - q_prev) / dt   (0 at the first call after reset without q0)
///     q_prev = q
///     e = qd - q ;  e_dot = qd_dot - qdot
///     eint_new = eint + e dt
///     v = Kp e + Kd e_dot + Ki eint_new + tau_ff
///     tau = clip(v, -tau_max, tau_max)  (no clipping if tau_max is empty)
///     anti-windup: if anti_windup and v was clipped and e has the sign of v, keep eint (do not integrate)
///                  and recompute tau = clip(Kp e + Kd e_dot + Ki eint + tau_ff); otherwise eint = eint_new.
/// Members: kp, ki, kd, dt, tau_max, anti_windup, eint, q_prev, terms (P, I, D of the last call),
/// saturated (bool, last call).  Gains may be changed between calls (ROS parameters do this).
/// The constructor (given) calls reset().
struct JointPID {
    JointPID(double kp, double ki, double kd, double dt, std::optional<double> tau_max = std::nullopt,
             bool anti_windup = true);

    /// Error integral = 0, q_prev = q0 (empty if q0 is empty), terms = 0, saturated = false.
    void reset(std::optional<double> q0 = std::nullopt);

    /// One servo cycle (see above); returns the torque.  Updates eint, q_prev, terms and saturated.
    double update(double qd, double qd_dot, double q, std::optional<double> qdot = std::nullopt, double tau_ff = 0.0);

    double kp, ki, kd, dt;
    std::optional<double> tau_max;
    bool anti_windup;
    double eint = 0.0;
    std::optional<double> q_prev;
    PidTerms terms;
    bool saturated = false;
};

/// Result of simulate_joint (Python: dict(t, q, dq, tau, e)), one entry per sample of t.
struct JointSim {
    Eigen::VectorXd t, q, dq, tau, e;  ///< tau = applied torque, e = qd - q
};

/// Closed-loop simulation of  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) under a sampled controller.
/// t is the controller time grid (dt = t[1] - t[0] must equal ctrl.dt, else std::invalid_argument).
/// ctrl.reset(q0) first; then at every sample k:
///     q_meas = q_k rounded to a multiple of enc_res (if given; an encoder), qdot_meas = dq_k if
///     velocity == "measured", else empty (the controller differences q_meas, MR Fig. 11.15);
///     tau_k = ctrl.update(qd[k], qd_dot[k], q_meas, qdot_meas, tau_ff[k] if tau_ff is not empty else 0);
///     the actuator applies tau_k clipped to +-plant.tau_max (if not empty), held constant until t_{k+1};
///     integrate (q, q') from t_k to t_{k+1} with `substeps` rk4_step calls of plant_rhs(..., gravity).
/// ctrl is passed by reference (as in Python, the caller's controller is used and changed).
/// tau_ff: an empty vector (default) means no feedforward.
JointSim simulate_joint(const Plant& plant, JointPID& ctrl, const Eigen::VectorXd& t, const Eigen::VectorXd& qd,
                        const Eigen::VectorXd& qd_dot, double q0, double dq0 = 0.0, int substeps = 4,
                        const std::string& velocity = "measured", std::optional<double> enc_res = std::nullopt,
                        const Eigen::VectorXd& tau_ff = Eigen::VectorXd(), bool gravity = true);

}  // namespace lab

#endif  // SBEG461_LAB5_HPP
