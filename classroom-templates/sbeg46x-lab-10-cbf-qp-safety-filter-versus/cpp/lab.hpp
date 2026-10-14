// SBEG 46X Lab 10 -- CBF-QP safety filter versus saturation on an exoskeleton joint and on the Hapkit (C++ track):
// declarations.
//
// The same functions as starter.py (module `safety` of your course library sbeg46x_medrob), in namespace `lab`,
// with Eigen types instead of NumPy arrays. Write your code in starter.cpp (the 9 functions of Parts A-C). Do not
// change these declarations: test_lab.cpp calls them. The provided functions (parameters, gravity, references,
// patient and operator models, baselines, the small QP solver and the two simulators) are in helpers.cpp.
//
// Exoskeleton joint (MR Sec. 11.4.1, Eq. 11.21, angle from the hanging position, as in Lab 8):
//     M theta_ddot + b theta_dot + g(theta) = tau + tau_h,   g(theta) = g_s sin(theta) + g_c cos(theta),
//     |tau| <= tau_max  (actuator),   tau_h = torque of the patient/wearer on the joint (cuff force sensor).
// Hapkit handle (x = r_h theta):  m_h x_ddot + b_h x_dot + k_g x = F + F_h,  |F| <= 2 N,  watchdog at |x| > 40 mm.
// Safe sets:  joint range  h_hi = theta_max - theta >= 0,  h_lo = theta - theta_min >= 0;
//             forbidden region of the Hapkit  h = x_max - x >= 0.
// Barrier condition (first order): h_dot >= -alpha h.  Second order (relative degree two, exponential CBF):
//     psi1 = h_dot + alpha1 h >= 0,   psi2 = psi1_dot + alpha2 psi1 = h_ddot + (alpha1 + alpha2) h_dot + alpha1 alpha2 h >= 0.
// One-input CBF-QP  min (u - u_nom)^2  s.t.  a u >= b   ->   u* = u_nom + max(0, b - a u_nom) / a  (a != 0).
// References: Tedrake, Underactuated Robotics, Sec. 9.1.6 (barrier functions); MR Sec. 11.4.1 (Eq. 11.21),
// Sec. 11.7 (impedance, Eqs. 11.62-11.65); Ames et al., ECC 2019 (not in the course library).
// Units: SI.  All values are illustrative.
//
// Python -> C++: the parameter dicts of data/lab10.json -> the structs JointParams / HapkitParams (same field names;
// hk["operator"] -> hk.op, hk["push"] -> hk.cases.at("push")); a tuple -> std::pair or a small struct; the info
// dict {active, feasible} -> lab::FilterInfo; the log dicts of the simulators -> lab::JointLog / lab::HapkitLog;
// None (W, u_min, u_max of cbf_qp) -> an empty Eigen object; -np.inf / np.inf -> -lab::INF / lab::INF;
// ValueError -> std::invalid_argument.
#ifndef SBEG46X_LAB10_HPP
#define SBEG46X_LAB10_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// +infinity (np.inf): use -INF / INF for a one-sided interval.
inline constexpr double INF = std::numeric_limits<double>::infinity();

// ==========================================================================================
// Provided (given in helpers.cpp, do not edit): parameters, gravity, references, patient and operator models,
// baselines, a small QP solver and the simulators
// ==========================================================================================

/// PD + gravity-compensation gains from the natural frequency and damping ratio (Lab 8).
struct PdGains {
    double wn = 0.0;
    double zeta = 0.0;
};

/// 'step' experiment of a joint: set-points [[t_0, theta_0], [t_1, theta_1], ...] and PD gains.
struct StepTask {
    double theta0 = 0.0;
    double T = 0.0;
    std::vector<std::pair<double, double>> steps;  ///< (t_k, theta_k)
    PdGains pd;
};

/// 'spasm' experiment of the elbow: therapy reference, assist-as-needed gains, patient model and spasm.
struct SpasmTask {
    double theta0 = 0.0;
    double T = 0.0;
    struct Ref { double mid = 0.0, amp = 0.0, period = 1.0; } ref;
    struct Aan { double K = 0.0, B = 0.0, delta = 0.0; } aan;
    struct Patient { double c = 0.0, Kp = 0.0, Bp = 0.0; } patient;
    struct Spasm { double amp = 0.0, t_on = 0.0, t_off = 0.0, ramp = 0.0; } spasm;
    double tau_h_max = 0.0;
};

/// One exoskeleton joint of data/lab10.json ("elbow" or "knee"): Python j["M"] -> j.M, j["rom"] -> j.rom, ...
struct JointParams {
    std::string model;
    double M = 0.0, b = 0.0, g_s = 0.0, g_c = 0.0;
    double tau_rated = 0.0, tau_max = 0.0;
    double tau_rate = 0.0;                             ///< torque-rate limit (N m/s); 0 = none (the elbow has none)
    std::pair<double, double> mech_limits{0.0, 0.0};   ///< mechanical end stops (rad)
    std::pair<double, double> rom{0.0, 0.0};           ///< prescribed range of motion [theta_min, theta_max] (rad)
    double dt = 0.001;
    int n_sub = 1;
    struct Cbf { double alpha1 = 0.0, alpha2 = 0.0; } cbf;
    struct Stop { double K = 0.0, B = 0.0; } stop;
    StepTask step;
    bool has_spasm = false;                            ///< only the elbow has the 'spasm' experiment
    SpasmTask spasm;
};

/// One Hapkit run case ("push" or "hard_push").
struct HapkitCase {
    double x_int = 0.0;    ///< operator's intended position (m)
    double T = 0.0;        ///< duration (s)
    double F_h_max = 0.0;  ///< operator-force bound used by the filter (N)
};

/// The Hapkit of data/lab10.json in handle coordinates: hk["m_h"] -> hk.m_h, hk["operator"] -> hk.op,
/// hk["push"] -> hk.cases.at("push").
struct HapkitParams {
    std::string model;
    double m_h = 0.0, b_h = 0.0, k_g = 0.0;
    double T = 0.001, x_res = 0.0, v_filter_hz = 0.0;
    double force_max = 0.0, x_range = 0.0, x_max = 0.0;
    struct Cbf { double alpha1 = 0.0, alpha2 = 0.0; } cbf;
    struct Wall { double K = 0.0, B = 0.0; } wall;
    struct Operator { double m = 0.0, b = 0.0, k = 0.0, x0 = 0.0, t0 = 0.0, speed = 0.0; } op;
    std::map<std::string, HapkitCase> cases;
};

/// All lab parameters (data/lab10.json, illustrative values).
struct Params {
    JointParams elbow;
    JointParams knee;
    HapkitParams hapkit;
};

/// Lab parameters from data/lab10.json (an empty path = the lab's data/lab10.json).
Params load_params(const std::string& path = "");

/// g(theta) = g_s sin(theta) + g_c cos(theta)  (N m).
double gravity(double theta, const JointParams& j);

/// Piecewise-constant set-point: task.steps = [(t_0, theta_0), (t_1, theta_1), ...].
double step_setpoint(double t, const StepTask& task);

/// Slow flexion-extension therapy movement theta_d = mid - amp cos(2 pi t / period) and its rate (theta_d, omega_d).
std::pair<double, double> therapy_reference(double t, const SpasmTask::Ref& ref);

/// Pulse of height amp on [t_on, t_off] with raised-cosine ramps of length ramp.
double smooth_pulse(double t, double amp, double t_on, double t_off, double ramp);

/// Torque of the patient on the elbow: a fraction c of a voluntary tracking effort plus a flexor spasm.
double patient_torque(double t, double theta, double omega, const SpasmTask& task);

/// Intended hand position: rests at x0, moves towards x_int at op.speed from op.t0, then holds.
double operator_target(double t, const HapkitParams::Operator& op, double x_int);

/// BASELINE 'saturation' (as in the course-project package): a command u that would push x further outside
/// [x_min, x_max] is set to zero once x has reached the limit.  It acts only AT the limit.
double saturate_toward_limit(double x, double u, double x_min = -INF, double x_max = INF);

/// BASELINE virtual wall (Week 4, Lab 4): a one-sided spring-damper that pushes back once x > x_max.
double virtual_wall_force(double x, double v, double x_max, double K = 200.0, double B = 0.0);

/// Result information of solve_qp (Python: the dict info).
struct QPInfo {
    bool success = false;
    double fun = 0.0;  ///< optimal cost 1/2 z'Hz + f'z
    int nit = 0;       ///< number of active sets examined
};

/// min 1/2 z'Hz + f'z  s.t.  A_ub z <= b_ub,  lb <= z <= ub  (H symmetric positive definite).
/// An empty A_ub / b_ub / lb / ub = None; entries of lb / ub may be -INF / INF.  Returns (z, info).
/// C++ version: exact active-set enumeration of the KKT conditions for the small problems of the labs
/// (at most 8 variables and 20 constraints including the finite bounds, else std::invalid_argument); x0 is accepted
/// for the same signature as the Python helper (SciPy SLSQP) but not needed.
std::pair<Eigen::VectorXd, QPInfo> solve_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                                            const Eigen::MatrixXd& A_ub = Eigen::MatrixXd(),
                                            const Eigen::VectorXd& b_ub = Eigen::VectorXd(),
                                            const Eigen::VectorXd& lb = Eigen::VectorXd(),
                                            const Eigen::VectorXd& ub = Eigen::VectorXd(),
                                            const Eigen::VectorXd& x0 = Eigen::VectorXd());

/// Mechanical end stops of the joint at j.mech_limits (stiff one-sided spring-damper j.stop).
double end_stop_torque(double theta, double omega, const JointParams& j);

/// Nominal PD + gravity compensation (MR Eq. 11.38 with K_i = 0), gains from wn and zeta (Lab 8).
double pd_torque(double theta, double omega, double th_d, const JointParams& j, const PdGains& pd);

/// Log of simulate_joint (Python: the dict of numpy arrays); one entry per controller sample.
struct JointLog {
    Eigen::VectorXd t, theta, omega, tau, tau_nom, tau_h, th_d;
    std::vector<bool> active, feasible;
};

/// Closed-loop run of one exoskeleton joint.  The controller runs every dt (zero-order hold); the plant
/// M theta'' + b theta' + g = tau + tau_h + tau_stop (mechanical end stops) is integrated with n_sub semi-implicit
/// Euler sub-steps.
///   experiment  "step"  : nominal = PD + gravity compensation to the set-points of j.step (tau_h = 0)
///               "spasm" : nominal = gravity compensation + assist-as-needed (aan_torque) around the therapy
///                         reference, with the patient torque of patient_torque (voluntary effort + spasm)
///   mode        "none"  : nominal command, clipped at +-tau_max (and at the torque-rate limit if any)
///               "sat"   : as "none" + set-point clipped to the ROM + saturate_toward_limit on u = tau - g
///               "cbf"   : nominal command through joint_safety_filter (ROM barriers + actuator limits)
///   tau_h_mode  what the filter knows about tau_h: "measured" (cuff sensor), "zero" (ignored),
///               "robust" (worst case |tau_h| <= tau_h_max of the task).
///   alphas      (alpha1, alpha2); std::nullopt = j.cbf.
/// Calls YOUR rom_torque_bounds, joint_safety_filter and aan_torque.
JointLog simulate_joint(const JointParams& j, const std::string& experiment, const std::string& mode,
                        const std::string& tau_h_mode = "measured",
                        std::optional<std::pair<double, double>> alphas = std::nullopt);

/// Log of simulate_hapkit; one entry per firmware sample.
struct HapkitLog {
    Eigen::VectorXd t, x, v, F, F_h;
    std::vector<bool> active, feasible, tripped;
};

/// Hapkit forbidden region (x <= x_max) with an operator pushing towards x_int (the handle is held).
/// Firmware every T: quantised position, filtered backward-difference velocity, controller, saturation at
/// +-force_max, watchdog (motor off, latched) outside +-x_range.  Plant: (m_h + m_op) x'' = F - b_h x' - k_g x
/// + k_op (x_target - x) - b_op x'  (the hand is part of the moving mass).
///   hk_case  "push" or "hard_push" (operator target inside the reach of the 2 N motor, or beyond it)
///   mode     "none" (transparent, F = 0) | "wall" (virtual_wall_force, K of hk.wall) | "cbf" (hapkit_safety_filter
///            with the worst-case operator force F_h_max of the case).
/// Calls YOUR hapkit_safety_filter.
HapkitLog simulate_hapkit(const HapkitParams& hk, const std::string& hk_case, const std::string& mode,
                          std::optional<std::pair<double, double>> alphas = std::nullopt);

// ==========================================================================================
// Part A -- one-input CBF-QP filters
// ==========================================================================================

/// Closed-form CBF-QP with one input (course-project signature):
///     min (u - u_nom)^2   s.t.   Lf_h + Lg_h u >= -alpha h.
/// If Lg_h = 0 the input cannot change h_dot: return u_nom unchanged.  Otherwise return u_nom when it already
/// satisfies the condition, else the input that makes the condition hold with equality.
double cbf_filter_1d(double u_nom, double h, double Lf_h, double Lg_h, double alpha);

/// Result of filter_interval (Python: the tuple (u, active, feasible)).
struct IntervalResult {
    double u = 0.0;
    bool active = false;
    bool feasible = true;
};

/// 1-D QP  min (u - u_nom)^2  s.t.  lo <= u <= hi  (lo, hi may be -INF / INF).
/// Returns {u, active, feasible}: u = clip(u_nom, lo, hi); active = (u != u_nom); feasible = (lo <= hi).
/// When lo > hi the set is empty: return hi (the upper bound has priority) with feasible = false.
IntervalResult filter_interval(double u_nom, double lo, double hi);

/// CBF-QP with several inputs and constraints (course-project signature):
///     min (u - u_nom)' W (u - u_nom)   s.t.   A u >= b  (one row per barrier),  u_min <= u <= u_max.
/// Use the provided solve_qp (H = 2W, f = -2 W u_nom, A_ub = -A, b_ub = -b).  An empty W = identity; empty
/// u_min / u_max = no bounds.  Returns u.
Eigen::VectorXd cbf_qp(const Eigen::VectorXd& u_nom, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       const Eigen::MatrixXd& W = Eigen::MatrixXd(), const Eigen::VectorXd& u_min = Eigen::VectorXd(),
                       const Eigen::VectorXd& u_max = Eigen::VectorXd());

// ==========================================================================================
// Part B -- exoskeleton joint: range-of-motion barriers (relative degree two), assist-as-needed
// ==========================================================================================

/// Second-order (exponential) CBF for the joint range [theta_min, theta_max] = j.rom, written as bounds
/// on the motor torque tau of  M theta'' + b theta' + g(theta) = tau + tau_h:
///     upper barrier h = theta_max - theta:  theta'' <= alpha1 alpha2 (theta_max - theta) - (alpha1 + alpha2) omega
///     lower barrier h = theta - theta_min:  theta'' >= -alpha1 alpha2 (theta - theta_min) - (alpha1 + alpha2) omega
/// Returns (tau_lo, tau_hi).
std::pair<double, double> rom_torque_bounds(double theta, double omega, const JointParams& j, double alpha1,
                                            double alpha2, double tau_h = 0.0);

/// Python: the info dict of joint_safety_filter / hapkit_safety_filter.
struct FilterInfo {
    bool active = false;
    bool feasible = true;
};

/// Combine the barrier bounds (tau_lo, tau_hi) with the actuator bounds (act_lo, act_hi) and filter tau_nom.
/// Actuator bounds are physical and always hold.  If the combined set is empty, keep the actuator-feasible
/// torque closest to the barrier interval (upper barrier first), i.e.
///     u_c = min(max(tau_nom, tau_lo), tau_hi),   tau = clip(u_c, act_lo, act_hi).
/// Returns (tau, info) with info = {active, feasible}.
std::pair<double, FilterInfo> joint_safety_filter(double tau_nom, std::pair<double, double> cbf_bounds,
                                                  std::pair<double, double> act_bounds);

/// Assist-as-needed torque with a dead-band ('virtual tunnel') of half-width delta around the reference:
///     tau_a = -K dz(e) - B e_dot,   dz(e) = sign(e) max(|e| - delta, 0).
/// No stiffness inside the tunnel (the patient moves freely), a spring outside it.
double aan_torque(double e, double e_dot, double K, double B, double delta);

/// Result of violation_metrics (Python: the dict).
struct ViolationMetrics {
    double max_violation = 0.0;
    double time_outside = 0.0;
    double s_max = 0.0;
    double s_min = 0.0;
};

/// Range violation of a signal s(t) (theta or x) for the limits [lo, hi] (use -INF / INF for one-sided):
/// returns {max_violation = max(0, max(s - hi), max(lo - s)), time_outside = total time with s outside,
/// s_max, s_min}.  t is uniformly sampled (dt = t(1) - t(0)); t and s have the same length (>= 2).
ViolationMetrics violation_metrics(const Eigen::VectorXd& t, const Eigen::VectorXd& s, double lo, double hi);

// ==========================================================================================
// Part C -- Hapkit forbidden region
// ==========================================================================================

/// Upper bound on the motor force F from the second-order CBF of the forbidden region h = x_max - x >= 0
/// for  m_h x'' + b_h x' + k_g x = F + F_h  with an unmeasured operator force |F_h| <= F_h_max (worst case
/// F_h = +F_h_max pushes into the region):
///     F <= m_h [alpha1 alpha2 (x_max - x) - (alpha1 + alpha2) v] + b_h v + k_g x - F_h_max.
double hapkit_force_bound(double x, double v, const HapkitParams& hk, double alpha1, double alpha2, double F_h_max);

/// Filter the nominal handle force: F <= hapkit_force_bound(...), |F| <= force_max.
/// Returns (F, info) like joint_safety_filter.
std::pair<double, FilterInfo> hapkit_safety_filter(double F_nom, double x, double v, const HapkitParams& hk,
                                                   double alpha1, double alpha2, double F_h_max);

}  // namespace lab

#endif  // SBEG46X_LAB10_HPP
