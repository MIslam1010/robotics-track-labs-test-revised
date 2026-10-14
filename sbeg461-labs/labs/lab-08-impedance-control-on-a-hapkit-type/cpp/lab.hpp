// SBEG 461 Lab 8 -- Impedance control on a Hapkit-type haptic device and a simulated rehabilitation robot
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided helpers (models, loaders, rk4_step, the rehab-robot kinematics, dynamics and simulator, ...) are
// in helpers.cpp (given code, do not edit).
//
// Notation follows Modern Robotics (MR) Ch. 11.  Rendered impedance (MR Eq. 11.62):  m x'' + b x' + k x = f;
// task-space impedance  M x'' + B x' + K x = f_ext  (MR Eq. 11.64); impedance-control law (MR Eq. 11.65);
// admittance-control law  x''_d = M^-1 (f_ext - B x' - K x)  (MR Eq. 11.66); low-level torque control by motor
// current, tau = k_t I through a transmission (MR Sec. 11.8).  Units: SI (s, m, kg, N, rad, N m).
//
// Sign conventions used everywhere
//   * x is the handle (or hand) position, F_motor the force the device applies TO the user's hand,
//     f_hand the force the hand applies TO the device.  Device:  m_h x'' + b_h x' + k_g x = F_motor + f_hand.
//   * A rendered spring-damper about x0 applies F_motor = -K (x - x0) - B (v - v0).
//   * Rehabilitation robot:  M(q) q'' + c(q, q') = tau + J^T(q) f_ext   (f_ext = force of the hand on the robot;
//     MR Eq. 11.48 writes the same thing with F_tip = -f_ext).
//
// Parts
//   A. The haptic device as a 1-DOF plant: handle mass, gravity stiffness, force -> motor current (MR Sec. 11.8),
//      free-decay identification of m_h and b_h
//   B. Rendering virtual springs, dampers and walls at 1 kHz: sampled-data simulation, static push test, the exact
//      stability limit of the sampled loop (eigenvalues) and the first-order delay estimate, energy check
//   C. The rehabilitation robot: apparent (task-space) inertia (MR Eq. 8.90), impedance control without inertia
//      shaping (MR Eq. 11.65 with M = 0) and admittance control (MR Eq. 11.66 + inverse dynamics)
//
// Python -> C++: the model dicts -> lab::HapkitModel / lab::RehabModel; the device dict -> lab::Device; the
// result dicts of simulate_device / simulate_rehab / identify_device / static_push_test -> lab::DeviceSim /
// lab::RehabSim / lab::Identification / lab::PushTest; the tuples (wn, zeta) and (M, c) -> lab::FreeDecayFit and
// std::pair; call-backs (law, f_hand, controller, f_ext) -> std::function (an empty one = Python None);
// force_lag_hz = None -> 0.0; "K, B: matrices or scalars" -> an Eigen::Matrix2d version (you write it) plus a
// scalar version (given: it calls yours with K I, B I); ValueError -> std::invalid_argument.
#ifndef SBEG461_LAB8_HPP
#define SBEG461_LAB8_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types only)

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Transmission and limits of the Hapkit-type device (Python: hapkit()["params"]).
struct HapkitParams {
    double rh = 0.0;                       ///< handle grip radius from the pivot [m]
    double rs = 0.0;                       ///< sector radius [m]
    double rp = 0.0;                       ///< motor pulley radius [m]
    double ratio = 0.0;                    ///< transmission ratio rs / rp
    double rotor_inertia = 0.0;            ///< motor rotor inertia [kg m^2]
    double reflected_rotor_inertia = 0.0;  ///< rotor_inertia * ratio^2 [kg m^2]
    double force_max = 0.0;                ///< derated handle force limit [N]
    double x_range = 0.0;                  ///< handle range +- [m]
    double loop_rate_hz = 0.0;             ///< control loop rate [Hz]
};

/// The 1-DOF haptic paddle in MR list form (Python: the dict returned by course_models.hapkit()).
struct HapkitModel {
    Eigen::MatrixXd Slist;    ///< 6 x 1 screw axis (one per COLUMN)
    Eigen::Matrix4d M;        ///< home configuration of the handle frame {b}
    mr::TransformList Mlist;  ///< 2 frames M_{i-1,i} (CoM frame, MR Sec. 8.3.2)
    mr::InertiaList Glist;    ///< 1 spatial inertia (MR Eq. 8.32)
    Eigen::Vector3d g;        ///< gravity in {s} [m/s^2] (in the plane of motion: the paddle is vertical)
    Eigen::MatrixXd limits;   ///< joint limits [rad], 1 x 2 (min, max)
    Eigen::VectorXd qdot_max, tau_max;
    HapkitParams params;
};

/// Geometry of the rehabilitation robot (Python: rehab_2r()["params"]).
struct RehabParams {
    double L1 = 0.0, L2 = 0.0;  ///< link lengths [m]
    double m1 = 0.0, m2 = 0.0;  ///< link masses [kg]
    Eigen::Vector2d path_A, path_B;  ///< ends of the therapy path [m]
};

/// The planar 2R rehabilitation robot in MR list form (Python: the dict returned by course_models.rehab_2r()).
struct RehabModel {
    Eigen::MatrixXd Slist;    ///< 6 x 2 screw axes (one per COLUMN)
    Eigen::Matrix4d M;        ///< home configuration of the hand frame
    mr::TransformList Mlist;  ///< 3 frames M_{i-1,i}
    mr::InertiaList Glist;    ///< 2 spatial inertias
    Eigen::Vector3d g;        ///< gravity in {s} (along -z: horizontal plane, g(q) = 0)
    Eigen::MatrixXd limits;   ///< 2 x 2 (min, max) [rad]
    Eigen::VectorXd qdot_max, tau_max, q_home;
    RehabParams params;
};

/// Nominal device and loop parameters of data/device.json (Python: the dict returned by load_device()).
struct Device {
    double T = 0.0;          ///< sample time [s]
    double x_res = 0.0;      ///< position sensor resolution [m]
    double force_max = 0.0;  ///< force limit [N]
    double x_range = 0.0;    ///< handle range +- [m]
    double k_t = 0.0;        ///< motor torque constant [N m/A]
    double i_max = 0.0;      ///< current limit [A]
    double m_h = 0.0;        ///< nominal handle mass [kg]
    double b_h = 0.0;        ///< nominal handle damping [N s/m]
    double k_g = 0.0;        ///< nominal gravity stiffness [N/m]
    std::string note;
};

/// (wn, zeta) of a free-decay fit (Python: a tuple).  `auto [wn, ze] = lab::fit_free_decay(t, x);` works.
struct FreeDecayFit {
    double wn = 0.0, zeta = 0.0;
};

/// Result of identify_device (Python: dict(m_h, k_g, b_h, tau_eff, b_app)).
struct Identification {
    double m_h = 0.0, k_g = 0.0, b_h = 0.0, tau_eff = 0.0;
    Eigen::VectorXd b_app;  ///< apparent damping of each test [N s/m]
};

/// Result of simulate_device (Python: dict t, x, v, x_meas, v_est, F_cmd, F_applied, saturated).
/// Every vector has one entry per sample k = 0 .. n (n = round(T_end / T)).
struct DeviceSim {
    Eigen::VectorXd t, x, v, x_meas, v_est, F_cmd, F_applied;
    std::vector<bool> saturated;
};

/// Result of static_push_test (Python: dict(x_ss, K_rendered, error_pct)).
struct PushTest {
    double x_ss = 0.0, K_rendered = 0.0, error_pct = 0.0;
};

/// Result of simulate_rehab (Python: dict t, q, dq, p, v, tau, f).  Matrices are (n + 1) x 2, one ROW per sample.
struct RehabSim {
    Eigen::VectorXd t;
    Eigen::MatrixXd q, dq, p, v, tau, f;
};

/// law(t, x_meas, v_est) -> commanded force, and f_hand(t, x, v) -> hand force (simulate_device).
using ForceLaw = std::function<double(double t, double x, double v)>;
/// controller(t, q, dq, f_meas) -> tau (simulate_rehab).
using RehabController = std::function<Eigen::Vector2d(double t, const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                                      const Eigen::Vector2d& f_meas)>;
/// f_ext(t, p, v) -> hand force on the robot (simulate_rehab).
using HandForce = std::function<Eigen::Vector2d(double t, const Eigen::Vector2d& p, const Eigen::Vector2d& v)>;
/// x' = f(x) for rk4_step.
using StateFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd& x)>;

// ---------------------------------------------------------------------------
// Provided (given in helpers.cpp; do not edit): models, data files, rehab-robot kinematics and dynamics
// ---------------------------------------------------------------------------

/// Hapkit-type 1-DOF haptic paddle (course_models.hapkit, numbers identical).  Handle pivots about z_s; {b} at
/// the handle grip, rh = 0.090 m above the pivot; sector radius rs = 0.073 m, motor pulley rp = 0.0042 m
/// (ratio ~ 17.4).  Handle displacement x ~ rh theta.  Derated limits: 2 N, +-40 mm, 1 kHz loop.
/// ILLUSTRATIVE VALUES (teaching values, not data of a real product).
HapkitModel hapkit();

/// Planar end-effector rehabilitation robot in a horizontal plane (course_models.rehab_2r, numbers identical;
/// 461 project scenario S1).  ILLUSTRATIVE VALUES.
RehabModel rehab_2r();

/// The rehab robot used by the functions below (Python: the module constant REHAB = cm.rehab_2r()).
const RehabModel& rehab();

/// Nominal device and loop parameters from data/device.json (illustrative values): T (s), x_res (m),
/// force_max (N), k_t (N m/A), i_max (A) and the nominal identified m_h, b_h, k_g used in Parts B and C.
Device load_device();

/// A logged free-decay test: the handle is held at x0 and released while the device renders a spring
/// K (N/m) with B = 0 at 1 kHz.  Returns (t, x) in s and m (x quantised to the sensor resolution).
std::pair<Eigen::VectorXd, Eigen::VectorXd> load_free_decay(int K);

/// One classical Runge-Kutta step of x' = f(x) (inputs held constant over the step).
Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& x, double h);

/// Hand (end-effector) position p = (x, y) in {s} of the planar rehab robot (MR Example 4.1 style).
Eigen::Vector2d rehab_fk(const Eigen::Vector2d& q);

/// 2 x 2 Jacobian of the hand position, p' = J(q) q'  (linear-velocity rows of J_s for a planar arm).
Eigen::Matrix2d rehab_jacobian(const Eigen::Vector2d& q);

/// Time derivative J'(q, q') of rehab_jacobian.
Eigen::Matrix2d rehab_jacobian_dot(const Eigen::Vector2d& q, const Eigen::Vector2d& dq);

/// Elbow-positive (theta_2 > 0) inverse kinematics of the hand position p = (x, y).
Eigen::Vector2d rehab_ik(const Eigen::Vector2d& p);

/// Closed-form (M(q), c(q, q')) of the rehab robot (horizontal plane, g(q) = 0); identical to
/// mr::mass_matrix / mr::vel_quadratic_forces of rehab_2r() (checked by the tests).
std::pair<Eigen::Matrix2d, Eigen::Vector2d> rehab_dynamics(const Eigen::Vector2d& q, const Eigen::Vector2d& dq);

/// Simulate the rehab robot  M q'' + c = tau + J^T f_ext  (f_ext = force of the hand ON the robot).
///
/// controller(t, q, dq, f_meas) -> tau is called every dt (zero-order hold); tau is clipped to tau_max.
/// f_ext(t, p, v) -> hand force, evaluated inside the RK4 steps (empty std::function = no hand).
/// f_meas is the force-sensor reading at the sample: f_ext itself, or a first-order low-pass filtered
/// version with cutoff force_lag_hz (a sensor/filter lag; MR p. 447); force_lag_hz <= 0 means no lag (Python None).
RehabSim simulate_rehab(const RehabController& controller, const Eigen::Vector2d& q0, double T_end,
                        double dt = 0.001, const HandForce& f_ext = HandForce(),
                        const Eigen::Vector2d& dq0 = Eigen::Vector2d::Zero(), int substeps = 4,
                        double force_lag_hz = 0.0);

/// Spectral radius max |eig(A)|.
double spectral_radius(const Eigen::MatrixXd& A);

/// Ratio of the peak |x| in the last quarter of the run to the peak |x| in [t_from, t_from + quarter):
/// < 1 decaying, > 1 growing (a quick stability verdict from a simulation).
double oscillation_growth(const DeviceSim& sim, double t_from = 0.5);

/// Hand-force profile: amplitude * unit(direction) for t_on <= t < t_off, else 0 (a call-back for simulate_rehab).
HandForce pulse_force(const Eigen::Vector2d& direction, double amplitude, double t_on, double t_off);

/// Displacement of the 1-DOF impedance m x'' + b x' + k x = f(t) (MR Eq. 11.62) from rest for the rectangular
/// force pulse, sampled at the times t (exact: matrix exponential of the state equation).
Eigen::VectorXd one_dof_pulse_response(double m, double b, double k, double amplitude, double t_on, double t_off,
                                       const Eigen::VectorXd& t);

// ---- C++ only (the Python lab uses SciPy / NumPy for these) ----

/// Matrix exponential e^A (= scipy.linalg.expm; scaling and squaring with a Pade approximant).
Eigen::MatrixXd expm(const Eigen::MatrixXd& A);

/// Indices of the local maxima of x that are at least `distance` samples apart, in increasing order
/// (= scipy.signal.find_peaks(x, distance=distance)[0]: flat peaks give their middle index; when two peaks are
/// closer than `distance`, the higher one is kept; of two EQUAL heights the later one is kept -- SciPy's choice
/// for exactly equal heights depends on NumPy's sort and may differ; the lab's data files give the same peaks).
std::vector<Eigen::Index> find_peaks(const Eigen::VectorXd& x, int distance);

/// Least-squares straight line y ~ slope x + intercept (= np.polyfit(x, y, 1)).  Returns (slope, intercept).
std::pair<double, double> linear_fit(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

// ---------------------------------------------------------------------------
// Part A -- the haptic device as a 1-DOF plant
// ---------------------------------------------------------------------------

/// Effective translational mass m_h (kg) felt at the handle at theta = 0:
/// m_h = (I_pivot + J_rotor * ratio^2) / rh^2, with I_pivot the link inertia about the joint axis
/// (= mr::mass_matrix at theta = 0) and the rotor inertia reflected through the transmission ratio rs/rp
/// (MR Sec. 8.9.2, apparent inertia G^2 J).
double handle_mass(const HapkitModel& model);

/// Translational stiffness k_g (N/m) at the handle caused by gravity on the sector mass below the pivot:
/// k_g = (d g(theta)/d theta at 0) / rh^2, with g(theta) = mr::gravity_forces.
double gravity_stiffness(const HapkitModel& model);

/// Motor current I (A) that produces the handle force F (N) (MR Sec. 11.8, current control of a geared motor):
/// joint torque tau = F rh, motor torque tau_m = tau / ratio, I = tau_m / k_t (element by element).
/// Friction in the transmission is ignored (that is exactly the error MR p. 448 warns about).
Eigen::VectorXd handle_force_to_current(const Eigen::VectorXd& F, const HapkitModel& model, double k_t);
/// Scalar F (given: calls the vector version).
double handle_force_to_current(double F, const HapkitModel& model, double k_t);

/// Natural frequency wn (rad/s) and damping ratio zeta of a logged free decay x(t) (rest at x = 0).
///
/// Method: positive peaks (lab::find_peaks with a minimum spacing of half the dominant period), keep peaks
/// above min_fraction of the first one; damped period T_d = mean spacing; logarithmic decrement
/// delta = -slope of ln(peak) against peak number (least squares);
/// zeta = delta / sqrt(4 pi^2 + delta^2); wn = (2 pi / T_d) / sqrt(1 - zeta^2).
FreeDecayFit fit_free_decay(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double min_fraction = 0.1);

/// Identify the device from free decays at several rendered stiffnesses Ks with fits [(wn, zeta), ...].
///
/// Model of each test:  m_h x'' + b_app x' + (K + k_g) x = 0,  so  wn^2 = K/m_h + k_g/m_h  and
/// b_app = 2 zeta wn m_h.  The sample-and-hold of the rendered spring acts like a delay tau_eff, which to first
/// order removes damping K tau_eff:  b_app = b_h - tau_eff K.  Two straight-line fits give
/// m_h = 1/slope(wn^2 vs K), k_g = intercept * m_h, b_h = intercept(b_app vs K), tau_eff = -slope.
Identification identify_device(const Eigen::VectorXd& Ks, const std::vector<FreeDecayFit>& fits);

// ---------------------------------------------------------------------------
// Part B -- rendering springs, dampers and walls with a sampled-data loop
// ---------------------------------------------------------------------------

/// Force rendered on the hand by a virtual spring-damper about (x0, v0):  F = -K (x - x0) - B (v - v0)
/// (MR Eq. 11.62 without the mass term: impedance control with M = 0, MR p. 446).
double spring_damper_force(double x, double v, double K, double B, double x0 = 0.0, double v0 = 0.0);

/// Unilateral virtual wall occupying side*(x - x_wall) > 0.  Outside the wall F = 0.  Inside,
/// F = -K (x - x_wall) - B v, but the wall may only PUSH the hand out (never pull it in): the result is
/// clipped so that side*F <= 0.
double wall_force(double x, double v, double K, double B, double x_wall, int side = +1);

/// Sampled-data simulation of the haptic device rendering `law`.
///
/// Plant (continuous):  m_h x'' + b_h x' + k_g x = F_applied + f_hand(t, x, v)  with dev.m_h, dev.b_h, dev.k_g;
/// integrated with `substeps` RK4 steps per sample (F_applied held: zero-order hold; f_hand evaluated inside the
/// steps with t = t_k; an empty f_hand = no hand).
/// Controller every T = dev.T at t_k = k T (k = 0 .. n, n = round(T_end / T)):
///   1. x_meas = x rounded to the sensor resolution dev.x_res (if quantize), else x;
///   2. v_est = (x_meas_k - x_meas_{k-1}) / T  (backward difference; v_est = 0 at k = 0);
///   3. F_cmd = law(t_k, x_meas, v_est), saturated to +-dev.force_max (saturated[k] = |law| > force_max);
///   4. F_applied during [t_k, t_k+1) is the F_cmd computed delay_steps samples earlier (0 before that).
DeviceSim simulate_device(const Device& dev, const ForceLaw& law, double T_end, double x0 = 0.0, double v0 = 0.0,
                          const ForceLaw& f_hand = ForceLaw(), int delay_steps = 0, bool quantize = true,
                          int substeps = 10);

/// Rendered stiffness from a static push: a constant hand force F_load (e.g. a hanging weight) acts on the device
/// rendering a spring K, damper B about x0 = 0.  The steady deflection is the mean of x over the last 0.2 s;
/// K_rendered = F_load / x_ss.  Returns (x_ss, K_rendered, error_pct = 100 (K_rendered - K)/K).
/// (K_rendered includes the small gravity stiffness k_g of the paddle.)
PushTest static_push_test(const Device& dev, double K, double B, double F_load, double T_end = 2.0);

/// Exact discrete-time closed-loop matrix of the linear sampled-data loop of simulate_device
/// (no quantisation, no saturation).
///
/// Plant ZOH discretisation of  m x'' + b x' + k_g x = F  (A_c = [[0, 1], [-k_g/m, -b/m]], B_c = [0, 1/m]):
/// A_d = expm(A_c T), B_d = int_0^T expm(A_c s) ds B_c (use the augmented-matrix exponential, lab::expm).
/// Controller c_k = -K x_k - B (x_k - x_{k-1}) / T; applied force F_k = c_{k-d}, d = delay_steps.
/// State z_k = [x_k, v_k, x_{k-1}, c_{k-1}, ..., c_{k-d}] (dimension 3 + d).
Eigen::MatrixXd closed_loop_matrix(double m, double b, double k_g, double K, double B, double T,
                                   int delay_steps = 0);

/// Largest K (N/m) for which closed_loop_matrix(...) has spectral radius < 1 (bisection on [0, K_hi], to within
/// tol).  Assumes the loop is stable at K = 0 and loses stability once as K grows.
double max_stable_K(double m, double b, double k_g, double B, double T, int delay_steps = 0, double K_hi = 1e5,
                    double tol = 0.01);

/// First-order (Taylor) estimate of the stiffness limit: the spring force is late by the zero-order hold (T/2)
/// plus the loop delay (d T), and a late spring  K x(t - tau) ~ K x - K tau x'  removes damping K tau.
/// The loop stays damped while b + B - K tau > 0:  K_max ~ (b + B) / (T/2 + d T).
double first_order_K_limit(double b, double B, double T, int delay_steps = 0);

/// Total energy of device + rendered spring at the samples: E = 1/2 m_h v^2 + 1/2 (K + k_g)(x - x0)^2
/// (k_g about x = 0, i.e. E = 1/2 m_h v^2 + 1/2 K (x - x0)^2 + 1/2 k_g x^2).
/// For a passive rendering with no hand E must not grow.
Eigen::VectorXd device_energy(const Device& dev, const DeviceSim& sim, double K, double x0 = 0.0);

// ---------------------------------------------------------------------------
// Part C -- the rehabilitation robot: apparent inertia, impedance and admittance control
// ---------------------------------------------------------------------------

/// Task-space (apparent) inertia of the hand, Lambda = J^-T M J^-1  (MR Eq. 8.90), 2 x 2 (kg).
Eigen::Matrix2d task_inertia(const Eigen::Vector2d& q);

/// Mass felt when the hand is pushed along the direction u (normalise it first):  m_u = 1 / (u^T Lambda^-1 u)
/// (the acceleration along u per unit force along u is u^T Lambda^-1 u).
double apparent_mass(const Eigen::Vector2d& q, const Eigen::Vector2d& u);

/// Impedance (stiffness + damping) control without inertia shaping or force sensing:
/// tau = J^T ( K (p_d - p) + B (v_d - v) )  -- MR Eq. 11.65 with the mass term and the arm-dynamics
/// compensation dropped (MR p. 446: 'it is not uncommon to eliminate the mass compensation term ... and to
/// set M = 0'); g(q) = 0 in the horizontal plane.  K, B: 2 x 2 matrices.
Eigen::Vector2d impedance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                         const Eigen::Vector2d& p_d, const Eigen::Vector2d& v_d,
                                         const Eigen::Matrix2d& K, const Eigen::Matrix2d& B);
/// Scalar K, B (given: calls the matrix version with K I, B I).
Eigen::Vector2d impedance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                         const Eigen::Vector2d& p_d, const Eigen::Vector2d& v_d, double K, double B);

/// Admittance control (MR Sec. 11.7.2): desired hand acceleration from the sensed force,
/// a_d = M_d^-1 (f_meas - B_d v - K_d (p - p0))  (MR Eq. 11.66 with the spring about p0); joint acceleration
/// q''_d = J^-1 (a_d - J' q') (J is square here, J^dagger = J^-1); inverse dynamics including the sensed
/// hand force:  tau = M(q) q''_d + c(q, q') - J^T f_meas.  M_d, B_d, K_d: 2 x 2 matrices.
Eigen::Vector2d admittance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                          const Eigen::Vector2d& f_meas, const Eigen::Vector2d& p0,
                                          const Eigen::Matrix2d& M_d, const Eigen::Matrix2d& B_d,
                                          const Eigen::Matrix2d& K_d);
/// Scalar M_d, B_d, K_d (given: calls the matrix version with M_d I, B_d I, K_d I).
Eigen::Vector2d admittance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                          const Eigen::Vector2d& f_meas, const Eigen::Vector2d& p0, double M_d,
                                          double B_d, double K_d);

}  // namespace lab

#endif  // SBEG461_LAB8_HPP
