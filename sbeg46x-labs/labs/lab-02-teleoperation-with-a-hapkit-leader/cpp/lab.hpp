// SBEG 46X Lab 2 -- Teleoperation with a Hapkit leader: scaling, clutch, tremor filter, two-port models
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided helpers (load_params, load_log, minjerk, reach_intent, simulate_teleop, jacobian_body, the two
// course models hapkit() and eod_6r_ugv(), and the array versions of saturate_force / hybrid_matrix) are in
// helpers.cpp (given code, do not edit).
//
// Conventions (lecture notes, Week 2):
//   leader  : Z_m v_m = f_h + a_m          (Z_m = m_m s + b_m; f_h = hand force on the handle; a_m = leader motor force)
//   follower: Z_s v_s = a_s - f_e          (f_e = force of the follower on the environment, f_e = Z_e v_e)
//   hybrid matrix  [f_h, -v_e]^T = H [v_h, f_e]^T,  ideal (scaled) transparency H = [[0, beta], [-alpha, 0]]
//   controller C(s) = b_c + k_c / s (PD on position, written as an impedance)
//   P-P: a_s = C(alpha v_m - v_s),   a_m = -beta * a_s
//   P-F: a_s = C(alpha v_m - v_s),   a_m = -beta * f_e   (follower force sensor)
// All parameter values are illustrative.  Units: SI.
//
// Python -> C++: the parameter dict p -> lab::Params (same field names; p["raw"]["environments"]["wall"] ->
// p.environments.at("wall"), p["raw"]["reach_task"] -> p.reach_task); dict(p, beta=b) -> a copy `Params q = p;
// q.beta = b;`; the log dict -> lab::LeaderLog; the model dicts -> lab::HapkitModel / lab::EodModel; the result
// dict of simulate_teleop -> lab::TeleopSim; tuples (b, a) and (a_m, a_s) -> std::pair; complex numbers ->
// std::complex<double>, a 2x2 complex matrix -> Eigen::Matrix2cd, "s may be an array" -> a given overload taking
// Eigen::VectorXcd and returning std::vector<Eigen::Matrix2cd>; boolean arrays -> std::vector<bool>;
// math.inf -> std::numeric_limits<double>::infinity(); ValueError -> std::invalid_argument.
#ifndef SBEG46X_LAB2_HPP
#define SBEG46X_LAB2_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <complex>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A unilateral environment: force only when x_s > x_wall, f_e = max(0, k (x_s - x_wall) + b v_s).
struct Environment {
    double k = 0.0;       ///< stiffness [N/m]
    double b = 0.0;       ///< damping [N s/m]
    double x_wall = 0.0;  ///< wall position [m]
};

/// Bilateral reach task of Part D (Python: p["raw"]["reach_task"]).
struct ReachTask {
    double x_start = 0.0, x_goal = 0.0, t_move = 0.0, t_end = 0.0;
};

/// Python: p["raw"]["laparoscopy"].
struct Laparoscopy {
    double alpha = 0.0, insertion_m = 0.0, stroke_m = 0.0;
};

/// Python: p["raw"]["tremor"] (band_hz = [band_lo, band_hi]).
struct Tremor {
    double f_tremor_hz = 0.0, band_lo = 0.0, band_hi = 0.0, voluntary_max_hz = 0.0;
};

/// Python: p["raw"]["eod"].
struct EodTask {
    double alpha = 0.0, approach_m = 0.0, stroke_m = 0.0, speed_fraction = 0.0;
};

/// The lab parameters (Python: the flat dict returned by load_params(); the fields of p["raw"] that the lab uses
/// are the members environments, reach_task, laparoscopy, tremor, eod).
struct Params {
    double m_m = 0.0, b_m = 0.0;            ///< leader mass [kg] and damping [N s/m]
    double m_s = 0.0, b_s = 0.0;            ///< follower mass and damping
    double m_h = 0.0, b_h = 0.0, k_h = 0.0; ///< hand mass, damping, stiffness
    double kc = 0.0, bc = 0.0;              ///< controller C(s) = bc + kc / s
    double alpha = 0.0, beta = 0.0;         ///< motion and force scales
    double f_max = 0.0;                     ///< motor force limit [N]
    double x_range = 0.0;                   ///< watchdog range +- [m]
    double dt = 0.0;                        ///< control period [s]
    int substeps = 1;                       ///< integration substeps per control period
    std::map<std::string, Environment> environments;  ///< "free", "tissue", "wall"
    ReachTask reach_task;
    Laparoscopy laparoscopy;
    Tremor tremor;
    EodTask eod;
};

/// A leader log (columns t, x_m, clutch of data/leader_log.csv).
struct LeaderLog {
    Eigen::VectorXd t, x_m;
    std::vector<bool> clutch;
};

/// Python: hapkit()["params"].
struct HapkitParams {
    double rh = 0.0, rs = 0.0, rp = 0.0, ratio = 0.0;
    double rotor_inertia = 0.0, reflected_rotor_inertia = 0.0;
    double force_max = 0.0, x_range = 0.0, loop_rate_hz = 0.0;
};

/// Hapkit-type 1-DOF haptic paddle (Python: the dict returned by course_models.hapkit()).
struct HapkitModel {
    Eigen::MatrixXd Slist;   ///< 6 x 1 screw axis (one per COLUMN)
    Eigen::Matrix4d M;       ///< home configuration of the grip
    double mass = 0.0;       ///< moving mass of handle + sector [kg]
    Eigen::Vector3d com;     ///< centre of mass at home in {s} [m]
    Eigen::Matrix3d I_com;   ///< rotational inertia about the CoM [kg m^2]
    Eigen::Vector3d g;       ///< gravity in {s}
    Eigen::MatrixXd limits;  ///< 1 x 2 (min, max) [rad]
    HapkitParams params;
};

/// Python: eod_6r_ugv()["params"].
struct EodParams {
    double shoulder_a = 0.0, shoulder_h = 0.0;  ///< Python: shoulder = (a, h)
    double L1 = 0.0, L2 = 0.0, Lg = 0.0;
    Eigen::VectorXd q_stow, q_ready;            ///< [rad]
    double qddot_max = 0.0;                     ///< [rad/s^2]
};

/// UGV-mounted EOD 6R arm (Python: the dict returned by course_models.eod_6r_ugv()).
struct EodModel {
    Eigen::MatrixXd Slist, Blist;  ///< 6 x 6 screw axes (one per COLUMN), space and body form
    Eigen::Matrix4d M;
    Eigen::MatrixXd limits;        ///< 6 x 2 (min, max) [rad]
    Eigen::VectorXd qdot_max;      ///< rated joint speeds [rad/s]
    EodParams params;
};

/// Result of simulate_teleop (Python: dict t, x_m, v_m, x_s, v_s, f_h, f_e, a_m, a_s, tripped).
struct TeleopSim {
    Eigen::VectorXd t, x_m, v_m, x_s, v_s, f_h, f_e, a_m, a_s;
    bool tripped = false;
};

/// control_forces(arch, xm, vm, xs, vs, fe, p) -> (a_m, a_s); an empty ControlLaw means lab::control_forces.
using ControlLaw = std::function<std::pair<double, double>(const std::string& arch, double xm, double vm, double xs,
                                                           double vs, double fe, const Params& p)>;

// ---------------------------------------------------------------------------
// Provided helpers (helpers.cpp, given)
// ---------------------------------------------------------------------------

/// Read the lab parameters (default: data/teleop_params.json of this lab).
Params load_params(const std::string& path = "");

/// Read a leader log (columns t, x_m, clutch; default: data/leader_log.csv of this lab).
LeaderLog load_log(const std::string& path = "");

/// Minimum-jerk (quintic) rest-to-rest profile from x0 at t0 to x1 at t1 (clipped outside [t0, t1]).
Eigen::VectorXd minjerk(const Eigen::VectorXd& t, double t0, double t1, double x0, double x1);

/// Intended hand trajectory of the bilateral reach task (Part D).
Eigen::VectorXd reach_intent(const Params& p, const Eigen::VectorXd& t);

/// Sampled-data simulation of the hand + leader + follower + environment (provided).
///   arch    : "pp" or "pf"
///   env     : unilateral contact, force only when x_s > x_wall
///   x_i     : intended hand position at each control sample [m]; the hand is a spring-damper k_h, b_h pulling
///             the handle towards x_i; its mass m_h is lumped with the handle
///   control : control_forces(arch, xm, vm, xs, vs, fe, p) -> (a_m, a_s); empty: lab::control_forces
/// dt = 0 / substeps = 0 mean p.dt / p.substeps.  The controller runs at 1/dt with zero-order hold; the mechanics
/// are integrated with `substeps` semi-implicit Euler steps per sample.  Motor forces are saturated at f_max and
/// the watchdog sets both motor forces to zero for the rest of the run if |x_m| > x_range (tripped = true).
TeleopSim simulate_teleop(const std::string& arch, const Params& p, const Environment& env, const Eigen::VectorXd& x_i,
                          double dt = 0.0, int substeps = 0, bool saturate = true, bool watchdog = true,
                          const ControlLaw& control = ControlLaw());

/// Body Jacobian J_b(theta) (MR Eq. 5.18), provided; your 361 library has the same function.
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& q);

/// Hapkit-type 1-DOF haptic paddle (numbers identical to Course resources: "Course robot models and hardware
/// safety checklist").  Grip rh = 0.090 m from the pivot; sector rs = 0.073 m; motor pulley rp = 0.0042 m.
HapkitModel hapkit();

/// UGV-mounted EOD 6R arm (kinematic).  At home the arm is stretched along +x_s and the gripper approach axis is
/// x_b.  Rated joint speed 30 deg/s; ready pose theta_r = (0, 30, -70, 0, 10, 0) deg.  q = 0 is singular.
EodModel eod_6r_ugv(double shoulder_a = 0.10, double shoulder_h = 0.20, double L1 = 0.80, double L2 = 0.75,
                    double Lg = 0.25);

/// Scalar version of saturate_force (given: it calls your vector version).
double saturate_force(double f, double f_max = 2.0);

/// hybrid_matrix for several values of s (given: it calls your scalar version once per entry).
std::vector<Eigen::Matrix2cd> hybrid_matrix(const std::string& arch, const Eigen::VectorXcd& s, const Params& p);

// ---------------------------------------------------------------------------
// Part A -- mapping, clutch, device safety
// ---------------------------------------------------------------------------

/// A1. Effective translational mass of the Hapkit at the handle grip (kg):
/// m_eff = (I_zz,com + m d^2 + I_rotor (rs/rp)^2) / rh^2, d = distance of the CoM from the pivot axis.
double hapkit_effective_mass(const HapkitModel& model);

/// A2. Saturate a commanded handle force at +-f_max (N), entry by entry.
Eigen::VectorXd saturate_force(const Eigen::VectorXd& f, double f_max = 2.0);

/// A3. True where the handle is inside the allowed range |x| <= x_range (m).
std::vector<bool> watchdog_ok(const Eigen::VectorXd& x, double x_range = 0.040);

/// A4. Follower set-point with motion scaling and clutching (indexing).
/// While clutch[k] is true the follower set-point holds.  At every release (and at k = 0 if not clutched)
/// the offsets are re-latched: x_sd = x_s,latch + alpha (x_m - x_m,latch), so the follower never jumps.
Eigen::VectorXd map_with_clutch(const Eigen::VectorXd& x_m, const std::vector<bool>& clutch, double alpha,
                                double x_s0 = 0.0);

/// A5. Limit the rate of change of a set-point to |dx/dt| <= v_max (first sample unchanged).
Eigen::VectorXd rate_limit(const Eigen::VectorXd& x_d, double v_max, double dt);

/// A6. Minimum number of leader strokes (forward motions between clutch actions) for a follower travel.
int clutch_strokes(double travel, double alpha, double stroke);

// ---------------------------------------------------------------------------
// Part B -- tremor filtering
// ---------------------------------------------------------------------------

/// B1. First-order low-pass, exact discretisation of 1/(tau s + 1), tau = 1/(2 pi fc):
/// y_k = y_{k-1} + a (x_k - y_{k-1}), a = 1 - exp(-2 pi fc dt), y_0 = x_0.
Eigen::VectorXd lowpass1(const Eigen::VectorXd& x, double fc, double dt);

/// B2. Second-order Butterworth low-pass (bilinear transform with pre-warping).  Returns (b, a), a[0] = 1,
/// each a 3-vector.
std::pair<Eigen::VectorXd, Eigen::VectorXd> butter2(double fc, double dt);

/// B3. Apply a causal IIR filter (difference equation, a[0] = 1), starting at rest at x[0]
/// (past inputs and outputs equal to x[0]; valid for a low-pass with unit DC gain).
Eigen::VectorXd iir_filter(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::VectorXd& x);

/// B4. RMS of the part of x with frequency in [f_lo, f_hi] Hz.  First remove the straight line joining the
/// first and last samples and the mean (so the record has no jump when the FFT repeats it), then use the
/// one-sided DFT and Parseval: rms^2 = 2 sum_band |X_k|^2 / N^2 (bins with 0 < f_k < 1/(2 dt) only).
double band_rms(const Eigen::VectorXd& x, double dt, double f_lo, double f_hi);

/// B5. Time (s) by which y lags x: the shift k*dt (k = 0 .. max_lag/dt) that minimises the mean squared
/// difference between y[k:] and x[:N-k].
double lag_estimate(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double dt, double max_lag = 0.5);

// ---------------------------------------------------------------------------
// Part C -- two-port models
// ---------------------------------------------------------------------------

/// C1. Hybrid matrix H(s) of the scaled P-P ("pp") or P-F ("pf") architecture, [f_h, -v_e] = H [v_h, f_e].
/// Throw std::invalid_argument for another arch.  (An overload for a vector of s values is given.)
Eigen::Matrix2cd hybrid_matrix(const std::string& arch, std::complex<double> s, const Params& p);

/// C2. Impedance felt by the operator, Z_t = f_h / v_h, when the follower touches Z_e:
/// Z_t = (h11 + (h11 h22 - h12 h21) Z_e) / (1 + h22 Z_e).
std::complex<double> transmitted_impedance(const Eigen::Matrix2cd& H, std::complex<double> Ze);

/// C3. Llewellyn's third condition 2 Re h11 Re h22 - Re(h12 h21) - |h12 h21| (>= 0 needed for absolute
/// stability, together with Re h11 >= 0, Re h22 >= 0 and stable h11, h22).
double llewellyn(const Eigen::Matrix2cd& H);

// ---------------------------------------------------------------------------
// Part D -- bilateral control in time
// ---------------------------------------------------------------------------

/// D1. Motor forces (a_m on the leader, a_s on the follower) before saturation; returns (a_m, a_s).
/// a_s = k_c (alpha x_m - x_s) + b_c (alpha v_m - v_s);  P-P: a_m = -beta a_s;  P-F: a_m = -beta f_e.
/// Throw std::invalid_argument for another arch.
std::pair<double, double> control_forces(const std::string& arch, double xm, double vm, double xs, double vs,
                                         double fe, const Params& p);

/// D2. A matrix of the linear in-contact model, state (x_m, v_m, x_s, v_s), wall at x = 0, hand intent 0:
/// (m_m + m_h) dv_m = -k_h x_m - (b_h + b_m) v_m + a_m,   m_s dv_s = a_s - b_s v_s - (k_e x_s + b_e v_s).
Eigen::Matrix4d closed_loop_matrix(const std::string& arch, const Params& p, double ke, double be);

/// D3. Largest force scale beta for which the in-contact linear model is stable (max Re(eig) < 0).
/// Returns +infinity if it is stable up to beta_hi.
double max_stable_beta(const std::string& arch, const Params& p, double ke, double be, double beta_hi = 1000.0);

// ---------------------------------------------------------------------------
// Part E -- EOD follower speed limit
// ---------------------------------------------------------------------------

/// E1. Joint rates (rad/s) that move the gripper at 1 m/s along its approach axis x_b with no rotation:
/// theta_dot = J_b^{-1} (0, 0, 0, 1, 0, 0).
Eigen::VectorXd eod_approach_rates(const EodModel& model, const Eigen::VectorXd& q);

/// E2. Largest approach speed (m/s) for which every joint stays within speed_fraction * qdot_max.
double max_approach_speed(const EodModel& model, const Eigen::VectorXd& q, double speed_fraction = 1.0);

}  // namespace lab

#endif  // SBEG46X_LAB2_HPP
