// SBEG 46X Lab 3 -- Delayed bilateral teleoperation: instability, wave variables, time-domain passivity
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Leader and follower are Hapkit-type 1-DOF devices (the Hapkit model of course_models.py).  Units: SI (s, m, kg, N).
// Velocities are written xdot (xdot_m, xdot_s); the letters u and v are the forward and returning WAVES.
//
// Sign conventions used everywhere
//   * Leader (the operator's hand mass m_h is lumped with the handle mass m_m):
//         (m_m + m_h) xddot_m = f_h - F_m - b_m xdot_m,   f_h = k_h (x_i - x_m) + b_h (xdot_i - xdot_m)
//     F_m is the force of the teleoperator on the leader; it RESISTS the operator when F_m > 0 and xdot_m > 0.
//     Power INTO the communication channel at the leader port:  P_m = F_m xdot_m.
//   * Follower:  m_s xddot_s = F_c - f_e - b_s xdot_s,   F_c = Kp (x_d - x_s) - Kd xdot_s.
//     x_d is the follower set-point delivered by the channel.  Power OUT of the channel at the follower port:
//     P_s = F_c xdot_d (xdot_d = rate of change of x_d).
//   * Channel energy  E_ch = E_in,m + E_in,s - E_out,m - E_out,s;  E_ch >= 0 at all times <=> passive channel.
//   * Week 2 sign: Lab 2 writes the leader motor force a_m = -F_m and the follower motor force a_s = F_c.
//
// Architectures (arch = "pp", "pf", "wave", "tdpa")
//   pp    position-position: x_d(t) = x_m(t - T),  F_m(t) = F_c(t - T)        [Week 2 P-P with alpha = beta = 1]
//   pf    position-force:    x_d(t) = x_m(t - T),  F_m(t) = f_e(t - T)        [Week 2 P-F; Anderson & Spong 1989]
//   wave  wave variables with wave impedance b (Niemeyer & Slotine 1991):
//         u = (b xdot + F)/sqrt(2b),  v = (b xdot - F)/sqrt(2b),  u_s(t) = u_m(t - T),  v_m(t) = v_s(t - T)
//   tdpa  the pp channel plus a passivity observer and a passivity controller at each port
//         (Hannaford & Ryu 2002):  E_out,s(n) <= E_in,m(n - N)   and   E_out,m(n) <= E_in,s(n - N)
//
// Python -> C++: ValueError -> std::invalid_argument; the parameter dict -> lab::Params; the result dict of
// simulate -> lab::SimResult; the dict of performance_metrics -> lab::Metrics; 2-tuples -> std::pair;
// 4-tuples -> small structs with the Python names; keyword arguments of simulate -> lab::SimOptions;
// complex -> std::complex<double>; math.inf / math.nan -> std::numeric_limits<double>::infinity() / quiet_NaN().
#ifndef SBEG46X_LAB3_HPP
#define SBEG46X_LAB3_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <complex>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ==========================================================================================
// Provided (do not edit): parameters, hand intent, network model, simulator
// ==========================================================================================

struct Device {
    double m = 0.0;  ///< effective handle mass (kg)
    double b = 0.0;  ///< viscous damping (N s/m)
};
struct Hand {
    double m = 0.0, b = 0.0, k = 0.0;  ///< hand mass (lumped with the leader), damping, stiffness
};
struct Controller {
    double Kp = 0.0, Kd = 0.0;  ///< follower PD gains
};
struct Environment {
    double k = 0.0, b = 0.0, x_wall = 0.0;  ///< Kelvin-Voigt contact at x_wall (k = 0: free space)
};
struct ReachTask {
    double x_start = 0.0, x_goal = 0.0, t_move = 0.0, t_end = 0.0;
};
struct WaveParams {
    double b = 0.0;      ///< wave impedance (N s/m)
    double tau_f = 0.0;  ///< time constant (s) of the passive low-pass filter on the wave arriving at the leader
};

/// The lab parameters of data/teleop.json (same field names as the JSON keys).
struct Params {
    double dt = 0.0;
    int substeps = 0;
    double force_max = 0.0, x_range = 0.0;
    Device leader, follower;
    Hand hand;
    Controller controller;
    std::map<std::string, Environment> environments;  ///< "free", "tissue", "wall"
    ReachTask reach_task;
    WaveParams wave;
};

/// Read the lab parameters (data/teleop.json, illustrative values; the same devices, hand, gains, environments and
/// reach task as Lab 2). An empty path means the lab's data/teleop.json.
Params load_params(const std::string& path = "");

/// Minimum-jerk (quintic) rest-to-rest profile from x0 at t0 to x1 at t1 (as in Lab 2), for every entry of t.
Eigen::VectorXd minjerk(const Eigen::VectorXd& t, double t0, double t1, double x0, double x1);

/// Intended hand trajectory: reach from x_start to x_goal in t_move s, then hold (pushes into the wall).
Eigen::VectorXd reach_intent(const Params& p, const Eigen::VectorXd& t);

/// One direction of a packet network, sampled at the control period dt.
///
/// A packet sent at sample n arrives at sample n + d[n] (d[n] >= 1; d[n] < 0 means the packet is lost).
/// The receiver uses the most recently SENT packet that has arrived (older, out-of-order packets are
/// discarded) and holds it until a newer one arrives: the usual zero-order hold of a teleoperation link.
/// Before the first packet arrives, receive() returns `initial`.
class DelayLine {
public:
    /// One delay per sample (at least n_steps of them); throws std::invalid_argument otherwise or if a delay is 0.
    DelayLine(const std::vector<int>& delays, int n_steps, const Eigen::VectorXd& initial);
    /// The same constant delay (samples) for every packet.
    DelayLine(int delay, int n_steps, const Eigen::VectorXd& initial);
    void send(int n, const Eigen::VectorXd& value);
    Eigen::VectorXd receive(int n);

private:
    std::vector<int> d_;
    Eigen::VectorXd initial_;
    std::map<int, Eigen::VectorXd> values_;
    std::map<int, std::vector<int>> arrivals_;
    int best_ = -1;
};

/// Per-packet delays in samples: mean one-way delay T (s), uniform jitter +-jitter (s), loss probability.
/// The minimum is one sample (a digital link always delays by at least one control period).
/// C++ note: with jitter or loss the random draws come from std::mt19937(seed), NOT from NumPy's generator, so the
/// packets differ from the Python run with the same seed (the statistics are the same).
std::vector<int> network_delays(double T, double dt, int n_steps, double jitter = 0.0, double loss = 0.0,
                                int seed = 3);

/// Unilateral Kelvin-Voigt contact (never pulls): f_e = max(0, k (x - x_wall) + b xdot) for x > x_wall.
double contact_force(double x, double xdot, const Environment& env);

/// The keyword arguments of the Python simulate() (same names and defaults).
struct SimOptions {
    std::string env = "tissue";             ///< key of Params::environments ...
    std::optional<Environment> env_custom;  ///< ... or an explicit environment (Python: env given as a dict)
    std::optional<double> t_end;            ///< default p.reach_task.t_end
    std::optional<double> b_wave;           ///< default p.wave.b
    double jitter = 0.0;
    double loss = 0.0;
    int seed = 3;
    bool saturate = true;
    bool watchdog = true;
    /// C++ only: explicit packet delays (samples) for the forward / backward line, e.g. the Python draws stored in
    /// cpp_cases.json. Empty = network_delays(T, dt, n, jitter, loss, seed) and (..., seed + 1).
    std::vector<int> fwd_delays, bwd_delays;
};

/// The arrays returned by simulate() (one value per control sample) and the scalars tripped, N, arch, T.
struct SimResult {
    Eigen::VectorXd t, x_i, x_m, xdot_m, x_s, xdot_s, x_d, F_m, F_c, f_e, P_m, P_s, E_ch, W_m, W_s, r_m, r_s;
    bool tripped = false;
    int N = 0;  ///< one-way delay in samples
    std::string arch;
    double T = 0.0;
};

/// Sampled-data simulation of hand + leader + delayed channel + follower + environment (provided).
///
/// arch   "pp"   position-position: x_d(t) = x_m(t - T), F_m(t) = F_c(t - T) (follower controller force)
///        "pf"   position-force:    x_d(t) = x_m(t - T), F_m(t) = f_e(t - T) (follower force sensor)
///        "wave" wave variables (impedance b_wave, default p.wave.b); the follower integrates the decoded velocity
///               command; the wave arriving at the leader is low-pass filtered (time constant p.wave.tau_f): a
///               filter with |H(jw)| <= 1 cannot add wave energy, and it removes the undamped wave reflections at
///               1/(4T) that grow at very short delays
///        "tdpa" the "pp" channel with a passivity observer and controller at both ports
/// T      one-way delay (s), equal in both directions; jitter (s) and loss (probability) per packet.
/// The controller runs at 1/dt with zero-order hold; the mechanics are integrated with `substeps` semi-implicit
/// Euler steps per sample.  As on the course Hapkits, motor forces are saturated at force_max (saturate = true) and
/// the watchdog switches both motor forces off for the rest of the run if a handle leaves +-x_range
/// (watchdog = true; delay is emulated in software, never by disabling the watchdog).
/// It calls YOUR follower_pd, wave_leader, wave_follower, tdpa_leader and tdpa_follower.
/// Throws std::invalid_argument for an unknown arch or environment.
SimResult simulate(const std::string& arch, double T, const Params& p, const SimOptions& opt = SimOptions());
/// The same with the parameters of data/teleop.json (Python: p=None).
SimResult simulate(const std::string& arch, double T, const SimOptions& opt = SimOptions());

// ==========================================================================================
// Part A -- why force reflection becomes unstable: the delay margin
// ==========================================================================================

/// Stiffness reflected to the leader when the follower presses on the wall: Kp in series with k_e.
double contact_stiffness(double Kp, double k_e);

/// Smallest round-trip delay tau* for which  m s^2 + b s + k + kp e^{-s tau} = 0  has a root on the imaginary
/// axis (the delayed spring kp reflected to a leader of mass m, damping b, stiffness k).
///
/// At s = j w the delay term has magnitude kp, so |k - m w^2 + j b w| = kp gives
///     m^2 w^4 + (b^2 - 2 k m) w^2 + k^2 - kp^2 = 0;
/// for each positive root w the phase condition e^{-j w tau} = -(k - m w^2 + j b w)/kp gives the smallest tau > 0.
/// Returns (tau*, w_c) with the crossing frequency w_c (rad/s), or (inf, NaN) if no root exists (then the loop is
/// stable for every delay).  m = 0 is allowed (first-order leader).
std::pair<double, double> delay_margin(double m, double b, double k, double kp);

// ==========================================================================================
// Part B -- wave variables
// ==========================================================================================

/// Wave variables of a power pair (F, xdot) for wave impedance b (Niemeyer & Slotine):
///     u = (b xdot + F)/sqrt(2b)   (forward wave),   v = (b xdot - F)/sqrt(2b)   (returning wave).
/// Power: F xdot = (u^2 - v^2)/2.  Returns (u, v).
std::pair<double, double> wave_encode(double F, double xdot, double b);

/// Leader side (input: leader velocity xdot_m and the returning wave v_m(t) = v_s(t - T)).
/// Returns (F_m, u_m) with F_m = b xdot_m - sqrt(2b) v_m and u_m = sqrt(2b) xdot_m - v_m,
/// so that wave_encode(F_m, xdot_m, b) == (u_m, v_m).
std::pair<double, double> wave_leader(double xdot_m, double v_m, double b);

/// Follower side (input: arriving wave u_s(t) = u_m(t - T) and the follower controller force F_c).
/// Returns (xdot_d, v_s) with xdot_d = (sqrt(2b) u_s - F_c)/b and v_s = sqrt(2b) xdot_d - u_s,
/// so that wave_encode(F_c, xdot_d, b) == (u_s, v_s).
std::pair<double, double> wave_follower(double u_s, double F_c, double b);

/// Impedance Z(j w) = F_m / xdot_m felt at the leader through a wave channel of one-way delay T:
/// termination "rigid" (xdot_d = 0):  Z = b coth(j w T);   "free" (F_c = 0):  Z = b tanh(j w T).
/// Low frequency: rigid -> spring of stiffness b/T; free -> mass b T.
std::complex<double> wave_apparent_impedance(double b, double T, double omega, const std::string& termination);

// ==========================================================================================
// Part C -- energy bookkeeping and time-domain passivity
// ==========================================================================================

/// Follower position controller F_c = Kp (x_d - x_s) - Kd xdot_s (damping on the follower velocity only,
/// so that F_c does not depend on the set-point step of the current sample).
double follower_pd(double x_d, double x_s, double xdot_s, double Kp, double Kd);

/// Split the sampled power P[k] flowing INTO a port into cumulative input and output energies:
/// E_in[k] = h sum_{i<=k} max(P_i, 0),  E_out[k] = h sum_{i<=k} max(-P_i, 0).  Returns (E_in, E_out).
std::pair<Eigen::VectorXd, Eigen::VectorXd> port_energies(const Eigen::VectorXd& P, double h);

/// Passivity controller as a scaling ratio r in [0, 1]: if the energy a port is about to output in this sample
/// (out_trial > 0) exceeds the available budget, scale the port's output variable by
/// r = max(budget, 0) / out_trial so that exactly the budget is released; otherwise r = 1.
double passivity_ratio(double out_trial, double budget);

/// Result of tdpa_leader (the Python 4-tuple, same names and order).
struct TdpaLeaderResult {
    double F_m = 0.0, E_in_m = 0.0, E_out_m = 0.0, r = 0.0;
};
/// Result of tdpa_follower (the Python 4-tuple, same names and order).
struct TdpaFollowerResult {
    double dx_d = 0.0, E_in_s = 0.0, E_out_s = 0.0, r = 0.0;
};

/// Passivity observer + controller at the leader port (impedance causality: input xdot_m, output F_m).
/// Power into the channel P = F_m xdot_m.  If P >= 0, add P h to E_in_m.  If P < 0 the channel would output
/// -P h; allowed budget = E_in_s_recv - E_out_m (energy that entered at the follower, delayed, minus what the
/// leader port has already released).  Scale F_m by passivity_ratio and add the released energy to E_out_m.
/// Returns {F_m_new, E_in_m, E_out_m, r}.
TdpaLeaderResult tdpa_leader(double F_m, double xdot_m, double h, double E_in_m, double E_out_m,
                             double E_in_s_recv);

/// Passivity observer + controller at the follower port (admittance causality: input F_c, output the set-point
/// step dx_d).  Energy out of the channel in this sample = F_c dx_d.  If <= 0, add -F_c dx_d to E_in_s.
/// Otherwise budget = E_in_m_recv - E_out_s; scale dx_d by passivity_ratio and add the released energy to E_out_s.
/// Returns {dx_d_new, E_in_s, E_out_s, r}.
TdpaFollowerResult tdpa_follower(double F_c, double dx_d, double E_in_s, double E_out_s, double E_in_m_recv);

// ==========================================================================================
// Part D -- performance metrics for the report
// ==========================================================================================

/// The dict returned by performance_metrics (same keys).
struct Metrics {
    double F_m_mean = 0.0, f_e_mean = 0.0, force_ratio = 0.0, drift = 0.0;
    double x_err_rms = 0.0, x_std = 0.0, E_ch_min = 0.0;
};

/// Steady-state metrics after t_from (s); uses sim.t, F_m, f_e, x_d, x_m, x_s, E_ch only:
///   F_m_mean, f_e_mean      mean force felt by the operator and mean contact force (N)
///   force_ratio             F_m_mean / f_e_mean (force transparency; 1 = ideal; NaN if |f_e_mean| <= 1e-12)
///   drift                   mean (x_d - x_m) (m): set-point drift between leader and follower
///   x_err_rms               rms of (x_s - x_m) (m)
///   x_std                   standard deviation of x_m (m, population: np.std): the size of a sustained oscillation
///   E_ch_min                minimum of the channel energy over the whole run (J); < 0 => active channel.
Metrics performance_metrics(const SimResult& sim, double t_from = 2.5);

/// One-way latency budget: total = sum of the components (ms); margin = limit - total (ms).
/// The components are (name, ms) pairs in the order given (Python: a dict).  Returns (total, margin).
std::pair<double, double> latency_budget(const std::vector<std::pair<std::string, double>>& components_ms,
                                         double limit_ms);

}  // namespace lab

#endif  // SBEG46X_LAB3_HPP
