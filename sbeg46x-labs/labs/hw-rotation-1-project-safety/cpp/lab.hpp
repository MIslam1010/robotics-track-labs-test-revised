// SBEG 46X Hardware rotation 1 (Week 8) -- project safety induction and a supervised teleoperation trial on a
// Hapkit pair (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp (the 12 functions marked TODO A1 ... D2). The provided part of starter.py is given
// in helpers.cpp and the course models in course_models.hpp / course_models.cpp: do not edit them, and do not
// change these declarations (test_lab.cpp calls them).
//
// Conventions (Labs 2-3 and the project starter):
//   device       m_h x'' + b_h x' + k_g x = F_motor + F_ext   (x positive away from the operator, 1 kHz firmware loop)
//   follower PD  F_c = Kp (x_d - x_s) - Kd v_s,   x_d = alpha x_m (one tick old: the station link has >= 1 sample delay)
//   leader       F_m = beta F_c is the force of the teleoperator ON the leader; the leader motor applies -F_m
//   tissue       f_env = k (x_s - x_w) + b v_s for x_s > x_w, rendered in the follower firmware (motor outputs F_c - f_env)
//   channel      E_ch(t) = integral (F_m v_m - F_c v_d) dt >= 0 for a passive channel (Lab 3), v_d = dx_d/dt
// Logs use the project format: t, cmd_1 (= x_d), meas_1 (= x_s), cur_1 (follower motor output), force (= F_m),
// aux_x_m, aux_F_c, aux_f_env (SI units).  All parameter values are illustrative.
//
// Python -> C++: the task and the session record stay JSON (sbeg::json = nlohmann::json: task["pp"]["Kp"].get<double>());
// a dict result -> a small struct with the same field names; a 1-D array -> Eigen::VectorXd; a log -> lab::Log;
// None -> std::nullopt (std::optional) or nullptr; np.inf -> std::numeric_limits<double>::infinity();
// ValueError -> std::invalid_argument; NotImplementedError -> std::logic_error("TODO: <name>").
#ifndef SBEG46X_LAB08_HPP
#define SBEG46X_LAB08_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <sbeg/labio.hpp>

#include "course_models.hpp"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// =====================================================================  provided helpers (helpers.cpp, given)

/// Path of a file in the lab's data folder (data/<name>), independent of the working directory.
std::string data_path(const std::string& name);

/// Station settings and test definitions (data/hw1_task.json when path is empty).
sbeg::json load_task(const std::string& path = "");

/// A session record (JSON); the template data/session_record_template.json when path is empty.
sbeg::json load_record(const std::string& path = "");

/// A log in the project format. cur and force are empty (size 0) when the log has no such column;
/// aux = {name: column} for the columns aux_<name> (log.aux.at("x_m")).
struct Log {
    Eigen::VectorXd t, cmd, meas, cur, force;
    std::map<std::string, Eigen::VectorXd> aux;
};

/// Write a log in the project format: t, cmd_1, meas_1[, cur_1][, force][, aux_<name> ...] (empty vector = no column).
void write_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::VectorXd& cmd,
               const Eigen::VectorXd& meas, const Eigen::VectorXd& cur = Eigen::VectorXd(),
               const Eigen::VectorXd& force = Eigen::VectorXd(),
               const std::vector<std::pair<std::string, Eigen::VectorXd>>& aux = {});

/// Read a log (a name such as "trial_log.csv" is looked up in data/). Rows are sorted by time and duplicate time
/// stamps dropped (the station clock is not exactly 1 kHz).
Log load_log(const std::string& path);

/// Minimum-jerk move from x0 (at t0) to x1 (at t0 + T), constant outside.
Eigen::VectorXd minjerk(const Eigen::VectorXd& t, double t0, double T, double x0, double x1);

/// The operator's intended leader position for the trial: reach, hold in the tissue, withdraw.
Eigen::VectorXd intent(const sbeg::json& task, const Eigen::VectorXd& t);

/// Lab 2 tissue wall rendered in the follower firmware: resisting force (N, positive opposes +x).
double tissue_force(double x, double v, const sbeg::json& env);

/// One Hapkit-type device at handle level, with the firmware output stage (2 N saturation, latched +-40 mm
/// watchdog, optional fade to zero after a stop trigger) and an optional Coulomb friction f_c (smoothed over 1 mm/s).
/// Same structure as the project starter's devices.Hapkit (f_c = 0 there). Python Device(m_h=..., f_c=...):
/// `lab::Device d; d.f_c = 0.008;` or lab::device_from(params, overrides).
struct Device {
    double m_h = 0.05705, b_h = 0.15, k_g = 1.2111, T = 1e-3, x_res = 2e-5, force_max = 2.0, x_range = 0.040;
    double f_c = 0.0;
    double x = 0.0, v = 0.0, F_motor = 0.0;
    bool tripped = false;
    int sat_count = 0;

    double measure() const;                       ///< quantised position (x_res)
    double command(double F, double scale = 1.0); ///< watchdog, saturation count, clip to +-force_max, times scale
    void advance(double dt, double F_ext = 0.0, double m_extra = 0.0);  ///< semi-implicit Euler
};

/// A Device from a JSON object with keys m_h, b_h, k_g (and optionally f_c, x_res, force_max, x_range);
/// `overrides` (Python **kw) may set any field: device_from(params, {{"f_c", 0.008}}).
Device device_from(const sbeg::json& params, const sbeg::json& overrides = sbeg::json::object());

/// Firmware velocity: backward difference of the quantised position, first-order low-pass at fc Hz.
struct VelocityEstimator {
    double T, alpha, prev = 0.0, v = 0.0;
    bool has_prev = false;
    explicit VelocityEstimator(double T_ = 1e-3, double fc = 50.0);
    double operator()(double x);
};

/// Result of simulate_pp (Python: the dict of arrays; same names).
struct PPSim {
    Eigen::VectorXd t, x_int, x_m, x_s, xm_meas, xs_meas, x_d, F_m, F_c, f_env, F_s, E_ch;
    int sat_m = 0, sat_s = 0;
    bool tripped_m = false, tripped_s = false;
};

/// Position-position teleoperation of a Hapkit pair WITHOUT delay (the station link adds one 1 ms sample), the
/// operator's hand on the leader (spring-damper towards the intended position, its mass carried by the handle) and
/// the tissue wall rendered in the follower firmware.  t_stop: stop button pressed at this time (both motor outputs
/// fade linearly to zero over task["stop"]["fade_s"], starting one tick later). leader / follower / op (operator)
/// / x_int: nullptr = nominal (task). Returns t, x_m, x_s (true), xm_meas, xs_meas, x_d, F_m (force on the leader),
/// F_c, f_env, F_s (follower motor output), E_ch, and the saturation counts and watchdog flags.
PPSim simulate_pp(const sbeg::json& task, Device* leader = nullptr, Device* follower = nullptr,
                  const sbeg::json* op = nullptr, std::optional<double> T_end = std::nullopt,
                  std::optional<double> t_stop = std::nullopt, const Eigen::VectorXd* x_int = nullptr);

/// Result of simulate_follower: the replay grid t and the follower position x.
struct FollowerSim {
    Eigen::VectorXd t, x;
};

/// Replay the FOLLOWER device alone, driven by the logged set-point x_d(t) through the follower PD and minus the
/// logged rendered tissue force (the 'same command' simulation of project HW1-2). Uniform 1 kHz grid.
/// f_env_log: nullptr = no tissue force.
FollowerSim simulate_follower(const Eigen::VectorXd& t_log, const Eigen::VectorXd& x_d_log, double Kp, double Kd,
                              Device& dev, const Eigen::VectorXd* f_env_log = nullptr, double h = 1e-3,
                              int substeps = 5);

/// Result of simulate_free_decay: t, measured x, motor output F.
struct DecaySim {
    Eigen::VectorXd t, x, F;
};

/// Free-decay test: the firmware renders a spring F = -K x_meas (held for one period); the handle is held at x0
/// until t_release, then released with nobody's hand on it.
DecaySim simulate_free_decay(Device& dev, double K, double x0, double t_release, double T_end, double h = 1e-3,
                             int substeps = 5);

/// Root mean square of x.
double rms(const Eigen::VectorXd& x);

// ---- NumPy / SciPy stand-ins (given)

/// np.arange(start, stop, step) (same length and values as NumPy).
Eigen::VectorXd arange(double start, double stop, double step);

/// np.interp(x, xp, fp): linear interpolation, constant beyond the ends (xp increasing).
Eigen::VectorXd interp(const Eigen::VectorXd& x, const Eigen::VectorXd& xp, const Eigen::VectorXd& fp);

/// np.gradient(y, x): central differences inside (non-uniform spacing allowed), one-sided at the two ends.
Eigen::VectorXd gradient(const Eigen::VectorXd& y, const Eigen::VectorXd& x);

/// np.median(v): the middle value of v (mean of the two middle values for an even length).
double median(const Eigen::VectorXd& v);

/// scipy.optimize.curve_fit(model, xdata, ydata, p0, maxfev): unweighted nonlinear least squares by
/// Levenberg-Marquardt (forward-difference Jacobian, as MINPACK). model(x, p) returns the model at one x.
/// Returns the fitted parameters; throws std::runtime_error if it does not converge within maxfev evaluations.
Eigen::VectorXd curve_fit(const std::function<double(double, const Eigen::VectorXd&)>& model,
                          const Eigen::VectorXd& xdata, const Eigen::VectorXd& ydata, const Eigen::VectorXd& p0,
                          int maxfev = 20000);

// =====================================================================  Part A: pre-flight and dry run

/// Rule labels of the checklist.
inline const std::vector<std::string> RULES = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "B1",
                                               "B2", "B3", "B4", "B5", "B6", "B7", "I"};

/// d[key] as a double, or `fallback` when d is null, the key is missing or its value is null (an unfilled record
/// field). Given (Python: _num).
double num_or(const sbeg::json& d, const std::string& key, double fallback);

/// Python `bool(d.get(key, fallback))`: fallback when d is null or the key is missing; otherwise the truth value of
/// d[key] (null, false, 0, "" and empty lists/objects are false). Given.
bool flag_or(const sbeg::json& d, const std::string& key, bool fallback);

/// d[key] as a string, or "" when d is null, the key is missing or its value is null. Given.
std::string text_or(const sbeg::json& d, const std::string& key);

/// Python `d.get(key, {})`: d[key], or an empty JSON object when d is null or the key is missing. Given.
sbeg::json sub_or_empty(const sbeg::json& d, const std::string& key);

/// Check a session record against the hardware safety checklist (project brief section 5.2; Course resources:
/// Course robot models and hardware safety checklist). Returns a list of violations, each a string that starts
/// with the rule label ("A1: ...", "B4: ...", "I: ..." for the derated current); {} means the session may start.
///
/// Haptic station (every record): A5/B1 induction of every member who touches a device, supervisor present,
/// at least two team members (never alone); A7 stop/e-stop tested; B2 dry run done, predicted peak force
/// <= 2 N and peak |x| <= 40 mm written on the record; B3 force_max <= 2 N, watchdog enabled at <= 40 mm and
/// on loop overrun, only the operator's hand on the handle; B4 delay emulated in software, watchdog and force
/// limit never disabled, position-position ("pp") only with zero delay, a named stop holder who is not the
/// leader operator; B5 cables tidy, no loose items, devices fixed to the bench; B6 IMU low voltage (<= 5 V),
/// strapped over clothing; B7 CSV logging and a printed session log.
/// Arm station (record["arm"] not null): A1 software-limit margin >= 10 deg, A2 keep-out radius >= reach + 0.3 m
/// and taped, A3 speed fraction <= 0.25 on a first run (<= 0.50 otherwise), A4 payload <= 0.125 kg, A6 nobody
/// inside the keep-out zone while the arm can move, A7 e-stop tested, B5 soft mat under the workspace,
/// I servo current fraction <= 0.60.
std::vector<std::string> preflight(const sbeg::json& record);

/// Result of contact_equilibrium.
struct ContactEquilibrium {
    double x_s = 0.0, delta = 0.0, F_c = 0.0, F_m = 0.0, k_felt = 0.0;
};

/// Static P-P teleoperation at leader position x_m (no motion, no delay): follower PD force balances the
/// tissue spring and the follower's gravity stiffness,  Kp(alpha x_m - x_s) = k_e (x_s - x_w)^+ + k_g x_s.
/// Returns x_s, delta (penetration, >= 0), F_c, F_m = beta F_c, and k_felt = dF_m/dx_m (channel only,
/// in contact) = alpha beta Kp (k_e + k_g) / (Kp + k_e + k_g).
ContactEquilibrium contact_equilibrium(const sbeg::json& task, double x_m);

/// Result of dry_run (the numbers to write on the session log).
struct DryRun {
    double peak_F_m = 0.0, peak_F_s = 0.0, peak_x_m = 0.0, peak_x_s = 0.0;
    int sat = 0;
    double E_min = 0.0;
    bool tripped = false;
    double peak_force = 0.0, peak_x = 0.0;
    bool ok = false;
};

/// Dry run (rule B2): simulate the trial with the nominal devices and the nominal operator (simulate_pp)
/// and return the numbers to write on the session log: peak |F_m| (leader), peak |F_s| (follower motor),
/// peak_force (the larger), peak |x_m| and |x_s|, peak_x (the larger), saturation counts (sat = sat_m + sat_s),
/// min E_ch, tripped (either watchdog), and
/// ok = peak_force <= dry_run_force_fraction * 2 N and peak_x <= x_abs_max and no saturation, no trip.
DryRun dry_run(const sbeg::json& task);

// =====================================================================  Part B: bring-up and identification

/// Indices of the local maxima of sign*y. Runs of equal samples (quantised plateaus) count as one point; a run
/// is a maximum if both neighbouring runs are lower; its middle sample is returned. Maxima closer than 20 ms to
/// the previous one are dropped. Given (Python: _extrema); throws std::invalid_argument if there is none.
std::vector<Eigen::Index> extrema(const Eigen::VectorXd& tt, const Eigen::VectorXd& y, double sign);

/// Result of log_decrement.
struct LogDecrement {
    double zeta = 0.0, wd = 0.0, wn = 0.0;
    int n_peaks = 0;
};

/// Hand estimate from a free-decay record released at t0 (window t0 <= t <= t_end; t_end = nullopt: to the end).
/// Find the peaks and the troughs of x (a sample larger, or smaller, than its neighbours; a quantised plateau counts
/// as one sample, its middle; extrema at least 20 ms apart: use the given `extrema`). Pair each of the first n_max
/// peaks with the trough that follows it; the amplitude of cycle k is a_k = (peak_k - trough_k) / 2, which does not
/// depend on the offset of x. Damped period = mean spacing of those peaks; log decrement delta = mean of
/// ln(a_k / a_{k+1}); zeta = delta / sqrt(4 pi^2 + delta^2); wd = 2 pi / period; wn = wd / sqrt(1 - zeta^2).
LogDecrement log_decrement(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double t0,
                           std::optional<double> t_end = std::nullopt, int n_max = 6);

/// Result of fit_free_decay.
struct DecayFit {
    double m_h = 0.0, b_h = 0.0, wn = 0.0, zeta = 0.0, rms = 0.0;
};

/// Least-squares fit of x = A exp(-s tau) cos(wd tau + phi) + c (tau = t - t0, window t0 <= t <= t_end) to a
/// free-decay record taken while the firmware renders a spring K (use the given lab::curve_fit). Then
/// wn = sqrt(s^2 + wd^2), zeta = s / wn, m_h = (K + k_g) / wn^2 and b_h = 2 zeta wn m_h + K T / 2: the sampled, held
/// spring acts as a negative damper of about K T / 2 (Week 4; Colgate and Brown 1994), so the decay shows
/// b_h - K T / 2. Start the fit from log_decrement. rms = RMS of the fit residual.
DecayFit fit_free_decay(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double K, double k_g, double T, double t0,
                        std::optional<double> t_end = std::nullopt);

/// Result of watchdog_check (t_zero and delay are +inf if the output never stays at zero).
struct WatchdogCheck {
    double t_cross = 0.0, t_zero = 0.0, delay = 0.0;
    bool latched = false;
};

/// Watchdog test log (handle pushed out slowly by hand while the firmware renders a spring): t_cross = first
/// time |x| > x_range; t_zero = first time at or after t_cross from which the motor output stays exactly 0 to
/// the end; delay = t_zero - t_cross; latched = the output stays 0 even after the handle is back inside.
/// Throw std::invalid_argument if |x| never exceeds x_range.
WatchdogCheck watchdog_check(const Eigen::VectorXd& t, const Eigen::VectorXd& x, const Eigen::VectorXd& F,
                             double x_range);

// =====================================================================  Part C: the teleoperation trial

/// Linear interpolation of x_src(t_src) onto t_dst; NaN outside [t_src[0], t_src[-1]].
Eigen::VectorXd resample(const Eigen::VectorXd& t_src, const Eigen::VectorXd& x_src, const Eigen::VectorXd& t_dst);

/// Energy (J) that has entered the teleoperation channel from the logged port signals:
/// E_ch[k] = sum_{i <= k} (F_m[i] v_m[i] - F_c[i] v_d[i]) dt[i], with backward differences
/// v[i] = (x[i] - x[i-1]) / dt[i], dt[i] = t[i] - t[i-1] (v[0] = 0). A passive channel keeps E_ch >= 0.
Eigen::VectorXd channel_energy(const Eigen::VectorXd& t, const Eigen::VectorXd& F_m, const Eigen::VectorXd& x_m,
                               const Eigen::VectorXd& F_c, const Eigen::VectorXd& x_d);

/// Result of trial_metrics.
struct TrialMetrics {
    double rms_free = 0.0, x_m_hold = 0.0, x_s_hold = 0.0, F_m_hold = 0.0, delta_hold = 0.0;
    double F_m_pred = 0.0, transparency_error = 0.0, k_felt = 0.0;
    double peak_F_m = 0.0, peak_F_s = 0.0, peak_x = 0.0;
    int n_sat = 0;
    double E_min = 0.0;
    bool ok = false;
};

/// Metrics of a logged trial (project format). On a uniform 1 kHz grid (resample) compute:
/// rms_free  -- RMS of x_s - x_d in the "free" window (follower tracking in free motion, m)
/// x_m_hold, x_s_hold, delta_hold, F_m_hold -- means in the "hold" window; delta = x_s - x_w
/// F_m_pred  -- contact_equilibrium(task, x_m_hold).F_m;  transparency_error = F_m_hold / F_m_pred - 1
/// k_felt    -- least-squares slope of F_m against x_m over the samples in contact (x_s > x_w + contact_margin)
///              that move slowly (|v_m| < slow_speed, v_m by central differences: the given lab::gradient)
/// peak_F_m, peak_F_s (from cur), peak_x (max |x_m|, |x_s|), n_sat (samples with |force| or |cur| >= 2 N - 1e-9),
/// E_min     -- minimum of channel_energy on the ORIGINAL log samples
/// ok        -- peak forces <= 2 N with n_sat = 0, peak_x <= x_abs_max and E_min >= E_min limit.
TrialMetrics trial_metrics(const Log& log, const sbeg::json& task);

/// Result of sim_vs_hw (`max` = largest |x_sim - x_meas|).
struct SimVsHw {
    double rms = 0.0, max = 0.0;
    bool ok = false;
};

/// Simulation versus hardware for the follower (project HW1-2): replay the follower with simulate_follower
/// driven by the logged x_d (cmd) and the logged tissue force (aux f_env), with the device model `dev`; resample
/// the measured x_s onto the replay grid and return rms and max of (x_sim - x_meas) for t >= skip, and
/// ok = rms <= rms_sim_meas.
SimVsHw sim_vs_hw(const Log& log, const sbeg::json& task, Device dev, double skip = 0.5);

// =====================================================================  Part D: safe stop and the IMU at rest

/// Time from the stop trigger until |F| stays below threshold for the rest of the log (F has one row per sample
/// and one or more columns: all columns). 0 if already below; +inf if it never settles.
double fade_time(const Eigen::VectorXd& t, const Eigen::MatrixXd& F, double t_trigger, double threshold);

/// Result of imu_rest_stats.
struct ImuRestStats {
    double bias = 0.0, noise = 0.0, drift_deg = 0.0, tilt_deg = 0.0, residual_deg = 0.0;
};

/// IMU module at rest (B6 bring-up; project HW3-1/HW6-1): from the samples with t < t_rest
/// bias = mean gyro (rad/s), noise = standard deviation of the gyro (rad/s, ddof = 1),
/// drift_deg = |bias| * t_rest in degrees (the angle error after t_rest if the bias is not removed),
/// tilt_deg = atan2(-mean ay, mean ax) in degrees (accelerometer tilt of the mounting, ax = g cos, ay = -g sin);
/// and from the samples with t >= t_rest, residual_deg = the angle obtained by integrating (gyro - bias) with
/// the sample period (median time step; rectangle rule) over that interval, in degrees (it should be close to 0).
ImuRestStats imu_rest_stats(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, const Eigen::VectorXd& ax,
                            const Eigen::VectorXd& ay, double t_rest);

}  // namespace lab

#endif  // SBEG46X_LAB08_HPP
