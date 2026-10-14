// SBEG 461 Hardware rotation 1 (Week 8) -- joint control on the course arm: the PID step response of the servo
// loop, gravity compensation, and logging (C++ track): declarations.
//
// The same functions as starter.py (module hwrun.py of your sbeg461_dynamics library), in namespace `lab`, with
// Eigen types instead of NumPy arrays.  Write your code in starter.cpp.  Do not change these declarations:
// test_lab.cpp calls them.
//
// What runs on the arm.  The course arm has position-controlled smart servos (project brief Sec. 5.1).  The station
// streams a joint-angle command every 20 ms (50 Hz).  Inside each servo a firmware loop at 1 kHz is a PD controller
// of exactly the Lab 5 kind (MR Eq. 11.23 with Ki = 0, derivative on the measured velocity), acting on one joint of
// the arm after a bus/firmware delay d:
//     M_eff q'' + b q' + g_j(q) = tau,    tau = sat( Kp (q_cmd(t - d) - q) - Kd q' )            (MR Eq. 11.21)
// M_eff = M_jj(q) + the apparent rotor inertia G^2 I_rotor of the geared servo (MR Sec. 8.9.2).  You cannot change
// Kp and Kd (they are in the firmware), but you can identify them, and you can wrap your own loop around the servo at
// 50 Hz: gravity feedforward through the position command (MR Eq. 11.38 realised with a position servo) and an
// integral outer loop (the I of PID, MR Eq. 11.23).  The PID code of Lab 5 (pd_gains, pd_gravity_error, JointPID,
// simulate_joint) is given, unchanged.  All device numbers are ILLUSTRATIVE VALUES; identify them on your own unit.
//
// Notation: Modern Robotics (MR).  Units: SI (s, rad, rad/s, N m, A) unless a name ends in _deg.
// Time series and logs are matrices with ONE ROW PER SAMPLE (N x n), as the NumPy arrays; t is a VectorXd (N).
// Python -> C++: a dict result -> a small struct with the same field names; joints=(1, 2) (0-based) ->
// std::vector<int>{1, 2}; None -> an empty std::optional or an empty (size 0) Eigen object; np.inf ->
// std::numeric_limits<double>::infinity(); the session record and the task file (JSON) -> sbeg::json.
#ifndef SBEG461_HW1_HPP
#define SBEG461_HW1_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <sbeg/labio.hpp>  // sbeg::json (task file, session record)

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ===========================================================================================================
// Constants (as in starter.py)
// ===========================================================================================================
constexpr double PI = 3.14159265358979323846;
constexpr double DEG = PI / 180.0;
constexpr double DT = 0.02;                      ///< station stream / logger period (s), 50 Hz
constexpr double H = 0.001;                      ///< servo firmware period (s), 1 kHz
constexpr double ENC_RES = 2.0 * PI / 4096.0;    ///< 12-bit encoder (rad)

/// The modes of the hold test (Part D).
inline const std::vector<std::string> MODES = {"none", "ff", "int", "ff+int"};

/// Software joint limits >= 10 deg inside the hardware limits (shared checklist, rule A1).
constexpr double SOFT_LIMIT_MARGIN_DEG = 10.0;
/// 0.125 kg = 50 % of the rated 0.25 kg (brief Sec. 5.2, rule A4).
constexpr double PAYLOAD_MAX_KG = 0.5 * 0.25;
/// Approved outer-loop integral gain (1/s) and correction clamp (deg) of the station loop.
constexpr double OUTER_KI_MAX = 4.0;
constexpr double OUTER_CLAMP_MAX_DEG = 5.0;

/// Labels = the rule labels of project brief Sec. 5.2 (A1-A7 course arm, B1-B6 session rules).
inline const std::vector<std::string> PREFLIGHT_RULES = {
    "A5: every member who touches the arm has a signed induction",
    "A7: a TA or the instructor is named as supervisor",
    "A7: the e-stop was tested at the start of the session",
    "B4: a named member holds the e-stop",
    "A2/A6: the keep-out zone is taped and clear",
    "B5: the workspace is ready (mat, payload fixed, cables tidy)",
    "A1: every commanded angle is >= 10 deg inside the hardware limits",
    "A3/B2: the dry run predicts peak speeds <= the required fraction of rated speed",
    "B2: the dry run predicts peak torques <= 60 % of rated and is written on the log",
    "A (current): the servo current limit is <= 60 % of rated",
    "A4: the payload is <= 0.125 kg",
    "B2 (station loop): outer-loop integral gain <= 4 1/s and correction clamp <= 5 deg",
};

// ===========================================================================================================
// Provided (complete, in course_arm.cpp and helpers.cpp): model, task, logs
// ===========================================================================================================

/// The course arm (the lab's own copy of the shared course-arm model, course_arm.cpp = course_arm.py).
/// Python dict keys -> fields.  Slist 6 x 6 (axes are the COLUMNS), limits 6 x 2 (rad, [min, max] per joint).
struct Model {
    std::string name;
    int n = 6;
    Eigen::MatrixXd Slist;
    Eigen::Matrix4d M;
    Eigen::MatrixXd limits;
    Eigen::VectorXd qdot_max;   ///< rated speed (rad/s)
    Eigen::VectorXd tau_max;    ///< rated torque (N m)
    Eigen::VectorXd q_home;
    std::vector<Eigen::Matrix4d> Mlist;
    std::vector<sbeg::Matrix6d> Glist;
    Eigen::Vector3d g;
    double payload_max = 0.0;
    double derated_speed_fraction = 0.0;
    double derated_current_fraction = 0.0;
};

/// THE course arm.  `payload` (kg) adds a point mass at the tool point (same rule as the shared model).
Model course_arm_model(double payload = 0.0);

/// modern_robotics RNEA (MR Eqs. 8.50-8.54) for a model; gravity=false sets g = 0.
Eigen::VectorXd inverse_dynamics(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& ddq, bool gravity = true);

/// g(theta): RNEA with zero velocity and acceleration (MR Eq. 8.1), one configuration q (n).
Eigen::VectorXd gravity_torques(const Model& model, const Eigen::VectorXd& q);

/// g(theta) for every row of Q (N x n) -> N x n (Python: gravity_torques(model, q) with q of shape (N, n)).
Eigen::MatrixXd gravity_torques_rows(const Model& model, const Eigen::MatrixXd& Q);

/// diag M(theta) of the links (MR Eq. 8.1; mr::mass_matrix), n entries.
Eigen::VectorXd mass_diagonal(const Model& model, const Eigen::VectorXd& q);

/// Task data (data/<name>): poses, limits, nominal servo, step test, static poses, hold, move and stop tests.
sbeg::json load_task(const std::string& name = "rotation1_task.json");

/// Path of a file in the lab's data folder (Python: DATA / name).
std::string data_path(const std::string& name);

/// Write a CSV in the project log format (project starter `hardware.py`):
/// t, cmd_1..cmd_n[, meas_1..meas_n][, cur_1..cur_n]  (s, rad, A).  An empty meas / cur (0 x 0) is left out.
void write_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd,
               const Eigen::MatrixXd& meas = Eigen::MatrixXd(), const Eigen::MatrixXd& cur = Eigen::MatrixXd());

/// A log read by `load_log`: t (N), cmd, meas, cur (N x n; 0 x 0 when absent) and any extra columns without
/// "_" in the name (e.g. "pose", "dir", "mode") in `extra`.
struct Log {
    Eigen::VectorXd t;
    Eigen::MatrixXd cmd, meas, cur;
    std::map<std::string, Eigen::VectorXd> extra;
};

/// Read a log written by `write_log` or by the station logger.
Log load_log(const std::string& path);

/// The rows of a log with t0 <= t <= t1 (and, for the hold log, extra["mode"] == mode).
Log log_window(const Log& log, double t0, double t1, std::optional<double> mode = std::nullopt);

/// Step-test command: hold `pose` (n), then add `amplitude_deg` to joint j (1-based) from time t_steps[k] on, for
/// each joint of `joints`.  Returns (t (N), cmd (N x n)).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> step_command(const Eigen::VectorXd& pose, const std::vector<int>& joints,
                                                         const std::vector<double>& t_steps, double amplitude_deg,
                                                         double duration, double dt = DT);

/// The station move routine: rest at q0 for `dwell`, a rest-to-rest quintic to q1 in T s (MR Sec. 9.2.2.1, taught
/// in Week 10), rest for 2 dwell, and (if back) the same move back and a final dwell.  Returns (t, cmd).
std::pair<Eigen::VectorXd, Eigen::MatrixXd> smooth_move(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, double T,
                                                        double dt = DT, double dwell = 0.5, bool back = true);

/// Result of `check_command`.
struct CommandCheck {
    bool ok = false;
    bool within_limits = false;
    double peak_speed_ratio = 0.0;
    bool speed_ok = false;
    bool starts_at_rest = false;
    bool ends_at_rest = false;
};

/// Safety check of a command table (project brief Sec. 5.2, rules A1 and A3): joint limits with a margin, peak
/// finite-difference speed against speed_fraction * rated speed, rest at both ends.  (A step command fails the
/// speed test by construction: for steps use the dry-run speed instead.)
CommandCheck check_command(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, const Model& model,
                           double speed_fraction, double margin_deg = SOFT_LIMIT_MARGIN_DEG, double tol = 0.01);

/// Result of `tracking_metrics` (per joint, rad; lag in s).
struct TrackingMetrics {
    Eigen::VectorXd rms, max_abs;
    double lag = 0.0;
    Eigen::VectorXd rms_after_lag;
};

/// RMS and max |meas - cmd| per joint, the lag d (multiple of the sample time) that minimises the RMS of
/// meas(t) - cmd(t - d), and the RMS after removing it.
TrackingMetrics tracking_metrics(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Eigen::MatrixXd& meas,
                                 double max_lag = 0.2);

/// Averaged static poses (P poses): pose ids and the averaged cmd, meas, cur (P x n).
struct StaticPoses {
    std::vector<int> pose;
    Eigen::MatrixXd cmd, meas, cur;
};

/// Static-pose log (columns pose, dir, cmd_j, meas_j, cur_j; each pose approached from below, dir = +1, and from
/// above, dir = -1): average the two approaches of every pose, so that static friction cancels to first order.
StaticPoses average_approaches(const Log& log);

/// np.median(v): the middle value of v (mean of the two middle values for an even length).
double median(const Eigen::VectorXd& v);

/// Unit step response of q_dd + 2 zeta wn q_d + wn^2 q = wn^2 u (Astrom-Murray Eq. 6.22, k = 1) at times
/// tau >= 0 after the start of motion (0 for tau < 0).
Eigen::VectorXd step_response(const Eigen::VectorXd& tau, double wn, double zeta);

/// RMS(predicted - measured) / RMS(measured) (project `hardware.normalised_rms`; HWk-2 passes at <= 0.3).
double normalised_rms(const Eigen::VectorXd& predicted, const Eigen::VectorXd& measured);

/// Result of `least_squares`: the solution x and cost = 0.5 * |residual(x)|^2.
struct LeastSquaresResult {
    Eigen::VectorXd x;
    double cost = 0.0;
};

/// Bounded nonlinear least squares, the C++ stand-in for scipy.optimize.least_squares(fun, x0, bounds=(lb, ub),
/// x_scale=x_scale): minimises 0.5 |fun(x)|^2 with lb <= x <= ub (Levenberg-Marquardt, forward-difference
/// Jacobian, steps projected onto the bounds; at most max_iter iterations).
LeastSquaresResult least_squares(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& fun,
                                 const Eigen::VectorXd& x0, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 const Eigen::VectorXd& x_scale, int max_iter = 200);

// ===========================================================================================================
// Provided: the Lab 5 PID code, unchanged (MR Sec. 11.4.1)
// ===========================================================================================================

/// A single-joint plant  tau = M q'' + b q' + g(q)  (MR Eq. 11.21), g(q) = g_cos cos q + g_sin sin q.
struct Plant {
    std::string name;
    double M = 0.0;                   ///< inertia about the joint axis [kg m^2]
    double b = 0.0;                   ///< viscous friction [N m s/rad]
    double g_cos = 0.0;               ///< gravity coefficients of g(q) [N m]
    double g_sin = 0.0;
    std::optional<double> tau_max;    ///< actuator torque limit [N m]; empty = no limit (Python None)
    Eigen::VectorXd locked_q;         ///< course-arm plants: the locked joint angles
    int joint_index = 0;              ///< course-arm plants: joint number 1..6
};

/// g(q) = g_cos cos q + g_sin sin q (N m); the torque needed to hold the joint still at q.
double plant_gravity(const Plant& plant, double q);

/// State derivative of x = (q, q') under torque tau:  q'' = (tau - b q' - g(q)) / M.
Eigen::Vector2d plant_rhs(const Plant& plant, const Eigen::Vector2d& x, double tau, bool gravity = true);

/// The right-hand side f(x) of rk4_step (a lambda works).
using StateFunction = std::function<Eigen::Vector2d(const Eigen::Vector2d&)>;

/// One classical Runge-Kutta step of x' = f(x) (the input is held constant over the step).
Eigen::Vector2d rk4_step(const StateFunction& f, const Eigen::Vector2d& x, double h);

/// PD gains for M e'' + (b + Kd) e' + Kp e = 0 (MR Eq. 11.25) with target zeta, wn (MR Eq. 11.26):
/// Kp = M wn^2, Kd = 2 zeta wn M - b.  Returns (Kp, Kd).
std::pair<double, double> pd_gains(double M, double b, double zeta, double wn);

/// Steady-state error of PD setpoint control under gravity (MR Eq. 11.27): at rest Kp e = g(q), q = qd - e,
/// solved by Newton's method from e = g(qd)/Kp.  Returns e (rad).
double pd_gravity_error(const Plant& plant, double kp, double qd);

/// The P, I and D contributions of the last update.
struct PidTerms {
    double P = 0.0, I = 0.0, D = 0.0;
};

/// Discrete PID joint controller of Lab 5 (MR Eq. 11.23, pseudocode MR Fig. 11.15) with an optional output limit
/// and integrator anti-windup (MR p. 427; Astrom-Murray Sec. 10.4):
///     lab::JointPID ctrl(kp, ki, kd, dt, tau_max = std::nullopt, anti_windup = true);
///     ctrl.reset(q0);
///     double u = ctrl.update(qd, qd_dot, q, qdot = std::nullopt, tau_ff = 0.0);
/// e = qd - q, e_dot = qd_dot - qdot (qdot differenced from q if empty), eint += e dt,
/// u = Kp e + Kd e_dot + Ki eint + tau_ff, clipped to +-tau_max; while clipped with e of the sign of u the integral
/// is frozen.  The output is a torque in Lab 5; in the outer loop of Part D it is a position correction (rad), and
/// tau_max is then the correction clamp.
struct JointPID {
    JointPID(double kp, double ki, double kd, double dt, std::optional<double> tau_max = std::nullopt,
             bool anti_windup = true);
    void reset(std::optional<double> q0 = std::nullopt);
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

/// Lab 5 closed-loop simulation of tau = M q'' + b q' + g(q) (MR Eq. 11.21) under a sampled controller on the grid
/// t (dt = t[1] - t[0] = ctrl.dt, else std::invalid_argument); torque clipped to plant.tau_max; RK4 substeps.
JointSim simulate_joint(const Plant& plant, JointPID& ctrl, const Eigen::VectorXd& t, const Eigen::VectorXd& qd,
                        const Eigen::VectorXd& qd_dot, double q0, double dq0 = 0.0, int substeps = 4,
                        const std::string& velocity = "measured", std::optional<double> enc_res = std::nullopt,
                        const Eigen::VectorXd& tau_ff = Eigen::VectorXd(), bool gravity = true);

// ===========================================================================================================
// Provided: one course-arm joint as a Lab 5 plant, the servo model of the whole arm, log replay
// ===========================================================================================================

/// Lab 5 plant of course-arm joint `joint` (0-based) with the other joints locked at q_lock: g(q) = g_cos cos q +
/// g_sin sin q is exact for a horizontal joint axis, with g_cos = g_j(q_j = 0), g_sin = g_j(q_j = pi/2) (RNEA).
Plant joint_plant(const Model& model, const Eigen::VectorXd& q_lock, int joint, double M_eff, double b = 0.0,
                  std::optional<double> tau_max = std::nullopt);

/// The servo of every joint (Python: the servo dict): kp, kd, M_eff, b (n each) and the delay (s).
struct Servo {
    Eigen::VectorXd kp, kd, M_eff, b;
    double delay = 0.0;
};

/// The NOMINAL servo model at pose q (before any identification): M_eff = M_jj(q) + armature_j
/// (task["nominal_servo"]) and (Kp, Kd) = pd_gains(M_eff, b, 0.9, 25) for every joint.
Servo nominal_servo(const Model& model, const Eigen::VectorXd& q, const sbeg::json& task);

/// One joint of `make_servo` (Python: a dict kp, kd, M_eff[, b][, delay]).
struct JointServo {
    double kp = 0.0, kd = 0.0, M_eff = 0.0;
    std::optional<double> b;
    std::optional<double> delay;
};

/// Copy of `base` with the joints in `fits` (0-based keys) replaced; the delay of the arm is the mean of the
/// fitted delays (unchanged if none is given).
Servo make_servo(const Servo& base, const std::map<int, JointServo>& fits);

/// The station interface (the same methods as the station's hardware object): read() -> (q (n), cur (n), or an
/// empty vector when the device gives no currents), send(q_cmd), wait() = advance one station period.
struct ServoIO {
    virtual ~ServoIO() = default;
    virtual std::pair<Eigen::VectorXd, Eigen::VectorXd> read() = 0;
    virtual void send(const Eigen::VectorXd& q_cmd) = 0;
    virtual void wait() = 0;
};

/// Result of `ServoArm::run`: meas, q, qd, tau (N x n) and the peaks |q'|, |tau| (n) over all firmware steps.
struct ServoRun {
    Eigen::MatrixXd meas, q, qd, tau;
    Eigen::VectorXd peak_qd, peak_tau;
};

/// Simulation of the course arm under its joint servos (the simulation fallback and the "sim" of the sim-vs-hardware
/// comparison).  The joints in `joints` (0-based) move; the others are held rigidly at their command.  Every moving
/// joint j is the Lab 5 plant M_eff_j q_j'' + b_j q_j' + g_j(q) = tau_j, tau_j = Kp_j (c_j(t - delay) - q_j) -
/// Kd_j q_j' (a JointPID with Ki = 0, derivative on the measured velocity, at the firmware period h; optional
/// output limit tau_limit (n)), with g(q) of the whole arm from the RNEA, evaluated once per station period and
/// held (the coupling through M is neglected: MR Sec. 11.4.2.1).  Semi-implicit Euler at h.
///   reset(q_cmd0): settle at rest under q_cmd0 (with the gravity sag);  read(): encoder angles and servo torques
///   (currents tau/kt if kt is given);  send(q_cmd): applied after `delay`;  wait(): one station period;
///   run(t, cmd): open-loop run of a command table.
class ServoArm : public ServoIO {
public:
    ServoArm(const Model& model, const Servo& servo, std::vector<int> joints = {1, 2}, double dt = DT, double h = H,
             double enc_res = ENC_RES, const Eigen::VectorXd& tau_limit = Eigen::VectorXd(),
             const Eigen::VectorXd& kt = Eigen::VectorXd());
    void reset(const Eigen::VectorXd& q_cmd0);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> read() override;
    void send(const Eigen::VectorXd& q_cmd) override;
    void wait() override;
    ServoRun run(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd);

    Eigen::VectorXd q, qd, tau, peak_qd, peak_tau;  ///< state (true angles), last servo torques, peaks
    double t = 0.0;

private:
    Eigen::VectorXd active() const;
    const Model* model_;
    Servo servo_;
    std::vector<int> joints_;
    double h_, enc_res_;
    int nsub_;
    Eigen::VectorXd tau_limit_, kt_;
    std::vector<std::pair<double, Eigen::VectorXd>> queue_;
    std::vector<JointPID> pids_;
};

/// Replays a logged run through the station interface: read() returns the logged meas and cur of the current row,
/// send() records the command in `sent`, wait() moves to the next row.
class ReplayIO : public ServoIO {
public:
    explicit ReplayIO(const Log& log);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> read() override;
    void send(const Eigen::VectorXd& q_cmd) override;
    void wait() override;
    std::vector<Eigen::VectorXd> sent;

private:
    Eigen::MatrixXd meas_, cur_;
    Eigen::Index k_ = 0;
};

// ===========================================================================================================
// Part A -- pre-flight: units, the dry run and the session record
// ===========================================================================================================

/// Servo reading (degrees, servo zero) -> model angle (rad, PoE zero): q_model = q_servo - offset, in radians.
Eigen::VectorXd servo_to_model(const Eigen::VectorXd& q_servo_deg, const Eigen::VectorXd& offset_deg);

/// Inverse of `servo_to_model`: model angle (rad) -> servo command (degrees).
Eigen::VectorXd model_to_servo(const Eigen::VectorXd& q_model, const Eigen::VectorXd& offset_deg);

/// Result of `dry_run`.
struct DryRun {
    Eigen::MatrixXd meas;          ///< N x n predicted encoder angles
    Eigen::VectorXd peak_qd;       ///< peaks of |q'| over all firmware steps
    Eigen::VectorXd peak_tau;      ///< peaks of |servo torque| over all firmware steps
    Eigen::VectorXd speed_ratio;   ///< peak_qd / (speed_fraction * qdot_max)
    Eigen::VectorXd torque_ratio;  ///< peak_tau / (torque_fraction * tau_max)
    bool ok = false;               ///< every ratio <= 1
};

/// Dry-run prediction written on the session log before the arm is powered (brief Sec. 5.2, rule B2): run the
/// command table cmd (N x n) on the grid t through ServoArm(model, servo, joints, DT, H, 0.0) (no torque limit: we
/// want to see whether the command would need more) and return meas, peak_qd, peak_tau and the ratios against the
/// derated limits.  A step of amplitude A asks for a torque spike of about Kp A on top of the gravity torque.
DryRun dry_run(const Model& model, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Servo& servo,
               const std::vector<int>& joints = {1, 2}, double speed_fraction = 0.25, double torque_fraction = 0.6);

/// Checks a session record (data/session_record_template.json; read it with sbeg::load_json) against the hardware
/// safety checklist of project brief Sec. 5.2 and returns the list of FAILED rules (strings of PREFLIGHT_RULES, in
/// that order); an empty list means the session may start.
/// Record keys: members_at_arm (list of names), inducted (list of names), supervisor (str), estop_tested (bool),
/// estop_holder (str, one of members_at_arm), keepout_clear (bool), workspace_ready (bool), first_run (bool),
/// limits_ok (bool: check_command(...).within_limits of every command file), margin_deg (number),
/// dry_run_speed_fraction (number), dry_run_speed_ratio, dry_run_torque_ratio (numbers, the largest over all command
/// files), current_limit_fraction (number), payload_kg (number), outer_ki (1/s) and outer_clamp_deg (deg) of the
/// station loop (0 if it is not used).  The required speed fraction is 0.25 for a first run and 0.5 otherwise; a
/// dry run at a HIGHER fraction than required does not count.  A missing key (or null) counts as a failure.
std::vector<std::string> preflight(const sbeg::json& record);

// ===========================================================================================================
// Part B -- the step response of the servo's PD loop (identification)
// ===========================================================================================================

/// Hand method (Astrom-Murray Table 6.1; MR p. 415): from the fractional overshoot Mp = exp(-pi zeta/sqrt(1-zeta^2))
/// and the peak time t_peak = pi/(wn sqrt(1-zeta^2)) (from the START OF MOTION) return (zeta, wn).
std::pair<double, double> zeta_wn_from_step(double overshoot, double t_peak);

/// Result of `step_features`.
struct StepFeatures {
    double t_step = 0.0, A = 0.0, y0 = 0.0, y_ss = 0.0, t_move = 0.0, overshoot = 0.0, t_peak_abs = 0.0;
};

/// Features of ONE step in one joint signal (t, cmd, meas: N entries each).
///   * t_step: first sample time where cmd differs from cmd[0];  A: cmd[-1] - cmd[0];
///   * y0, y_ss: mean of meas before t_step, and mean over the last 0.3 s;
///   * t_move: first time (linear interpolation between samples) at which (meas - y0)/(y_ss - y0) reaches
///     `threshold` (a rough delay estimate);
///   * overshoot: (max(meas) - y_ss)/(y_ss - y0) for a positive step;  t_peak_abs: time of that extreme sample.
StepFeatures step_features(const Eigen::VectorXd& t, const Eigen::VectorXd& cmd, const Eigen::VectorXd& meas,
                           double threshold = 0.1);

/// Result of `identify_servo`.
struct ServoFit {
    double wn = 0.0, zeta = 0.0, delay = 0.0, y0 = 0.0, A_fit = 0.0, rms = 0.0;
};

/// Least-squares fit of (wn, zeta, delay) of the servo loop to ONE step on [t_step - 0.1, t_step + window]:
/// meas(t) ~ y0 + A_fit * step_response(t - t_step - delay); y0 and A_fit solved linearly for every trial.  Start from
/// the hand method and refine with the given `least_squares` (bounds wn in [2, 200], zeta in [0.05, 3], delay in
/// [0, 0.2], x_scale (10, 0.1, 0.01)).  Returns ServoFit{wn, zeta, delay, y0, A_fit, rms}.
ServoFit identify_servo(const Eigen::VectorXd& t, const Eigen::VectorXd& cmd, const Eigen::VectorXd& meas,
                        double window = 1.0);

// ===========================================================================================================
// Part C -- static poses (torque constant, loop stiffness, gravity check) and the servo's PD gains
// ===========================================================================================================

/// Least-squares k (N m/A) with tau_model ~ k * current.  Same definition as project `hardware.torque_constant`.
double torque_constant(const Eigen::VectorXd& current, const Eigen::VectorXd& tau_model);

/// Least-squares stiffness K (N m/rad): sag = cmd - meas ~ tau_model / K, 1/K = sum(tau*sag)/sum(tau^2).
double loop_stiffness(const Eigen::VectorXd& sag, const Eigen::VectorXd& tau_model);

/// Result of `gravity_check`: k, nrms, K over `joints`; g (P x n); tau_cur (P x len(joints)).
struct GravityCheck {
    Eigen::VectorXd k, nrms, K;
    Eigen::MatrixXd g, tau_cur;
};

/// Gravity check on the averaged static poses: for each joint j in `joints` (0-based), g_j at the MEASURED pose;
/// k_j = torque_constant(cur_j[fit], g_j[fit]); tau_cur_j = k_j cur_j; nrms_j = normalised_rms(g_j[check],
/// tau_cur_j[check]); K_j = loop_stiffness(cmd_j - meas_j, g_j) on all poses.
GravityCheck gravity_check(const StaticPoses& st, const Model& model, const std::vector<int>& joints = {1, 2, 3},
                           const std::vector<int>& fit = {0, 2, 4}, const std::vector<int>& check = {1, 3, 5});

/// Result of `servo_from_fit` (armature empty if M_link is not given).
struct ServoPD {
    double kp = 0.0, kd = 0.0, M_eff = 0.0;
    std::optional<double> armature;
};

/// The servo's PD loop from its step response and its static stiffness (Lab 5 `pd_gains` backwards, MR Eqs.
/// 11.26-11.27): kp = K, M_eff = K / wn^2, kd = 2 zeta wn M_eff - b; armature = M_eff - M_link (MR Sec. 8.9.2).
ServoPD servo_from_fit(double wn, double zeta, double K, double b = 0.0, std::optional<double> M_link = std::nullopt);

/// Predicted response of ONE joint to its command cmd (on the station grid t) with the Lab 5 code: fine grid
/// tf = t[0] + h * (0 .. round((t[-1] - t[0]) / h)); delayed zero-order hold c(tf) = cmd[k], k the last sample with
/// t[k] <= tf - delay (cmd[0] before); q0 = cmd[0] - pd_gravity_error(plant, kp, cmd[0]); simulate_joint(plant,
/// JointPID(kp, 0, kd, h), tf, c, zeros, q0, 0, 1); resample q onto t (linear interpolation); round to enc_res if
/// given.  Returns the predicted measurement (N).
Eigen::VectorXd predict_step(const Plant& plant, double kp, double kd, const Eigen::VectorXd& t,
                             const Eigen::VectorXd& cmd, double delay, double h = H,
                             std::optional<double> enc_res = std::nullopt);

// ===========================================================================================================
// Part D -- gravity compensation, the station loop, sim vs hardware, safe stop
// ===========================================================================================================

/// Gravity compensation through the position command: q_cmd_j = q_des_j + g_j(q_des) / K_j for j in `joints`
/// (K: n entries); the other joints get q_des (MR Eq. 11.38 with Ki = 0 realised with a PD position servo).
Eigen::VectorXd gravity_comp_command(const Model& model, const Eigen::VectorXd& q_des, const Eigen::VectorXd& K,
                                     const std::vector<int>& joints = {1, 2});

/// Result of `station_hold` (n x 6 each; cur is NaN where the device gives none).
struct HoldRun {
    Eigen::VectorXd t;
    Eigen::MatrixXd cmd, meas, cur;
};

/// The station loop of the hold test (50 Hz, n samples) on a servo interface `io`.  At sample k (t_k = k dt):
/// (q, cur) = io.read(); the command starts as q_des; from t_on on, add for mode "ff" or "ff+int" the gravity offset
/// gravity_comp_command(model, q_des, K, joints) - q_des times min(1, (t_k - t_on) / t_ramp); for mode "int" (from
/// t_on) or "ff+int" (from t_on + t_ramp) add for each joint j the correction u_j = JointPID(0, ki, 0, dt, clamp)
/// .update(q_des_j, 0, q_j) (one controller per joint, created at the start); then io.send(cmd) and, except after the
/// last sample, io.wait().  Mode "none" sends q_des throughout; another mode throws std::invalid_argument.
HoldRun station_hold(ServoIO& io, const Eigen::VectorXd& q_des, int n, const std::string& mode, const Model& model,
                     const Eigen::VectorXd& K, double ki = 3.0, double clamp = 5.0 * DEG, double t_on = 0.5,
                     double t_ramp = 0.5, const std::vector<int>& joints = {1, 2}, double dt = DT);

/// Result of `compare_runs` (per joint, rad; lag in s).
struct RunComparison {
    Eigen::VectorXd rms_cmd_meas, max_cmd_meas, rms_sim_meas, max_sim_meas;
    double lag = 0.0;
};

/// Sim-vs-hardware metrics after time `skip` (project brief Sec. 5.4), per joint: rms_cmd_meas, max_cmd_meas,
/// rms_sim_meas, max_sim_meas (simulation of the SAME command against the measurement), and lag (s) from
/// `tracking_metrics`.  cmd, meas, sim are N x n on the same time grid t.
RunComparison compare_runs(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Eigen::MatrixXd& meas,
                           const Eigen::MatrixXd& sim, double skip = 0.0);

/// Safe-stop time (HWk-4): time from `t_trigger` until the joint speed, a central difference over `window` s, stays
/// below `threshold` (rad/s, all joints) for the rest of the log; infinity if it never settles; 0 if never above.
double stop_time(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, double t_trigger,
                 double threshold = 2.0 * DEG, double window = 0.1);

}  // namespace lab

#endif  // SBEG461_HW1_HPP
