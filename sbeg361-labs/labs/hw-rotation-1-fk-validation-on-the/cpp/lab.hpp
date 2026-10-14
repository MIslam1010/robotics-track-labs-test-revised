// SBEG 361 Hardware rotation 1 (Week 8) -- FK validation on the course arm (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py (module validate.py of your sbeg361_kinematics library), in
// namespace `lab`, with Eigen types instead of NumPy arrays.  Write your code in starter.cpp.  Do not change these
// declarations: test_lab.cpp calls them.
//
// Notation of Modern Robotics (MR): screw axes S = (omega, v), space-form PoE (MR Eq. 4.14); Slist is 6 x n with the
// axes as COLUMNS (as in Python).  Angles: SERVO degrees in logs and files, MODEL radians in FK;
// q_model = deg2rad(q_servo - offset).  Joint trajectories, logs and point lists are matrices with ONE ROW PER
// SAMPLE / POSE (N x n, m x 3), as the NumPy arrays.
//
// Python -> C++: a dict result -> a small struct with the same field names; None -> std::optional (empty);
// ValueError -> std::invalid_argument; joints=(1, 2) -> std::vector<int>{1, 2}; np.inf -> infinity().
#ifndef SBEG361_HW1_HPP
#define SBEG361_HW1_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <sbeg/labio.hpp>  // sbeg::json (the validation plan)

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arm_station.hpp"  // given: Arm, SafetyViolation, SafetyGuard, SimArm, course arm model

namespace lab {

// ===========================================================================================================
// Complete helpers (given in starter.cpp, as in starter.py)
// ===========================================================================================================

/// 3x3 skew-symmetric matrix [omega].
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// e^{[S] theta} for a screw axis S = (omega, v) (MR Eqs. 3.88-3.89); the Lab 5 helper screw_exp(S, theta).
/// Same result as Lab 3's matrix_exp6(se3mat) called as matrix_exp6(vec_to_se3(S) * theta).
Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta);

/// Space-form PoE T = e^{[S1]th1} ... e^{[Sn]thn} M (MR Eq. 4.14; Lab 5).  You may paste your own.
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Positions in {s} (m) of the tool point p_tool (given in {b}) for each row of Q (model radians).  Returns m x 3.
Eigen::MatrixXd tool_points(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                            const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

/// Model angles (radians) from servo angles (degrees): q_model = deg2rad(q_servo - offset), row by row.
Eigen::MatrixXd servo_to_model(const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg);

/// Servo angles (degrees) from model angles (radians), row by row.
Eigen::MatrixXd model_to_servo(const Eigen::MatrixXd& q_model_rad, const Eigen::VectorXd& offsets_deg);

/// Calibration file -> servo_zero_offset_deg.
Eigen::VectorXd load_offsets(const std::string& path);

/// Validation plan (JSON): poses, segment time, rate, tool point, limits, speeds (see data/validation_plan.json).
sbeg::json load_plan(const std::string& path);

/// np.gradient(F, t, axis=0): derivative of every column of F (N x n) with respect to t [N]; second-order central
/// differences inside (also for uneven spacing), first-order one-sided differences at the two ends.
Eigen::MatrixXd gradient(const Eigen::MatrixXd& F, const Eigen::VectorXd& t);

/// np.median of a vector (mean of the two middle values for an even length).
double median(const Eigen::VectorXd& x);

/// A sampled joint trajectory: t [N] (s), q [N x n] (degrees).
struct Trajectory {
    Eigen::VectorXd t;
    Eigen::MatrixXd q;
};

/// Joint-space motion through via points (rows of q_via_deg, degrees) with the cubic time scaling
/// s(tau) = 3 tau^2 - 2 tau^3 on each segment of duration T_seg (formula given here; time scaling is taught in SBEG 461 Week 10; MR Sec. 9.2.2.1,
/// Eq. 9.11), then a hold of t_hold at the last point.  Peak speed of a segment = 1.5 |dq| / T_seg.
Trajectory smooth_segments(const Eigen::MatrixXd& q_via_deg, double T_seg, double rate_hz, double t_hold = 0.0);

/// A logged run (Python: dict t, cmd, meas, stop, aborted).  Angles in servo degrees.
struct Log {
    Eigen::VectorXd t;                   ///< [N] s
    Eigen::MatrixXd cmd;                 ///< [N x n] last command actually sent
    Eigen::MatrixXd meas;                ///< [N x n] angles reported by the servos
    Eigen::VectorXd stop;                ///< [N] 0 before the stop command, 1 from the stop command on
    std::optional<std::string> aborted;  ///< empty (Python None) or the SafetyViolation message
};

/// Write a log in the format of the project's hwcompare.py: t, cmd_1..n, meas_1..n, stop (degrees).
void write_log(const std::string& path, const Log& log);

/// Read a log written by write_log (or by hwcompare.py / run_session.py).
Log read_log(const std::string& path);

/// Tip-measurement sheet: pose, approach (+1/-1), servo_1..6 (deg), x_mm, y_mm, z_mm.
struct Tips {
    std::vector<std::string> pose;
    Eigen::VectorXi approach;  ///< [m] +1 or -1
    Eigen::MatrixXd Q;         ///< [m x 6] servo degrees
    Eigen::MatrixXd P;         ///< [m x 3] measured tool point (mm)
};
Tips read_tips(const std::string& path);

/// Index of the first sample with stop = 1 (the number of samples before the stop); N if there is no stop.
Eigen::Index before_stop(const Log& log);

// ===========================================================================================================
// Step 1 -- before the arm moves (safety gate)
// ===========================================================================================================

/// Result of check_plan (Python dict with the same keys).
struct PlanCheck {
    Eigen::VectorXd margin_deg;  ///< [n] smallest distance of each joint (model degrees) from its software limits
    Eigen::VectorXd peak_speed;  ///< [n] peak |dq/dt| (deg/s), by gradient (= np.gradient)
    double speed_limit = 0.0;    ///< speed_fraction * rated_deg_s
    bool ok_limits = false;      ///< margin >= margin_deg on every joint
    bool ok_speed = false;       ///< peak <= speed_limit on every joint
    bool ok = false;             ///< both
};

/// Planning-level safety check of a joint trajectory (also used on MEASURED angles for HW-4 and HW-5).
/// t [N] s; q_servo_deg [N x n]; soft_limits_deg [n x 2] in MODEL degrees (column 0 lower, column 1 upper).
PlanCheck check_plan(const Eigen::VectorXd& t, const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg,
                     const Eigen::MatrixXd& soft_limits_deg, double rated_deg_s, double speed_fraction = 0.25,
                     double margin_deg = 10.0);

// ===========================================================================================================
// Step 2 -- logged run (safety gate)
// ===========================================================================================================

/// Send the planned commands and log commanded and measured angles (servo degrees).
/// For each sample k: wait until t[k]; if stop_at is given and t[k] >= stop_at (first time), call arm.stop();
/// while not stopped, send q_cmd row k (a SafetyViolation also stops the run: call arm.stop() and keep logging);
/// then read the measured angles.  After the stop the cmd row holds the LAST command actually sent.
/// Returns t, cmd [N x n], meas [N x n], stop [N] (0/1), aborted (empty or the violation message).
Log run_logged(Arm& arm, const Eigen::VectorXd& t, const Eigen::MatrixXd& q_cmd_servo_deg,
               std::optional<double> stop_at = std::nullopt);

// ===========================================================================================================
// Step 3 -- joint space, task space, stop (HW-2, HW-3, HW-6)
// ===========================================================================================================

struct JointTracking {
    Eigen::VectorXd rms, max, mean;  ///< [n] each, degrees
    Eigen::Index n_samples = 0;
};
/// HW-2: statistics of e = meas - cmd (degrees) per joint, samples BEFORE the stop command.
JointTracking joint_tracking(const Log& log);

struct TaskTracking {
    Eigen::VectorXd err_mm;  ///< [N'] one value per sample before the stop
    double rms_mm = 0.0, max_mm = 0.0;
};
/// HW-3: tool-point distance |p(meas) - p(cmd)| (mm) via PoE FK, samples before the stop.
TaskTracking task_tracking(const Log& log, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                           const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

struct StopCheck {
    double t_stop = 0.0, t_halt = 0.0, motion_deg = 0.0;
};
/// HW-6 on a noisy, quantised log.  The rest pose is the per-joint MEDIAN of the measured angles over the last
/// tail_s seconds of the log; t_halt = time from the stop command until every joint stays within band_deg of
/// that rest pose; motion_deg = largest |meas - meas(at the stop command)| after the stop.
/// t_halt = infinity if a joint is still outside the band at the end.
/// Throws std::invalid_argument("no stop command in the log") if stop is never 1.
StopCheck stop_check(const Log& log, double band_deg = 0.3, double tail_s = 0.5);

// ===========================================================================================================
// Step 4 -- is the arm where the model says? (HW-1, accuracy, repeatability, backlash)
// ===========================================================================================================

/// HW-1: distance (mm) between the FK prediction of the tool point and the external measurement, per pose.
/// Inputs are SERVO degrees [m x 6] plus the arm's offsets (q_model = q_servo - offset), not the model joint angles
/// of Lab 5's tip_errors_mm(M, Slist, Q, ...) (hence the hw_ prefix; argument order differs too).
Eigen::VectorXd hw_tip_errors_mm(const Eigen::MatrixXd& Q_servo_deg, const Eigen::MatrixXd& P_meas_mm,
                                 const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                                 const Eigen::VectorXd& offsets_deg,
                                 const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero());

struct AccuracyRepeatability {
    Eigen::Vector3d barycentre;
    double AP_mm = 0.0;
    Eigen::VectorXd l_mm;  ///< [m]
    double RP_mm = 0.0;
};
/// Pose accuracy and repeatability of repeated visits (rows of P_mm, m x 3) to ONE commanded pose (industrial
/// convention).  barycentre = mean of the attained points; AP = |barycentre - p_ref|; l_i = |P_i - barycentre|;
/// RP = mean(l) + 3 std(l) with the sample standard deviation (ddof = 1, i.e. divide by m - 1).
AccuracyRepeatability accuracy_repeatability(const Eigen::MatrixXd& P_mm, const Eigen::Vector3d& p_ref_mm);

/// Distance (mm) between the barycentres of the visits approached from the + side and from the - side (rows = points).
double backlash_split(const Eigen::MatrixXd& P_plus_mm, const Eigen::MatrixXd& P_minus_mm);

// ===========================================================================================================
// Step 5 -- joint-offset calibration
// ===========================================================================================================

struct OffsetFit {
    Eigen::VectorXd delta_deg;  ///< [len(joints)]
    double rms_before_mm = 0.0, rms_after_mm = 0.0;
    Eigen::VectorXd new_offsets_deg;  ///< [n]
};
/// Zero-offset errors delta (degrees) of several joints (0-based indices) that best explain the measured
/// tool points: minimise sum_k |p(q_k + delta) - p_meas_k|^2 by Gauss-Newton with a central-difference
/// derivative.  The true link angle is q_servo - offset + delta, so the corrected calibration offset is
/// offset - delta.  rms_*_mm = sqrt(sum of squared residuals / number of poses).
OffsetFit fit_offsets(const Eigen::MatrixXd& Q_servo_deg, const Eigen::MatrixXd& P_meas_mm, const Eigen::Matrix4d& M,
                      const Eigen::MatrixXd& Slist, const Eigen::VectorXd& offsets_deg,
                      const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero(),
                      const std::vector<int>& joints = {1, 2}, int iters = 10, double h_deg = 1e-3);

// ===========================================================================================================
// Step 6 -- error budget
// ===========================================================================================================

struct ErrorBudget {
    Eigen::VectorXd lever_m;  ///< [n] m per rad
    Eigen::VectorXd dp_mm;    ///< [n]
    double worst_mm = 0.0;    ///< sum of dp_mm
    double rss_mm = 0.0;      ///< sqrt of the sum of squares
};
/// Tool-point error caused by small joint errors dtheta_deg [n], joint by joint (revolute joints only).
/// The axis of joint j at configuration theta is the home axis (omega_j, q_j) moved by
/// T_{j-1} = e^{[S1]th1} ... e^{[S_{j-1}]th_{j-1}}; q_j = omega_j x v_j (point of the axis closest to the
/// origin, |omega_j| = 1).  A rotation d theta_j moves the tool point p by |omega_j' x (p - q_j')| d theta_j.
ErrorBudget error_budget(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta,
                         const Eigen::Vector3d& p_tool, const Eigen::VectorXd& dtheta_deg);

/// The same with one error for every joint (default: 12-bit quantisation 360/4096 deg).  Given: calls the version above.
ErrorBudget error_budget(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta,
                         const Eigen::Vector3d& p_tool = Eigen::Vector3d::Zero(), double dtheta_deg = 360.0 / 4096.0);

}  // namespace lab

#endif  // SBEG361_HW1_HPP
