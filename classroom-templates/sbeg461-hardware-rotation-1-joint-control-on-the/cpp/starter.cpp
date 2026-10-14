// SBEG 461 Hardware rotation 1 (Week 8) -- joint control on the course arm: the PID step response of the servo
// loop, gravity compensation, and logging (C++ track).  STARTER: complete every function marked TODO.  The
// declarations and full documentation are in lab.hpp.
//
// Provided (complete; do not edit): course_arm.cpp (the course arm model) and helpers.cpp (inverse_dynamics,
// gravity_torques, gravity_torques_rows, mass_diagonal, load_task, data_path, write_log, load_log, log_window,
// step_command, smooth_move, check_command, tracking_metrics, average_approaches, step_response, normalised_rms,
// median, least_squares (the stand-in for scipy.optimize.least_squares), the Lab 5 PID code (plant_gravity,
// plant_rhs, rk4_step, pd_gains, pd_gravity_error, JointPID, simulate_joint) and joint_plant, nominal_servo,
// make_servo, ServoArm, ReplayIO).  Use them freely, as the provided functions of starter.py.
//
// Notation: Modern Robotics (MR).  Units: SI (s, rad, rad/s, N m, A) unless a name ends in _deg.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// =============================================================================================
// Part A -- pre-flight: units, the dry run and the session record
// =============================================================================================
Eigen::VectorXd servo_to_model(const Eigen::VectorXd& q_servo_deg, const Eigen::VectorXd& offset_deg) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: (q_servo_deg - offset_deg) * DEG (DEG = pi/180 is in lab.hpp).
    throw std::logic_error("TODO: servo_to_model");
}

Eigen::VectorXd model_to_servo(const Eigen::VectorXd& q_model, const Eigen::VectorXd& offset_deg) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: q_model * (180 / PI) + offset_deg.
    throw std::logic_error("TODO: model_to_servo");
}

DryRun dry_run(const Model& model, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Servo& servo,
               const std::vector<int>& joints, double speed_fraction, double torque_fraction) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: ServoArm arm(model, servo, joints, DT, H, 0.0); ServoRun res = arm.run(t, cmd); then divide the
    // peaks by the derated limits element-wise (a.cwiseQuotient(b)).
    throw std::logic_error("TODO: dry_run");
}

std::vector<std::string> preflight(const sbeg::json& record) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: one test per rule, in the order of PREFLIGHT_RULES; record.contains(key), .is_null(), .is_boolean(),
    // .get<double>(); a missing key or null counts as a failure.
    throw std::logic_error("TODO: preflight");
}

// =============================================================================================
// Part B -- the step response of the servo's PD loop (identification)
// =============================================================================================
std::pair<double, double> zeta_wn_from_step(double overshoot, double t_peak) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: invert Mp = exp(-pi zeta / sqrt(1 - zeta^2)) and t_peak = pi / (wn sqrt(1 - zeta^2)).
    throw std::logic_error("TODO: zeta_wn_from_step");
}

StepFeatures step_features(const Eigen::VectorXd& t, const Eigen::VectorXd& cmd, const Eigen::VectorXd& meas,
                           double threshold) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: index k0 of the first changed command sample; z = (meas - y0) / (y_ss - y0); maxCoeff(&k).
    throw std::logic_error("TODO: step_features");
}

ServoFit identify_servo(const Eigen::VectorXd& t, const Eigen::VectorXd& cmd, const Eigen::VectorXd& meas,
                        double window) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: inner linear least squares for (y0, A_fit) (X.colPivHouseholderQr().solve(y)); outer: the given
    // least_squares over (wn, zeta, delay) with a lambda returning the residual, from three delay guesses.
    throw std::logic_error("TODO: identify_servo");
}

// =============================================================================================
// Part C -- static poses (torque constant, loop stiffness, gravity check) and the servo's PD gains
// =============================================================================================
double torque_constant(const Eigen::VectorXd& current, const Eigen::VectorXd& tau_model) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: least squares through the origin: k = (i . tau) / (i . i).
    throw std::logic_error("TODO: torque_constant");
}

double loop_stiffness(const Eigen::VectorXd& sag, const Eigen::VectorXd& tau_model) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: 1/K = sum(tau sag) / sum(tau^2).
    throw std::logic_error("TODO: loop_stiffness");
}

GravityCheck gravity_check(const StaticPoses& st, const Model& model, const std::vector<int>& joints,
                           const std::vector<int>& fit, const std::vector<int>& check) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: g at the MEASURED poses (gravity_torques_rows), then torque_constant, normalised_rms and
    // loop_stiffness per joint; copy the fit/check entries in a loop.
    throw std::logic_error("TODO: gravity_check");
}

ServoPD servo_from_fit(double wn, double zeta, double K, double b, std::optional<double> M_link) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: kp = K, M_eff = K / wn^2, kd = 2 zeta wn M_eff - b (MR Eq. 11.26 backwards).
    throw std::logic_error("TODO: servo_from_fit");
}

Eigen::VectorXd predict_step(const Plant& plant, double kp, double kd, const Eigen::VectorXd& t,
                             const Eigen::VectorXd& cmd, double delay, double h, std::optional<double> enc_res) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: fine grid at h, delayed zero-order hold of cmd (std::upper_bound), q0 from pd_gravity_error, then
    // simulate_joint(plant, ctrl, tf, c, zeros, q0, 0.0, 1) with JointPID ctrl(kp, 0.0, kd, h); interpolate back onto t.
    throw std::logic_error("TODO: predict_step");
}

// =============================================================================================
// Part D -- gravity compensation, the station loop, sim vs hardware, safe stop
// =============================================================================================
Eigen::VectorXd gravity_comp_command(const Model& model, const Eigen::VectorXd& q_des, const Eigen::VectorXd& K,
                                     const std::vector<int>& joints) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: q_des plus g_j(q_des) / K_j for the joints in `joints`.
    throw std::logic_error("TODO: gravity_comp_command");
}

HoldRun station_hold(ServoIO& io, const Eigen::VectorXd& q_des, int n, const std::string& mode, const Model& model,
                     const Eigen::VectorXd& K, double ki, double clamp, double t_on, double t_ramp,
                     const std::vector<int>& joints, double dt) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: loop k = 0..n-1: io.read(), build the command (ramped offset, integral corrections with one
    // JointPID(0.0, ki, 0.0, dt, clamp) per joint), io.send(c), io.wait() except after the last sample.
    throw std::logic_error("TODO: station_hold");
}

RunComparison compare_runs(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Eigen::MatrixXd& meas,
                           const Eigen::MatrixXd& sim, double skip) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: rows with t >= skip; RMS and max of meas - cmd and of meas - sim per column; lag from tracking_metrics.
    throw std::logic_error("TODO: compare_runs");
}

double stop_time(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, double t_trigger, double threshold,
                 double window) {
    // TODO: implement (see lab.hpp and the lab handout).  Hint: central-difference speed over the window, last sample above the threshold after t_trigger.
    throw std::logic_error("TODO: stop_time");
}

}  // namespace lab
