// SBEG 46X Hardware rotation 1 (Week 8) -- project safety induction and a supervised teleoperation trial on a
// Hapkit pair (C++ STARTER).
//
// Fill in every function marked  // TODO A1 ... D2  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). The provided code (loaders, Device,
// VelocityEstimator, the three simulators, rms, num_or, flag_or, text_or, sub_or_empty, extrema, and the NumPy/SciPy
// stand-ins arange, interp, gradient, median, curve_fit) is in helpers.cpp; the course models (hapkit, handle_model,
// IMU, COURSE_ARM_SAFETY, HAPTIC_SAFETY) are in course_models.hpp. Do not edit those files.
// Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py hr1 --cpp
//
// Reminders:  task["pp"]["Kp"].get<double>() reads a number of the task (JSON); v.cwiseAbs().maxCoeff(), v.minCoeff(),
//   v.mean(), v.squaredNorm(), v.segment(i, n), v.head(n), v.tail(n); std::numeric_limits<double>::infinity() (<limits>)
//   and quiet_NaN(); std::isnan, std::isfinite. To keep the entries of a vector where a condition holds, collect them
//   in a std::vector<double> and copy it into an Eigen::VectorXd (Eigen has no boolean-mask indexing).
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// =====================================================================  Part A: pre-flight and dry run

std::vector<std::string> preflight(const sbeg::json& record) {
    // TODO A1: write this function (see lab.hpp and the lab handout, Part A).
    // Each violation is a string that starts with the rule label: "B4: stop holder must be ...". The given helpers
    // read the record like Python's dict.get: flag_or(record, "estop_tested", false), text_or(record, "stop_holder"),
    // num_or(dev, "force_max_N", infinity), sub_or_empty(record, "devices"); record.contains("arm"),
    // record["arm"].is_null(); a JSON list compares with == (record["hands_on_handle"] == sbeg::json::array({"operator"})).
    // Limits: HAPTIC_SAFETY.force_max, HAPTIC_SAFETY.x_range, COURSE_ARM_SAFETY.reach_m, ... (course_models.hpp).
    throw std::logic_error("TODO: preflight");
}

ContactEquilibrium contact_equilibrium(const sbeg::json& task, double x_m) {
    // TODO A2: write this function (see lab.hpp and the lab handout, Part A).
    // Solve without contact first; if then x_s > x_w, solve again with the tissue spring.
    throw std::logic_error("TODO: contact_equilibrium");
}

DryRun dry_run(const sbeg::json& task) {
    // TODO A3: write this function (see lab.hpp and the lab handout, Part A).
    // const PPSim s = simulate_pp(task);   then s.F_m, s.F_s, s.x_m, s.x_s, s.E_ch, s.sat_m, s.tripped_m, ...
    throw std::logic_error("TODO: dry_run");
}

// =====================================================================  Part B: bring-up and identification

LogDecrement log_decrement(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double t0, std::optional<double> t_end,
                           int n_max) {
    // TODO B1: write this function (see lab.hpp and the lab handout, Part B).
    // Window: t0 <= t <= (t_end ? *t_end : last t). Peaks: extrema(tt, xx, +1.0); troughs: extrema(tt, xx, -1.0)
    // (indices into the window). Keep the peaks before the last trough, at most n_max.
    throw std::logic_error("TODO: log_decrement");
}

DecayFit fit_free_decay(const Eigen::VectorXd& t, const Eigen::VectorXd& x, double K, double k_g, double T, double t0,
                        std::optional<double> t_end) {
    // TODO B2: write this function (see lab.hpp and the lab handout, Part B).
    // auto model = [](double tau, const Eigen::VectorXd& p) { return p(0) * std::exp(-p(1) * tau) * ... + p(4); };
    // p0 = (x0 - c0, zeta wn, wd, 0, c0) from log_decrement, c0 = mean of the last quarter of the window;
    // const Eigen::VectorXd p = curve_fit(model, tau, xx, p0, 20000);
    throw std::logic_error("TODO: fit_free_decay");
}

WatchdogCheck watchdog_check(const Eigen::VectorXd& t, const Eigen::VectorXd& x, const Eigen::VectorXd& F,
                             double x_range) {
    // TODO B3: write this function (see lab.hpp and the lab handout, Part B).
    throw std::logic_error("TODO: watchdog_check");
}

// =====================================================================  Part C: the teleoperation trial

Eigen::VectorXd resample(const Eigen::VectorXd& t_src, const Eigen::VectorXd& x_src, const Eigen::VectorXd& t_dst) {
    // TODO C1: write this function (see lab.hpp and the lab handout, Part C).  interp(t_dst, t_src, x_src) = np.interp.
    throw std::logic_error("TODO: resample");
}

Eigen::VectorXd channel_energy(const Eigen::VectorXd& t, const Eigen::VectorXd& F_m, const Eigen::VectorXd& x_m,
                               const Eigen::VectorXd& F_c, const Eigen::VectorXd& x_d) {
    // TODO C2: write this function (see lab.hpp and the lab handout, Part C).
    throw std::logic_error("TODO: channel_energy");
}

TrialMetrics trial_metrics(const Log& log, const sbeg::json& task) {
    // TODO C3: write this function (see lab.hpp and the lab handout, Part C).
    // Uniform grid: arange(std::ceil(log.t(0) / h) * h, log.t(last), h); log.aux.at("x_m"), log.force, log.cur;
    // v_m = gradient(x_m, tu) (= np.gradient); the least-squares slope of y against x is
    // sum((x - mean x) (y - mean y)) / sum((x - mean x)^2)  (np.polyfit(x, y, 1)[0]).
    throw std::logic_error("TODO: trial_metrics");
}

SimVsHw sim_vs_hw(const Log& log, const sbeg::json& task, Device dev, double skip) {
    // TODO C4: write this function (see lab.hpp and the lab handout, Part C).
    // simulate_follower(log.t, log.cmd, Kp, Kd, dev, &log.aux.at("f_env"), dt, substeps)
    throw std::logic_error("TODO: sim_vs_hw");
}

// =====================================================================  Part D: safe stop and the IMU at rest

double fade_time(const Eigen::VectorXd& t, const Eigen::MatrixXd& F, double t_trigger, double threshold) {
    // TODO D1: write this function (see lab.hpp and the lab handout, Part D).  F.row(i).cwiseAbs().maxCoeff()
    throw std::logic_error("TODO: fade_time");
}

ImuRestStats imu_rest_stats(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, const Eigen::VectorXd& ax,
                            const Eigen::VectorXd& ay, double t_rest) {
    // TODO D2: write this function (see lab.hpp and the lab handout, Part D).
    // Sample period: median of the time steps (given median); degrees = radians * 180 / PI (PI in course_models.hpp).
    throw std::logic_error("TODO: imu_rest_stats");
}

}  // namespace lab
