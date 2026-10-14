// SBEG 461 Lab 5 -- Error dynamics and PID control of a single joint (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp, with the full docstrings).
// Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 5 --cpp
//
// Notation follows Modern Robotics (MR) Ch. 11 (see lab.hpp): e = qd - q, e'' + 2 zeta wn e' + wn^2 e = 0.
// These functions become sbeg461_dynamics/control.py in your cumulative library (Python), or control.cpp if you
// keep a C++ library. The ROS 2 controller node (ros2/joint_pid_node.cpp) uses YOUR JointPID and pd_gains.
//
// Eigen reminders: t.size(), t(k), t(t.size() - 1) (= t[-1]), e.minCoeff(), v.cwiseAbs().maxCoeff(),
//   elementwise maths on arrays: (w * t.array()).cos(), .exp(), .sin(); back to a vector with .matrix();
//   Eigen::VectorXd v(n) (uninitialised), Eigen::VectorXd::Zero(n), Eigen::Vector3d(a, b, c).
// std::optional reminders: if (qdot) { use *qdot; }   std::optional<double> none;   std::nullopt = Python None.
// np.clip(v, lo, hi) -> std::clamp(v, lo, hi) (<algorithm>);  math.copysign -> std::copysign;
// np.round (to nearest, halves to even) -> std::nearbyint.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lab {

namespace {
[[maybe_unused]] constexpr double PI = 3.14159265358979323846;  // use PI, not M_PI (not standard C++)
}  // namespace

// ==========================================================================================
// Provided (do not edit): plants, reference trajectories, RK4 integrator
// ==========================================================================================

Plant load_plant(const std::string& name) {
    // Given.
    const sbeg::json all = sbeg::load_json(sbeg::lab_path("../data/joint_plants.json"));
    if (!all.contains(name)) throw std::invalid_argument("unknown plant '" + name + "'");
    const sbeg::json& j = all.at(name);
    auto opt = [&j](const char* key) -> std::optional<double> {
        if (!j.contains(key) || j.at(key).is_null()) return std::nullopt;
        return j.at(key).get<double>();
    };
    Plant p;
    p.name = j.value("name", name);
    p.M = j.at("M").get<double>();
    p.b = j.at("b").get<double>();
    p.g_cos = j.at("g_cos").get<double>();
    p.g_sin = j.at("g_sin").get<double>();
    p.tau_max = opt("tau_max");
    p.qdot_max = opt("qdot_max");
    if (j.contains("q_limits")) p.q_limits = sbeg::as<Eigen::Vector2d>(j.at("q_limits"));
    if (j.contains("locked_q")) p.locked_q = sbeg::as_vec(j.at("locked_q"));
    p.joint_index = j.value("joint_index", 0);
    p.payload = j.value("payload", 0.0);
    p.description = j.value("description", std::string());
    p.source = j.value("source", std::string());
    return p;
}

double plant_gravity(const Plant& plant, double q) {
    // Given.
    return plant.g_cos * std::cos(q) + plant.g_sin * std::sin(q);
}

Eigen::Vector2d plant_rhs(const Plant& plant, const Eigen::Vector2d& x, double tau, bool gravity) {
    // Given.
    const double q = x(0), dq = x(1);
    const double g = gravity ? plant_gravity(plant, q) : 0.0;
    return Eigen::Vector2d(dq, (tau - plant.b * dq - g) / plant.M);
}

Eigen::Vector2d rk4_step(const StateFunction& f, const Eigen::Vector2d& x, double h) {
    // Given.
    const Eigen::Vector2d k1 = f(x);
    const Eigen::Vector2d k2 = f(x + 0.5 * h * k1);
    const Eigen::Vector2d k3 = f(x + 0.5 * h * k2);
    const Eigen::Vector2d k4 = f(x + h * k3);
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
}

Eigen::VectorXd time_grid(double T, double dt) {
    // Given.
    const Eigen::Index n = static_cast<Eigen::Index>(std::nearbyint(T / dt));  // Python round(): halves to even
    Eigen::VectorXd t(n + 1);
    for (Eigen::Index k = 0; k <= n; ++k) t(k) = static_cast<double>(k) * dt;
    return t;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> step_reference(const Eigen::VectorXd& t, double q0, double q1,
                                                           double t_step) {
    // Given.
    Eigen::VectorXd qd(t.size());
    for (Eigen::Index k = 0; k < t.size(); ++k) qd(k) = t(k) >= t_step ? q1 : q0;
    return {qd, Eigen::VectorXd::Zero(t.size())};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ramp_reference(const Eigen::VectorXd& t, double q0, double rate) {
    // Given.
    const Eigen::VectorXd qd = (q0 + rate * t.array()).matrix();
    return {qd, Eigen::VectorXd::Constant(t.size(), rate)};
}

KneeSwing load_knee_swing() {
    // Given.
    const sbeg::CsvTable c = sbeg::read_csv(sbeg::lab_path("../data/knee_swing.csv"));
    return KneeSwing{c.data.col(0), c.data.col(1), c.data.col(2), c.data.col(3)};
}

// ==========================================================================================
// Part A -- error dynamics (MR Sec. 11.2)
// ==========================================================================================

std::pair<double, double> second_order_params(double a1, double a0) {
    // TODO: throw std::invalid_argument("a0 must be positive (wn^2 > 0)") if a0 <= 0; return {zeta, wn};
    throw std::logic_error("TODO: second_order_params");
}

Eigen::VectorXcd char_roots(const Eigen::VectorXd& coeffs) {
    // TODO: the roots are the eigenvalues of the companion matrix (this is what numpy.roots does): for
    // c0 s^p + c1 s^(p-1) + ... + cp, A is p x p with first row -c1/c0, ..., -cp/c0 and ones below the diagonal
    // (A(k, k - 1) = 1).  Eigen::EigenSolver<Eigen::MatrixXd> es(A, false); es.eigenvalues() is a VectorXcd.
    throw std::logic_error("TODO: char_roots");
}

bool is_stable(const Eigen::VectorXd& coeffs, double tol) {
    // TODO: every root r must have r.real() < -tol
    throw std::logic_error("TODO: is_stable");
}

Eigen::VectorXd unit_error_response(double zeta, double wn, const Eigen::VectorXd& t) {
    // TODO: three cases (MR p. 413); |zeta - 1| < 1e-9 is critical damping.  Work on t.array().
    throw std::logic_error("TODO: unit_error_response");
}

double overshoot_from_zeta(double zeta) {
    // TODO: use PI (defined above), not M_PI
    throw std::logic_error("TODO: overshoot_from_zeta");
}

double zeta_from_overshoot(double overshoot_pct) {
    // TODO
    throw std::logic_error("TODO: zeta_from_overshoot");
}

StepMetrics step_metrics(const Eigen::VectorXd& t, const Eigen::VectorXd& e, std::optional<double> ess, double band) {
    // TODO: ess defaults to the last sample (if (!ess) ...); NaN: std::numeric_limits<double>::quiet_NaN().
    // return StepMetrics{ess_value, overshoot, settling_time};
    throw std::logic_error("TODO: step_metrics");
}

// ==========================================================================================
// Part B -- motion control with velocity inputs (MR Sec. 11.3.1)
// ==========================================================================================

std::pair<double, double> pi_velocity_gains(double zeta, double wn) {
    // TODO: return {Kp, Ki};
    throw std::logic_error("TODO: pi_velocity_gains");
}

VelocitySim simulate_velocity_joint(const Eigen::VectorXd& t, const Eigen::VectorXd& qd, const Eigen::VectorXd& qd_dot,
                                    double kp, double ki, double q0, bool feedforward,
                                    std::optional<double> qdot_max) {
    // TODO: VelocitySim r{Eigen::VectorXd(n), Eigen::VectorXd(n), Eigen::VectorXd(n)};  then the loop of the
    // docstring (u_k from e_k, then q_{k+1} = q_k + u_k dt).
    throw std::logic_error("TODO: simulate_velocity_joint");
}

// ==========================================================================================
// Part C -- PID control with torque inputs (MR Sec. 11.4.1.1)
// ==========================================================================================

std::pair<double, double> pd_gains(double M, double b, double zeta, double wn) {
    // TODO: return {Kp, Kd};
    throw std::logic_error("TODO: pd_gains");
}

double ki_max(double M, double b, double kp, double kd) {
    // TODO
    throw std::logic_error("TODO: ki_max");
}

Eigen::VectorXd pid_char_poly(double M, double b, double kp, double ki, double kd) {
    // TODO: 3 coefficients if ki == 0, else 4 (Eigen::Vector3d(...) / Eigen::Vector4d(...) convert to VectorXd)
    throw std::logic_error("TODO: pid_char_poly");
}

double pd_gravity_error(const Plant& plant, double kp, double qd) {
    // TODO: Newton's method (at most 50 iterations, stop when |step| < 1e-14); use plant_gravity.
    throw std::logic_error("TODO: pd_gravity_error");
}

JointPID::JointPID(double kp0, double ki0, double kd0, double dt0, std::optional<double> tau_max0, bool anti_windup0)
    : kp(kp0), ki(ki0), kd(kd0), dt(dt0), tau_max(tau_max0), anti_windup(anti_windup0) {
    // Given.
    reset();
}

void JointPID::reset(std::optional<double> q0) {
    // TODO: eint, q_prev, terms (PidTerms{}), saturated
    throw std::logic_error("TODO: reset");
}

double JointPID::update(double qd, double qd_dot, double q, std::optional<double> qdot, double tau_ff) {
    // TODO: see lab.hpp (MR Fig. 11.15 + torque limit + anti-windup); keep terms and saturated up to date.
    throw std::logic_error("TODO: update");
}

JointSim simulate_joint(const Plant& plant, JointPID& ctrl, const Eigen::VectorXd& t, const Eigen::VectorXd& qd,
                        const Eigen::VectorXd& qd_dot, double q0, double dq0, int substeps,
                        const std::string& velocity, std::optional<double> enc_res, const Eigen::VectorXd& tau_ff,
                        bool gravity) {
    // TODO: torque once per sample, HELD for all substeps. The right-hand side for rk4_step is a lambda:
    //     const StateFunction f = [&plant, tau, gravity](const Eigen::Vector2d& s) {
    //         return plant_rhs(plant, s, tau, gravity); };
    // Store x before the update (q(k), dq(k)); e = qd - q at the end.
    throw std::logic_error("TODO: simulate_joint");
}

}  // namespace lab
