// SBEG 461 Lab 5, Part D (C++ track) -- the ROS-free logic of the PID controller node and of the joint simulator.
//
// Given, do not edit. The same classes as the Python nodes ros2/joint_pid_node.py (PidNodeLogic,
// default_controller, make_reference) and ros2/joint_sim_node.py (JointSimLogic), without any ROS dependency, so
// that test_lab.cpp checks them (test_ros_logic_matches_offline_simulation) and ros2/joint_pid_node.cpp wraps the
// controller logic in an rclcpp node. They use YOUR lab::JointPID and lab::pd_gains and the provided lab functions.
#ifndef SBEG461_LAB5_NODE_LOGIC_HPP
#define SBEG461_LAB5_NODE_LOGIC_HPP

#include "lab.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace node_logic {

/// ref(t) -> (qd, qd_dot)
using Reference = std::function<std::pair<double, double>(double)>;

/// The reference types of the node: "hold" (keep q_start), "step" (q_start, then q_target from t_step on),
/// "knee_swing" (data/knee_swing.csv, nearest sample). Any other name: std::invalid_argument.
inline Reference make_reference(const std::string& kind, double q_start, double q_target = 0.0, double t_step = 1.0) {
    if (kind == "hold") return [q_start](double) { return std::make_pair(q_start, 0.0); };
    if (kind == "step")
        return [q_start, q_target, t_step](double t) { return std::make_pair(t >= t_step ? q_target : q_start, 0.0); };
    if (kind == "knee_swing") {
        const lab::KneeSwing ks = lab::load_knee_swing();
        return [ks](double t) {
            const double step = ks.t(1) - ks.t(0);
            const Eigen::Index last = ks.t.size() - 1;
            const Eigen::Index i = std::min(static_cast<Eigen::Index>(std::nearbyint(t / step)), last);
            return std::make_pair(ks.qd(i), ks.qd_dot(i));
        };
    }
    throw std::invalid_argument("unknown reference '" + kind + "'");
}

/// ROS-free controller logic: on_joint_state(names, positions, velocities) -> (tau, debug list
/// [t, qd, q, e, tau, P, I, D, saturated]). Controller time is k * dt, k = number of states received.
class PidNodeLogic {
public:
    PidNodeLogic(lab::JointPID controller, std::string joint_name, std::string reference = "hold",
                 double q_target_in = 0.0, double t_step_in = 1.0, std::string velocity_in = "measured")
        : ctrl(std::move(controller)),
          joint(std::move(joint_name)),
          kind(std::move(reference)),
          q_target(q_target_in),
          t_step(t_step_in),
          velocity(std::move(velocity_in)) {}

    std::pair<double, std::vector<double>> on_joint_state(const std::vector<std::string>& names,
                                                          const std::vector<double>& positions,
                                                          const std::vector<double>& velocities) {
        const auto it = std::find(names.begin(), names.end(), joint);
        const std::size_t i = it == names.end() ? 0 : static_cast<std::size_t>(it - names.begin());
        if (i >= positions.size()) throw std::invalid_argument("joint state without a position for " + joint);
        const double q = positions[i];
        std::optional<double> dq;
        if (velocity == "measured" && velocities.size() > i) dq = velocities[i];
        if (!ref) {  // first sample: start the reference here
            ref = make_reference(kind, q, q_target, t_step);
            ctrl.reset(q);
        }
        const double t = static_cast<double>(k) * ctrl.dt;
        const auto [qd, vd] = ref(t);
        const double tau = ctrl.update(qd, vd, q, dq);
        ++k;
        const lab::PidTerms& T = ctrl.terms;
        return {tau, {t, qd, q, qd - q, tau, T.P, T.I, T.D, ctrl.saturated ? 1.0 : 0.0}};
    }

    lab::JointPID ctrl;  ///< the controller (the ROS node changes ctrl.kp, ctrl.ki, ctrl.kd at run time)
    std::string joint, kind;
    double q_target, t_step;
    std::string velocity;
    Reference ref;  ///< empty until the first joint state
    long long k = 0;
};

/// JointPID with kp, kd from lab::pd_gains for the plant (unless overridden by a value >= 0) and the plant's
/// torque limit (unless tau_max >= 0 is given); ki < 0 is treated as 0.
inline lab::JointPID default_controller(const std::string& plant_name, double zeta, double wn, double dt,
                                        double kp = -1.0, double ki = 0.0, double kd = -1.0, double tau_max = -1.0,
                                        bool anti_windup = true) {
    const lab::Plant p = lab::load_plant(plant_name);
    const auto [kp0, kd0] = lab::pd_gains(p.M, p.b, zeta, wn);
    const std::optional<double> lim = tau_max < 0 ? p.tau_max : std::optional<double>(tau_max);
    return lab::JointPID(kp < 0 ? kp0 : kp, std::max(ki, 0.0), kd < 0 ? kd0 : kd, dt, lim, anti_windup);
}

/// Plant state and integration, free of ROS. apply(tau) stores the command (clipped to the plant's tau_max,
/// zero-order hold); step() advances one sample with `substeps` RK4 steps; state() = (q, dq, tau).
class JointSimLogic {
public:
    JointSimLogic(lab::Plant plant_in, double dt_in, double q0, double dq0 = 0.0, int substeps_in = 4)
        : plant(std::move(plant_in)), dt(dt_in), substeps(substeps_in), x(q0, dq0) {}

    void apply(double tau_in) { tau = plant.tau_max ? std::clamp(tau_in, -*plant.tau_max, *plant.tau_max) : tau_in; }

    std::tuple<double, double, double> state() const { return {x(0), x(1), tau}; }

    std::tuple<double, double, double> step() {
        const double h = dt / substeps;
        const double u = tau;
        const lab::StateFunction f = [this, u](const Eigen::Vector2d& s) { return lab::plant_rhs(plant, s, u); };
        for (int i = 0; i < substeps; ++i) x = lab::rk4_step(f, x, h);
        ++k;
        return state();
    }

    lab::Plant plant;
    double dt;
    int substeps;
    Eigen::Vector2d x;
    double tau = 0.0;
    long long k = 0;
};

}  // namespace node_logic

#endif  // SBEG461_LAB5_NODE_LOGIC_HPP
