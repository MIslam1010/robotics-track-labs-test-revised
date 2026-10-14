// SBEG 461 Lab 6 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_config, arm_model, make_case, tau_limits     data/lab06_config.json and the robot models
//   time_grid, quintic_reference, cartesian_reference  references
//   rk4_step, diag_gain, Plant, simulate_arm           the closed-loop simulator
//   JointController / TaskController constructors      (you write reset() and operator() in starter.cpp)
//
// These are the "Provided" functions of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

sbeg::json load_config() { return sbeg::load_json(sbeg::lab_path("../data/lab06_config.json")); }

Robot arm_model(double payload, double mass_scale, const Eigen::VectorXd& friction) {
    Robot r;
    static_cast<course_arm::Robot&>(r) = course_arm::course_arm(0.0, payload);
    for (auto& G : r.Glist) G = (mass_scale * G).eval();
    r.Blist = mr::adjoint(mr::trans_inv(r.M)) * r.Slist;
    r.b = friction.size() == 0 ? Eigen::VectorXd::Zero(6) : friction;
    r.mass_scale = mass_scale;
    r.payload = payload;
    return r;
}

std::pair<Robot, Robot> make_case(const std::string& name, const sbeg::json& cfg) {
    const sbeg::json& c = cfg.at("cases").at(name);
    const Eigen::VectorXd friction = sbeg::as_vec(cfg.at("friction_b"));
    Robot plant = arm_model(cfg.at("payload").get<double>(), 1.0, friction);
    Robot model = arm_model(c.at("model_payload").get<double>(), c.at("model_mass_scale").get<double>(),
                            c.at("model_friction").get<bool>() ? friction : Eigen::VectorXd());
    return {plant, model};
}

std::pair<Robot, Robot> make_case(const std::string& name) { return make_case(name, load_config()); }

Eigen::VectorXd tau_limits(const Robot& robot, const sbeg::json& cfg) {
    return cfg.at("tau_limit_fraction").get<double>() * robot.tau_max;
}

Eigen::VectorXd tau_limits(const Robot& robot) { return tau_limits(robot, load_config()); }

Eigen::VectorXd time_grid(double T, double dt) {
    const long n = std::lround(T / dt);
    Eigen::VectorXd t(n + 1);
    for (long i = 0; i <= n; ++i) t(i) = static_cast<double>(i) * dt;
    return t;
}

JointReference quintic_reference(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, double T,
                                 const Eigen::VectorXd& t) {
    const Eigen::Index N = t.size(), n = q0.size();
    const Eigen::VectorXd d = q1 - q0;
    JointReference r;
    r.qd.resize(N, n);
    r.dqd.resize(N, n);
    r.ddqd.resize(N, n);
    for (Eigen::Index k = 0; k < N; ++k) {
        const double tau = std::clamp(t(k) / T, 0.0, 1.0);
        const double s = 10 * std::pow(tau, 3) - 15 * std::pow(tau, 4) + 6 * std::pow(tau, 5);
        double sd = (30 * tau * tau - 60 * std::pow(tau, 3) + 30 * std::pow(tau, 4)) / T;
        double sdd = (60 * tau - 180 * tau * tau + 120 * std::pow(tau, 3)) / (T * T);
        if (t(k) > T) sd = sdd = 0.0;
        r.qd.row(k) = (q0 + s * d).transpose();
        r.dqd.row(k) = (sd * d).transpose();
        r.ddqd.row(k) = (sdd * d).transpose();
    }
    return r;
}

CartesianReference cartesian_reference(const Eigen::Matrix4d& X0, const Eigen::Vector3d& dp_s, double T,
                                       const Eigen::VectorXd& t) {
    const Eigen::Matrix3d R0 = X0.topLeftCorner<3, 3>();
    const Eigen::Vector3d p0 = X0.topRightCorner<3, 1>();
    const JointReference p = quintic_reference(Eigen::VectorXd::Zero(3), dp_s, T, t);
    CartesianReference r;
    const Eigen::Index N = t.size();
    r.Vd = Eigen::MatrixXd::Zero(N, 6);
    r.dVd = Eigen::MatrixXd::Zero(N, 6);
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::Vector3d pk = p.qd.row(k).transpose();
        r.Xd.push_back(mr::rp_to_trans(R0, p0 + pk));
        r.Vd.block<1, 3>(k, 3) = p.dqd.row(k) * R0;  // rows: R0^T p'
        r.dVd.block<1, 3>(k, 3) = p.ddqd.row(k) * R0;
    }
    return r;
}

Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& x, double h) {
    const Eigen::VectorXd k1 = f(x);
    const Eigen::VectorXd k2 = f(x + 0.5 * h * k1);
    const Eigen::VectorXd k3 = f(x + 0.5 * h * k2);
    const Eigen::VectorXd k4 = f(x + h * k3);
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
}

Eigen::VectorXd diag_gain(const Gain& K, Eigen::Index n) {
    return K.is_scalar() ? Eigen::VectorXd::Constant(n, K.scalar()) : K.vector();
}

Plant::Plant(const Robot& robot) : r(robot), b(robot.b.size() ? robot.b : Eigen::VectorXd::Zero(robot.n)) {}

Eigen::VectorXd Plant::accel(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& tau) const {
    const Eigen::VectorXd tau_net = tau - b.cwiseProduct(dq);
    return mr::forward_dynamics(q, dq, tau_net, r.g, Vector6d::Zero(), r.Mlist, r.Glist, r.Slist);
}

SimResult simulate_arm(const Plant& P, Controller& controller, const Eigen::VectorXd& t, const Eigen::VectorXd& q0,
                       const Eigen::VectorXd& dq0, int substeps, const Eigen::VectorXd& tau_limit) {
    const Eigen::Index N = t.size(), n = q0.size();
    const double dt = t(1) - t(0);
    const double h = dt / substeps;
    controller.reset();
    Eigen::VectorXd x(2 * n);
    x << q0, (dq0.size() == 0 ? Eigen::VectorXd::Zero(n) : dq0);
    SimResult out;
    out.t = t;
    out.q.resize(N, n);
    out.dq.resize(N, n);
    out.tau.resize(N, n);
    for (Eigen::Index k = 0; k < N; ++k) {
        out.q.row(k) = x.head(n).transpose();
        out.dq.row(k) = x.tail(n).transpose();
        Eigen::VectorXd tau = controller(static_cast<int>(k), x.head(n), x.tail(n));
        if (tau.size() != n) throw std::invalid_argument("simulate_arm: the controller must return n torques");
        if (tau_limit.size() != 0) tau = tau.cwiseMax(-tau_limit).cwiseMin(tau_limit);
        out.tau.row(k) = tau.transpose();
        if (k + 1 < N) {
            const StateFunction f = [&P, &tau, n](const Eigen::VectorXd& s) {
                Eigen::VectorXd ds(2 * n);
                ds << s.tail(n), P.accel(s.head(n), s.tail(n), tau);
                return ds;
            };
            for (int i = 0; i < substeps; ++i) x = rk4_step(f, x, h);
        }
    }
    return out;
}

SimResult simulate_arm(const Robot& plant, Controller& controller, const Eigen::VectorXd& t, const Eigen::VectorXd& q0,
                       const Eigen::VectorXd& dq0, int substeps, const Eigen::VectorXd& tau_limit) {
    return simulate_arm(Plant(plant), controller, t, q0, dq0, substeps, tau_limit);
}

JointController::JointController(const std::string& law_, const Eigen::VectorXd& t_, const Eigen::MatrixXd& qd_,
                                 const Eigen::MatrixXd& dqd_, const Eigen::MatrixXd& ddqd_, const Gain& Kp_,
                                 const Gain& Ki_, const Gain& Kd_, const std::optional<Robot>& model_)
    : law(law_), t(t_), qd(qd_), dqd(dqd_), ddqd(ddqd_), Kp(Kp_), Ki(Ki_), Kd(Kd_), model(model_) {
    if (law != "pid" && law != "pd_gravity" && law != "feedforward" && law != "computed_torque")
        throw std::invalid_argument("law must be one of pid, pd_gravity, feedforward, computed_torque");
    if (law != "pid" && !model) throw std::invalid_argument("law " + law + " needs a model");
    dt = t(1) - t(0);
    reset();
}

TaskController::TaskController(const Eigen::VectorXd& t_, const mr::TransformList& Xd_, const Eigen::MatrixXd& Vd_,
                               const Eigen::MatrixXd& dVd_, const Gain& kp_, const Gain& ki_, const Gain& kd_,
                               const Robot& model_)
    : t(t_), Xd(Xd_), Vd(Vd_), dVd(dVd_), kp(kp_), ki(ki_), kd(kd_), model(model_) {
    dt = t(1) - t(0);
    reset();
}

}  // namespace lab
