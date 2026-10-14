// SBEG 461 Hardware rotation 1 (Week 8) -- provided helpers (C++ track).  GIVEN code: do not edit; use these
// functions freely.
//
// The "Provided" part of starter.py, translated to C++:
//   inverse_dynamics, gravity_torques, gravity_torques_rows, mass_diagonal   RNEA and M(q) through mrcpp
//   load_task, data_path, write_log, load_log, log_window                    task file and station logs
//   step_command, smooth_move                                                station command tables
//   check_command, tracking_metrics, average_approaches, step_response,
//   normalised_rms, median                                                   checks and metrics
//   plant_gravity, plant_rhs, rk4_step, pd_gains, pd_gravity_error,
//   JointPID, simulate_joint                                                 the Lab 5 PID code, unchanged
//   joint_plant, nominal_servo, make_servo, ServoArm, ReplayIO               one joint as a Lab 5 plant; the servo
//                                                                            model of the arm; log replay
// and one replacement for SciPy:
//   least_squares                                                            bounded Levenberg-Marquardt
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab {
namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

double median_step(const VectorXd& t) {
    if (t.size() < 2) throw std::invalid_argument("at least two samples are needed");
    return median((t.tail(t.size() - 1) - t.head(t.size() - 1)).eval());
}

}  // namespace

double median(const Eigen::VectorXd& v) {
    if (v.size() == 0) throw std::invalid_argument("median of an empty vector");
    std::vector<double> w(v.data(), v.data() + v.size());
    std::sort(w.begin(), w.end());
    const std::size_t m = w.size() / 2;
    return w.size() % 2 ? w[m] : 0.5 * (w[m - 1] + w[m]);
}

// ------------------------------------------------------------------ dynamics
Eigen::VectorXd inverse_dynamics(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& ddq, bool gravity) {
    const Eigen::Vector3d g = gravity ? model.g : Eigen::Vector3d::Zero();
    return mr::inverse_dynamics(q, dq, ddq, g, mr::Vector6d::Zero(), model.Mlist, model.Glist, model.Slist);
}

Eigen::VectorXd gravity_torques(const Model& model, const Eigen::VectorXd& q) {
    const VectorXd z = VectorXd::Zero(q.size());
    return inverse_dynamics(model, q, z, z);
}

Eigen::MatrixXd gravity_torques_rows(const Model& model, const Eigen::MatrixXd& Q) {
    MatrixXd G(Q.rows(), Q.cols());
    for (Index k = 0; k < Q.rows(); ++k) G.row(k) = gravity_torques(model, Q.row(k).transpose()).transpose();
    return G;
}

// ------------------------------------------------------------------ task and logs
std::string data_path(const std::string& name) { return sbeg::lab_path("../data/" + name); }

sbeg::json load_task(const std::string& name) { return sbeg::load_json(data_path(name)); }

void write_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd,
               const Eigen::MatrixXd& meas, const Eigen::MatrixXd& cur) {
    const Index N = t.size(), n = cmd.cols();
    if (cmd.rows() != N) throw std::invalid_argument("write_log: cmd must have one row per sample");
    std::vector<std::string> header = {"t"};
    std::vector<const MatrixXd*> blocks = {&cmd};
    for (Index j = 0; j < n; ++j) header.push_back("cmd_" + std::to_string(j + 1));
    if (meas.size() > 0) {
        if (meas.rows() != N || meas.cols() != n) throw std::invalid_argument("write_log: meas must be N x n");
        blocks.push_back(&meas);
        for (Index j = 0; j < n; ++j) header.push_back("meas_" + std::to_string(j + 1));
    }
    if (cur.size() > 0) {
        if (cur.rows() != N || cur.cols() != n) throw std::invalid_argument("write_log: cur must be N x n");
        blocks.push_back(&cur);
        for (Index j = 0; j < n; ++j) header.push_back("cur_" + std::to_string(j + 1));
    }
    std::ofstream out(sbeg::fs_path(path), std::ios::binary);
    if (!out) throw std::runtime_error("cannot write file: " + path);
    for (std::size_t k = 0; k < header.size(); ++k) out << (k ? "," : "") << header[k];
    out << "\n" << std::setprecision(15);
    auto r7 = [](double x) { return std::nearbyint(x * 1e7) / 1e7; };  // np.round(..., 7)
    for (Index i = 0; i < N; ++i) {
        out << r7(t(i));
        for (const MatrixXd* b : blocks)
            for (Index j = 0; j < n; ++j) out << "," << r7((*b)(i, j));
        out << "\n";
    }
}

Log load_log(const std::string& path) {
    const sbeg::CsvTable T = sbeg::read_csv(path);
    Log L;
    auto group = [&](const std::string& prefix) {
        std::vector<std::pair<int, Index>> cols;
        for (std::size_t c = 0; c < T.header.size(); ++c) {
            const std::string& h = T.header[c];
            if (h.rfind(prefix + "_", 0) == 0) cols.emplace_back(std::stoi(h.substr(prefix.size() + 1)), Index(c));
        }
        std::sort(cols.begin(), cols.end());
        MatrixXd M(T.data.rows(), static_cast<Index>(cols.size()));
        for (std::size_t k = 0; k < cols.size(); ++k) M.col(static_cast<Index>(k)) = T.data.col(cols[k].second);
        return cols.empty() ? MatrixXd() : M;
    };
    for (std::size_t c = 0; c < T.header.size(); ++c) {
        const std::string& h = T.header[c];
        if (h == "t") L.t = T.data.col(static_cast<Index>(c));
        else if (h.find('_') == std::string::npos) L.extra[h] = T.data.col(static_cast<Index>(c));
    }
    L.cmd = group("cmd");
    L.meas = group("meas");
    L.cur = group("cur");
    return L;
}

// ------------------------------------------------------------------ command check and metrics
CommandCheck check_command(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, const Model& model,
                           double speed_fraction, double margin_deg, double tol) {
    const Index N = t.size(), n = q.cols();
    if (N < 2 || q.rows() != N) throw std::invalid_argument("check_command: q must have one row per sample (N >= 2)");
    const double m = margin_deg * DEG;
    CommandCheck c;
    c.within_limits = true;
    for (Index j = 0; j < n; ++j)
        if (q.col(j).minCoeff() < model.limits(j, 0) + m || q.col(j).maxCoeff() > model.limits(j, 1) - m)
            c.within_limits = false;
    double ratio = 0.0;
    VectorXd steps(N - 1);
    for (Index k = 0; k + 1 < N; ++k) {
        const VectorXd dq = q.row(k + 1).transpose() - q.row(k).transpose();
        const double dt = t(k + 1) - t(k);
        for (Index j = 0; j < n; ++j)
            ratio = std::max(ratio, std::abs(dq(j) / dt) / (speed_fraction * model.qdot_max(j)));
        steps(k) = dq.cwiseAbs().maxCoeff();
    }
    c.peak_speed_ratio = ratio;
    c.starts_at_rest = steps(0) < 1e-3;
    c.ends_at_rest = steps(N - 2) < 1e-3;
    c.speed_ok = ratio <= 1.0 + tol;
    c.ok = c.within_limits && c.speed_ok && c.starts_at_rest && c.ends_at_rest;
    return c;
}

TrackingMetrics tracking_metrics(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd, const Eigen::MatrixXd& meas,
                                 double max_lag) {
    const Index N = t.size(), n = cmd.cols();
    if (cmd.rows() != N || meas.rows() != N || meas.cols() != n)
        throw std::invalid_argument("tracking_metrics: cmd and meas must be N x n");
    const MatrixXd e = meas - cmd;
    const double dt = median_step(t);
    double best_r = std::numeric_limits<double>::infinity();
    Index best_k = 0;
    const Index kmax = static_cast<Index>(std::nearbyint(max_lag / dt));
    for (Index k = 0; k <= kmax && k < N; ++k) {
        const MatrixXd ek = meas.bottomRows(N - k) - cmd.topRows(N - k);
        const double r = std::sqrt(ek.array().square().mean());
        if (r < best_r - 1e-15) {
            best_r = r;
            best_k = k;
        }
    }
    const MatrixXd ek = meas.bottomRows(N - best_k) - cmd.topRows(N - best_k);
    TrackingMetrics m;
    m.rms = e.array().square().colwise().mean().sqrt().transpose();
    m.max_abs = e.cwiseAbs().colwise().maxCoeff().transpose();
    m.lag = static_cast<double>(best_k) * dt;
    m.rms_after_lag = ek.array().square().colwise().mean().sqrt().transpose();
    return m;
}

// ------------------------------------------------------------------ static poses, step response
StaticPoses average_approaches(const Log& log) {
    const auto it = log.extra.find("pose");
    if (it == log.extra.end()) throw std::invalid_argument("average_approaches: the log has no 'pose' column");
    const VectorXd& pose = it->second;
    std::vector<int> ids;
    for (Index i = 0; i < pose.size(); ++i) ids.push_back(static_cast<int>(pose(i)));
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    StaticPoses s;
    s.pose = ids;
    const Index P = static_cast<Index>(ids.size()), n = log.cmd.cols();
    s.cmd = MatrixXd::Zero(P, n);
    s.meas = MatrixXd::Zero(P, n);
    s.cur = MatrixXd::Zero(P, n);
    for (Index p = 0; p < P; ++p) {
        int cnt = 0;
        for (Index i = 0; i < pose.size(); ++i) {
            if (static_cast<double>(ids[static_cast<std::size_t>(p)]) != pose(i)) continue;
            s.cmd.row(p) += log.cmd.row(i);
            s.meas.row(p) += log.meas.row(i);
            s.cur.row(p) += log.cur.row(i);
            ++cnt;
        }
        s.cmd.row(p) /= cnt;
        s.meas.row(p) /= cnt;
        s.cur.row(p) /= cnt;
    }
    return s;
}

Eigen::VectorXd step_response(const Eigen::VectorXd& tau, double wn, double zeta) {
    VectorXd y(tau.size());
    for (Index i = 0; i < tau.size(); ++i) {
        const double x = std::max(tau(i), 0.0);
        double v;
        if (zeta < 1.0 - 1e-6) {
            const double wd = wn * std::sqrt(1 - zeta * zeta);
            v = 1 - std::exp(-zeta * wn * x) * (std::cos(wd * x) + zeta / std::sqrt(1 - zeta * zeta) * std::sin(wd * x));
        } else if (zeta > 1.0 + 1e-6) {
            const double a = wn * (zeta + std::sqrt(zeta * zeta - 1)), b = wn * (zeta - std::sqrt(zeta * zeta - 1));
            v = 1 - (a * std::exp(-b * x) - b * std::exp(-a * x)) / (a - b);
        } else {
            v = 1 - std::exp(-wn * x) * (1 + wn * x);
        }
        y(i) = tau(i) > 0 ? v : 0.0;
    }
    return y;
}

// ------------------------------------------------------------------ SciPy replacement
LeastSquaresResult least_squares(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& fun,
                                 const Eigen::VectorXd& x0, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 const Eigen::VectorXd& x_scale, int max_iter) {
    const Index p = x0.size();
    if (lb.size() != p || ub.size() != p || x_scale.size() != p)
        throw std::invalid_argument("least_squares: x0, lb, ub and x_scale must have the same length");
    auto clip = [&](const VectorXd& x) { return x.cwiseMax(lb).cwiseMin(ub).eval(); };
    VectorXd x = clip(x0);
    VectorXd r = fun(x);
    double cost = 0.5 * r.squaredNorm();
    double lambda = 1e-3;
    for (int it = 0; it < max_iter; ++it) {
        MatrixXd J(r.size(), p);  // forward differences (backward at an upper bound), in scaled variables
        for (Index k = 0; k < p; ++k) {
            double hk = 1e-7 * std::max(std::abs(x(k)), x_scale(k));
            VectorXd xk = x;
            if (xk(k) + hk > ub(k)) hk = -hk;
            xk(k) += hk;
            J.col(k) = (fun(xk) - r) / hk * x_scale(k);
        }
        const MatrixXd A = J.transpose() * J;
        const VectorXd g = J.transpose() * r;
        if (g.lpNorm<Eigen::Infinity>() < 1e-14) break;
        bool accepted = false;
        VectorXd xn, rn;
        double cn = cost;
        for (int inner = 0; inner < 30 && !accepted; ++inner) {
            const MatrixXd Al = A + lambda * MatrixXd::Identity(p, p);
            const VectorXd du = Al.ldlt().solve(-g);
            xn = clip(x + x_scale.cwiseProduct(du));
            rn = fun(xn);
            cn = 0.5 * rn.squaredNorm();
            if (std::isfinite(cn) && cn < cost) {
                accepted = true;
                lambda = std::max(lambda / 3.0, 1e-12);
            } else {
                lambda *= 4.0;
            }
        }
        if (!accepted) break;
        const double step = (xn - x).cwiseQuotient(x_scale).norm();
        const double drop = cost - cn;
        x = xn;
        r = rn;
        cost = cn;
        if (drop <= 1e-12 * cost || step <= 1e-10) break;
    }
    return LeastSquaresResult{x, cost};
}

// ------------------------------------------------------------------ more task and log helpers
Eigen::VectorXd mass_diagonal(const Model& model, const Eigen::VectorXd& q) {
    return mr::mass_matrix(q, model.Mlist, model.Glist, model.Slist).diagonal();
}

Log log_window(const Log& log, double t0, double t1, std::optional<double> mode) {
    std::vector<Index> rows;
    const auto it = log.extra.find("mode");
    if (mode && it == log.extra.end()) throw std::invalid_argument("log_window: the log has no 'mode' column");
    for (Index i = 0; i < log.t.size(); ++i) {
        bool keep = log.t(i) >= t0 - 1e-9 && log.t(i) <= t1 + 1e-9;
        if (mode) keep = keep && it->second(i) == *mode;
        if (keep) rows.push_back(i);
    }
    auto pick = [&](const MatrixXd& M) {
        if (M.size() == 0) return MatrixXd();
        MatrixXd out(static_cast<Index>(rows.size()), M.cols());
        for (std::size_t r = 0; r < rows.size(); ++r) out.row(static_cast<Index>(r)) = M.row(rows[r]);
        return out;
    };
    Log w;
    w.t = pick(log.t);
    w.cmd = pick(log.cmd);
    w.meas = pick(log.meas);
    w.cur = pick(log.cur);
    for (const auto& kv : log.extra) w.extra[kv.first] = pick(kv.second);
    return w;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> step_command(const Eigen::VectorXd& pose, const std::vector<int>& joints,
                                                         const std::vector<double>& t_steps, double amplitude_deg,
                                                         double duration, double dt) {
    if (joints.size() != t_steps.size()) throw std::invalid_argument("step_command: one step time per joint");
    const Index N = static_cast<Index>(std::nearbyint(duration / dt)) + 1;
    VectorXd t(N);
    for (Index k = 0; k < N; ++k) t(k) = static_cast<double>(k) * dt;
    MatrixXd cmd = pose.transpose().replicate(N, 1);
    for (std::size_t s = 0; s < joints.size(); ++s)
        for (Index k = 0; k < N; ++k)
            if (t(k) >= t_steps[s] - 1e-9) cmd(k, joints[s] - 1) += amplitude_deg * DEG;
    return {t, cmd};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> smooth_move(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, double T,
                                                        double dt, double dwell, bool back) {
    const Index nd = static_cast<Index>(std::nearbyint(dwell / dt)), nT = static_cast<Index>(std::nearbyint(T / dt));
    std::vector<VectorXd> rows;
    auto hold = [&](const VectorXd& q, Index m) {
        for (Index i = 0; i < m; ++i) rows.push_back(q);
    };
    auto move = [&](const VectorXd& a, const VectorXd& b) {
        for (Index i = 0; i < nT; ++i) {
            const double r = static_cast<double>(i) / static_cast<double>(nT);
            const double s = 10 * r * r * r - 15 * r * r * r * r + 6 * r * r * r * r * r;
            rows.push_back((a + s * (b - a)).eval());
        }
    };
    hold(q0, nd);
    move(q0, q1);
    if (back) {
        hold(q1, 2 * nd);
        move(q1, q0);
        hold(q0, nd + 1);
    } else {
        hold(q1, 2 * nd + 1);
    }
    const Index N = static_cast<Index>(rows.size());
    VectorXd t(N);
    MatrixXd cmd(N, q0.size());
    for (Index k = 0; k < N; ++k) {
        t(k) = static_cast<double>(k) * dt;
        cmd.row(k) = rows[static_cast<std::size_t>(k)].transpose();
    }
    return {t, cmd};
}

double normalised_rms(const Eigen::VectorXd& predicted, const Eigen::VectorXd& measured) {
    if (predicted.size() != measured.size()) throw std::invalid_argument("normalised_rms: sizes differ");
    return std::sqrt((predicted - measured).array().square().mean()) / std::sqrt(measured.array().square().mean());
}

// ------------------------------------------------------------------ Lab 5 PID code (unchanged)
double plant_gravity(const Plant& plant, double q) { return plant.g_cos * std::cos(q) + plant.g_sin * std::sin(q); }

Eigen::Vector2d plant_rhs(const Plant& plant, const Eigen::Vector2d& x, double tau, bool gravity) {
    const double q = x(0), dq = x(1);
    const double g = gravity ? plant_gravity(plant, q) : 0.0;
    return Eigen::Vector2d(dq, (tau - plant.b * dq - g) / plant.M);
}

Eigen::Vector2d rk4_step(const StateFunction& f, const Eigen::Vector2d& x, double h) {
    const Eigen::Vector2d k1 = f(x);
    const Eigen::Vector2d k2 = f(x + 0.5 * h * k1);
    const Eigen::Vector2d k3 = f(x + 0.5 * h * k2);
    const Eigen::Vector2d k4 = f(x + h * k3);
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
}

std::pair<double, double> pd_gains(double M, double b, double zeta, double wn) {
    return {M * wn * wn, 2.0 * zeta * wn * M - b};
}

double pd_gravity_error(const Plant& plant, double kp, double qd) {
    double e = plant_gravity(plant, qd) / kp;
    for (int it = 0; it < 50; ++it) {
        const double q = qd - e;
        const double f = kp * e - plant_gravity(plant, q);
        const double fp = kp + (-plant.g_cos * std::sin(q) + plant.g_sin * std::cos(q));
        const double step = f / fp;
        e -= step;
        if (std::abs(step) < 1e-14) break;
    }
    return e;
}

JointPID::JointPID(double kp0, double ki0, double kd0, double dt0, std::optional<double> tau_max0, bool anti_windup0)
    : kp(kp0), ki(ki0), kd(kd0), dt(dt0), tau_max(tau_max0), anti_windup(anti_windup0) {
    reset();
}

void JointPID::reset(std::optional<double> q0) {
    eint = 0.0;
    q_prev = q0;
    terms = PidTerms{};
    saturated = false;
}

double JointPID::update(double qd, double qd_dot, double q, std::optional<double> qdot, double tau_ff) {
    const double qdot_used = qdot ? *qdot : (q_prev ? (q - *q_prev) / dt : 0.0);
    q_prev = q;
    const double e = qd - q;
    const double edot = qd_dot - qdot_used;
    double eint_new = eint + e * dt;
    double v = kp * e + kd * edot + ki * eint_new + tau_ff;
    double tau = v;
    saturated = false;
    if (tau_max && std::abs(v) > *tau_max) {
        saturated = true;
        tau = std::copysign(*tau_max, v);
        if (anti_windup && e * v > 0) {
            eint_new = eint;
            v = kp * e + kd * edot + ki * eint_new + tau_ff;
            tau = std::clamp(v, -*tau_max, *tau_max);
        }
    }
    eint = eint_new;
    terms = PidTerms{kp * e, ki * eint, kd * edot};
    return tau;
}

JointSim simulate_joint(const Plant& plant, JointPID& ctrl, const Eigen::VectorXd& t, const Eigen::VectorXd& qd,
                        const Eigen::VectorXd& qd_dot, double q0, double dq0, int substeps,
                        const std::string& velocity, std::optional<double> enc_res, const Eigen::VectorXd& tau_ff,
                        bool gravity) {
    const Index n = t.size();
    if (n < 2 || qd.size() != n || qd_dot.size() != n)
        throw std::invalid_argument("simulate_joint: t, qd and qd_dot must have the same length (>= 2)");
    const double dt = t(1) - t(0);
    if (std::abs(dt - ctrl.dt) > 1e-12) throw std::invalid_argument("controller dt must equal the time-grid spacing");
    ctrl.reset(q0);
    Eigen::Vector2d x(q0, dq0);
    JointSim r{t, VectorXd(n), VectorXd(n), VectorXd(n), VectorXd(n)};
    const double h = dt / substeps;
    const std::optional<double> lim = plant.tau_max;
    for (Index k = 0; k < n; ++k) {
        r.q(k) = x(0);
        r.dq(k) = x(1);
        const double qm = enc_res ? *enc_res * std::nearbyint(x(0) / *enc_res) : x(0);
        const std::optional<double> vm = velocity == "measured" ? std::optional<double>(x(1)) : std::optional<double>();
        const double ff = tau_ff.size() == 0 ? 0.0 : tau_ff(k);
        double tau = ctrl.update(qd(k), qd_dot(k), qm, vm, ff);
        if (lim) tau = std::clamp(tau, -*lim, *lim);
        r.tau(k) = tau;
        if (k + 1 < n) {
            const StateFunction f = [&plant, tau, gravity](const Eigen::Vector2d& s) {
                return plant_rhs(plant, s, tau, gravity);
            };
            for (int i = 0; i < substeps; ++i) x = rk4_step(f, x, h);
        }
    }
    r.e = qd - r.q;
    return r;
}

// ------------------------------------------------------------------ course-arm joint plant, servo model, replay
Plant joint_plant(const Model& model, const Eigen::VectorXd& q_lock, int joint, double M_eff, double b,
                  std::optional<double> tau_max) {
    VectorXd q = q_lock;
    q(joint) = 0.0;
    const double gc = gravity_torques(model, q)(joint);
    q(joint) = PI / 2;
    const double gs = gravity_torques(model, q)(joint);
    Plant p;
    p.name = "course_arm_j" + std::to_string(joint + 1);
    p.M = M_eff;
    p.b = b;
    p.g_cos = gc;
    p.g_sin = gs;
    p.tau_max = tau_max;
    p.locked_q = q_lock;
    p.joint_index = joint + 1;
    return p;
}

Servo nominal_servo(const Model& model, const Eigen::VectorXd& q, const sbeg::json& task) {
    const sbeg::json& ns = task.at("nominal_servo");
    Servo s;
    s.M_eff = mass_diagonal(model, q) + sbeg::as_vec(ns.at("armature_kgm2"));
    const Index n = s.M_eff.size();
    s.b = VectorXd::Constant(n, ns.at("b").get<double>());
    s.kp.resize(n);
    s.kd.resize(n);
    for (Index j = 0; j < n; ++j) {
        const std::pair<double, double> g = pd_gains(s.M_eff(j), s.b(j), ns.at("zeta").get<double>(), ns.at("wn").get<double>());
        s.kp(j) = g.first;
        s.kd(j) = g.second;
    }
    s.delay = ns.at("delay_s").get<double>();
    return s;
}

Servo make_servo(const Servo& base, const std::map<int, JointServo>& fits) {
    Servo s = base;
    double dsum = 0.0;
    int dn = 0;
    for (const auto& kv : fits) {
        const int j = kv.first;
        s.kp(j) = kv.second.kp;
        s.kd(j) = kv.second.kd;
        s.M_eff(j) = kv.second.M_eff;
        if (kv.second.b) s.b(j) = *kv.second.b;
        if (kv.second.delay) {
            dsum += *kv.second.delay;
            ++dn;
        }
    }
    if (dn > 0) s.delay = dsum / dn;
    return s;
}

ServoArm::ServoArm(const Model& model, const Servo& servo, std::vector<int> joints, double dt, double h, double enc_res,
                   const Eigen::VectorXd& tau_limit, const Eigen::VectorXd& kt)
    : model_(&model), servo_(servo), joints_(std::move(joints)), h_(h), enc_res_(enc_res),
      nsub_(static_cast<int>(std::nearbyint(dt / h))), tau_limit_(tau_limit), kt_(kt) {
    reset(VectorXd::Zero(servo.kp.size()));
}

void ServoArm::reset(const Eigen::VectorXd& q_cmd0) {
    const VectorXd c = q_cmd0;
    q = c;
    for (int it = 0; it < 60; ++it) {  // static sag: Kp (c - q) = g(q)
        const VectorXd g = gravity_torques(*model_, q);
        for (int j : joints_) q(j) = c(j) - g(j) / servo_.kp(j);
    }
    qd = VectorXd::Zero(q.size());
    t = 0.0;
    queue_.clear();
    queue_.emplace_back(-std::numeric_limits<double>::infinity(), c);
    tau = gravity_torques(*model_, q);
    pids_.clear();
    for (int j : joints_) {
        pids_.emplace_back(servo_.kp(j), 0.0, servo_.kd(j), h_);
        pids_.back().reset(q(j));
    }
    peak_qd = VectorXd::Zero(q.size());
    peak_tau = tau.cwiseAbs();
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ServoArm::read() {
    VectorXd qm = q;
    if (enc_res_ != 0.0) qm = (q / enc_res_).unaryExpr([](double x) { return std::nearbyint(x); }) * enc_res_;
    VectorXd out = tau;
    if (kt_.size() > 0) out = tau.cwiseQuotient(kt_);
    return {qm, out};
}

void ServoArm::send(const Eigen::VectorXd& q_cmd) { queue_.emplace_back(t + servo_.delay, q_cmd); }

Eigen::VectorXd ServoArm::active() const {
    VectorXd c = queue_.front().second;
    for (const auto& e : queue_) {
        if (e.first <= t + 1e-9) c = e.second;
        else break;
    }
    return c;
}

void ServoArm::wait() {
    const VectorXd g = gravity_torques(*model_, q);
    const Index n = q.size();
    std::vector<bool> moving(static_cast<std::size_t>(n), false);
    for (int j : joints_) moving[static_cast<std::size_t>(j)] = true;
    for (int s = 0; s < nsub_; ++s) {
        const VectorXd c = active();
        VectorXd u = g;  // held joints carry their gravity torque
        for (std::size_t i = 0; i < joints_.size(); ++i) {
            const int j = joints_[i];
            double v = pids_[i].update(c(j), 0.0, q(j), qd(j));
            if (tau_limit_.size() > 0) v = std::clamp(v, -tau_limit_(j), tau_limit_(j));
            u(j) = v;
        }
        for (Index j = 0; j < n; ++j)
            if (!moving[static_cast<std::size_t>(j)]) q(j) = c(j);
        for (int j : joints_) {
            const double qdd = (u(j) - servo_.b(j) * qd(j) - g(j)) / servo_.M_eff(j);
            qd(j) = qd(j) + h_ * qdd;
        }
        for (int j : joints_) q(j) = q(j) + h_ * qd(j);
        t += h_;
        tau = u;
        peak_qd = peak_qd.cwiseMax(qd.cwiseAbs());
        peak_tau = peak_tau.cwiseMax(u.cwiseAbs());
    }
    while (queue_.size() > 1 && queue_[1].first <= t + 1e-9) queue_.erase(queue_.begin());
}

ServoRun ServoArm::run(const Eigen::VectorXd& tt, const Eigen::MatrixXd& cmd) {
    const Index N = tt.size(), n = cmd.cols();
    if (cmd.rows() != N) throw std::invalid_argument("ServoArm::run: cmd must have one row per sample");
    reset(cmd.row(0).transpose());
    ServoRun r{MatrixXd(N, n), MatrixXd(N, n), MatrixXd(N, n), MatrixXd(N, n), VectorXd(), VectorXd()};
    for (Index k = 0; k < N; ++k) {
        r.meas.row(k) = read().first.transpose();
        r.q.row(k) = q.transpose();
        r.qd.row(k) = qd.transpose();
        r.tau.row(k) = tau.transpose();
        send(cmd.row(k).transpose());
        if (k + 1 < N) wait();
    }
    r.peak_qd = peak_qd;
    r.peak_tau = peak_tau;
    return r;
}

ReplayIO::ReplayIO(const Log& log) : meas_(log.meas), cur_(log.cur) {
    if (meas_.rows() == 0) throw std::invalid_argument("ReplayIO: the log has no meas columns");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ReplayIO::read() {
    const VectorXd c = cur_.size() > 0 ? VectorXd(cur_.row(k_).transpose()) : VectorXd();
    return {meas_.row(k_).transpose(), c};
}

void ReplayIO::send(const Eigen::VectorXd& q_cmd) { sent.push_back(q_cmd); }

void ReplayIO::wait() { k_ = std::min(k_ + 1, meas_.rows() - 1); }

}  // namespace lab
