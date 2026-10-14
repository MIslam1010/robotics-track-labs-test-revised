// SBEG 46X Lab 10 -- provided functions (C++ track). Given code: do not edit; use these functions freely.
//
// The "Provided" section of starter.py, translated to C++ operation by operation: load_params, gravity,
// step_setpoint, therapy_reference, smooth_pulse, patient_torque, operator_target, saturate_toward_limit,
// virtual_wall_force, solve_qp (small exact QP solver), end_stop_torque, pd_torque, simulate_joint, simulate_hapkit.
// The two simulators call YOUR functions rom_torque_bounds, joint_safety_filter, aan_torque and hapkit_safety_filter.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;

std::pair<double, double> pair_of(const sbeg::json& a) {
    if (!a.is_array() || a.size() != 2) throw std::runtime_error("expected a pair [a, b] in lab10.json: " + a.dump());
    return {a[0].get<double>(), a[1].get<double>()};
}

JointParams joint_from_json(const sbeg::json& d) {
    JointParams j;
    j.model = d.at("model").get<std::string>();
    j.M = d.at("M").get<double>();
    j.b = d.at("b").get<double>();
    j.g_s = d.at("g_s").get<double>();
    j.g_c = d.at("g_c").get<double>();
    j.tau_rated = d.at("tau_rated").get<double>();
    j.tau_max = d.at("tau_max").get<double>();
    j.tau_rate = d.contains("tau_rate") ? d.at("tau_rate").get<double>() : 0.0;
    j.mech_limits = pair_of(d.at("mech_limits"));
    j.rom = pair_of(d.at("rom"));
    j.dt = d.at("dt").get<double>();
    j.n_sub = d.at("n_sub").get<int>();
    j.cbf.alpha1 = d.at("cbf").at("alpha1").get<double>();
    j.cbf.alpha2 = d.at("cbf").at("alpha2").get<double>();
    j.stop.K = d.at("stop").at("K").get<double>();
    j.stop.B = d.at("stop").at("B").get<double>();
    const sbeg::json& s = d.at("step");
    j.step.theta0 = s.at("theta0").get<double>();
    j.step.T = s.at("T").get<double>();
    for (const auto& e : s.at("steps")) j.step.steps.push_back(pair_of(e));
    j.step.pd.wn = s.at("pd").at("wn").get<double>();
    j.step.pd.zeta = s.at("pd").at("zeta").get<double>();
    j.has_spasm = d.contains("spasm");
    if (j.has_spasm) {
        const sbeg::json& p = d.at("spasm");
        SpasmTask& t = j.spasm;
        t.theta0 = p.at("theta0").get<double>();
        t.T = p.at("T").get<double>();
        t.ref.mid = p.at("ref").at("mid").get<double>();
        t.ref.amp = p.at("ref").at("amp").get<double>();
        t.ref.period = p.at("ref").at("period").get<double>();
        t.aan.K = p.at("aan").at("K").get<double>();
        t.aan.B = p.at("aan").at("B").get<double>();
        t.aan.delta = p.at("aan").at("delta").get<double>();
        t.patient.c = p.at("patient").at("c").get<double>();
        t.patient.Kp = p.at("patient").at("Kp").get<double>();
        t.patient.Bp = p.at("patient").at("Bp").get<double>();
        t.spasm.amp = p.at("spasm").at("amp").get<double>();
        t.spasm.t_on = p.at("spasm").at("t_on").get<double>();
        t.spasm.t_off = p.at("spasm").at("t_off").get<double>();
        t.spasm.ramp = p.at("spasm").at("ramp").get<double>();
        t.tau_h_max = p.at("tau_h_max").get<double>();
    }
    return j;
}

HapkitParams hapkit_from_json(const sbeg::json& d) {
    HapkitParams h;
    h.model = d.at("model").get<std::string>();
    h.m_h = d.at("m_h").get<double>();
    h.b_h = d.at("b_h").get<double>();
    h.k_g = d.at("k_g").get<double>();
    h.T = d.at("T").get<double>();
    h.x_res = d.at("x_res").get<double>();
    h.v_filter_hz = d.at("v_filter_hz").get<double>();
    h.force_max = d.at("force_max").get<double>();
    h.x_range = d.at("x_range").get<double>();
    h.x_max = d.at("x_max").get<double>();
    h.cbf.alpha1 = d.at("cbf").at("alpha1").get<double>();
    h.cbf.alpha2 = d.at("cbf").at("alpha2").get<double>();
    h.wall.K = d.at("wall").at("K").get<double>();
    h.wall.B = d.at("wall").at("B").get<double>();
    const sbeg::json& o = d.at("operator");
    h.op.m = o.at("m").get<double>();
    h.op.b = o.at("b").get<double>();
    h.op.k = o.at("k").get<double>();
    h.op.x0 = o.at("x0").get<double>();
    h.op.t0 = o.at("t0").get<double>();
    h.op.speed = o.at("speed").get<double>();
    for (const char* name : {"push", "hard_push"}) {
        const sbeg::json& c = d.at(name);
        HapkitCase hc;
        hc.x_int = c.at("x_int").get<double>();
        hc.T = c.at("T").get<double>();
        hc.F_h_max = c.at("F_h_max").get<double>();
        h.cases[name] = hc;
    }
    return h;
}
}  // namespace

// ------------------------------------------------------------------------------------------ parameters, models
Params load_params(const std::string& path) {
    const sbeg::json d = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/lab10.json") : path);
    Params P;
    P.elbow = joint_from_json(d.at("elbow"));
    P.knee = joint_from_json(d.at("knee"));
    P.hapkit = hapkit_from_json(d.at("hapkit"));
    return P;
}

double gravity(double theta, const JointParams& j) { return j.g_s * std::sin(theta) + j.g_c * std::cos(theta); }

double step_setpoint(double t, const StepTask& task) {
    if (task.steps.empty()) throw std::invalid_argument("step_setpoint: no set-points");
    double th = task.steps.front().second;
    for (const auto& s : task.steps)
        if (t >= s.first - 1e-12) th = s.second;
    return th;
}

std::pair<double, double> therapy_reference(double t, const SpasmTask::Ref& ref) {
    const double w = 2 * PI / ref.period;
    return {ref.mid - ref.amp * std::cos(w * t), ref.amp * w * std::sin(w * t)};
}

double smooth_pulse(double t, double amp, double t_on, double t_off, double ramp) {
    if (t <= t_on || t >= t_off + ramp) return 0.0;
    if (t < t_on + ramp) return amp * 0.5 * (1 - std::cos(PI * (t - t_on) / ramp));
    if (t <= t_off) return amp;
    return amp * 0.5 * (1 + std::cos(PI * (t - t_off) / ramp));
}

double patient_torque(double t, double theta, double omega, const SpasmTask& task) {
    const auto& p = task.patient;
    const auto& sp = task.spasm;
    const auto [th_d, om_d] = therapy_reference(t, task.ref);
    const double vol = p.c * (p.Kp * (th_d - theta) + p.Bp * (om_d - omega));
    return vol + smooth_pulse(t, sp.amp, sp.t_on, sp.t_off, sp.ramp);
}

double operator_target(double t, const HapkitParams::Operator& op, double x_int) {
    if (t <= op.t0) return op.x0;
    return std::min(x_int, op.x0 + op.speed * (t - op.t0));
}

// ------------------------------------------------------------------------------------------ baselines
double saturate_toward_limit(double x, double u, double x_min, double x_max) {
    if ((x >= x_max && u > 0) || (x <= x_min && u < 0)) return 0.0;
    return u;
}

double virtual_wall_force(double x, double v, double x_max, double K, double B) {
    if (x > x_max) return -K * (x - x_max) - B * std::max(v, 0.0);
    return 0.0;
}

// ------------------------------------------------------------------------------------------ small QP solver
std::pair<Eigen::VectorXd, QPInfo> solve_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                                            const Eigen::MatrixXd& A_ub, const Eigen::VectorXd& b_ub,
                                            const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                            const Eigen::VectorXd& x0) {
    const Eigen::Index n = f.size();
    if (n == 0 || H.rows() != n || H.cols() != n) throw std::invalid_argument("solve_qp: H must be n x n, f of length n");
    if (A_ub.size() > 0 && (A_ub.cols() != n || A_ub.rows() != b_ub.size()))
        throw std::invalid_argument("solve_qp: A_ub must be m x n and b_ub of length m");
    if ((lb.size() != 0 && lb.size() != n) || (ub.size() != 0 && ub.size() != n))
        throw std::invalid_argument("solve_qp: lb and ub must be empty or of length n");
    // all inequalities as rows  c_i' z <= d_i
    std::vector<Eigen::VectorXd> C;
    std::vector<double> d;
    for (Eigen::Index i = 0; i < A_ub.rows(); ++i) {
        C.push_back(A_ub.row(i).transpose());
        d.push_back(b_ub(i));
    }
    for (Eigen::Index i = 0; i < n; ++i) {
        if (ub.size() == n && std::isfinite(ub(i))) {
            C.push_back(Eigen::VectorXd::Unit(n, i));
            d.push_back(ub(i));
        }
        if (lb.size() == n && std::isfinite(lb(i))) {
            C.push_back(-Eigen::VectorXd::Unit(n, i));
            d.push_back(-lb(i));
        }
    }
    const int m = static_cast<int>(C.size());
    if (n > 8 || m > 20) throw std::invalid_argument("solve_qp: problem too large for the lab solver (n <= 8, m <= 20)");
    QPInfo info;
    Eigen::VectorXd best = Eigen::VectorXd::Zero(n);
    double best_cost = INF;
    // enumerate the active sets of size 0 ... min(n, m) (bit masks); keep the KKT point of lowest cost
    const int kmax = static_cast<int>(std::min<Eigen::Index>(n, m));
    for (unsigned mask = 0; mask < (1u << m); ++mask) {
        std::vector<int> act;
        for (int i = 0; i < m; ++i)
            if (mask & (1u << i)) act.push_back(i);
        if (static_cast<int>(act.size()) > kmax) continue;
        ++info.nit;
        const Eigen::Index k = static_cast<Eigen::Index>(act.size());
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n + k, n + k);
        Eigen::VectorXd r(n + k);
        K.topLeftCorner(n, n) = H;
        r.head(n) = -f;
        for (Eigen::Index a = 0; a < k; ++a) {
            const Eigen::VectorXd& c = C[static_cast<std::size_t>(act[static_cast<std::size_t>(a)])];
            K.block(0, n + a, n, 1) = c;
            K.block(n + a, 0, 1, n) = c.transpose();
            r(n + a) = d[static_cast<std::size_t>(act[static_cast<std::size_t>(a)])];
        }
        Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
        if (!lu.isInvertible()) continue;
        const Eigen::VectorXd sol = lu.solve(r);
        const Eigen::VectorXd z = sol.head(n);
        if (k > 0 && sol.tail(k).minCoeff() < -1e-10) continue;  // multipliers of active rows must be >= 0
        bool feasible = true;
        for (int i = 0; i < m && feasible; ++i)
            feasible = C[static_cast<std::size_t>(i)].dot(z) <= d[static_cast<std::size_t>(i)] + 1e-9;
        if (!feasible) continue;
        const double cost = 0.5 * z.dot(H * z) + f.dot(z);
        if (cost < best_cost) {
            best_cost = cost;
            best = z;
            info.success = true;
        }
    }
    info.fun = info.success ? best_cost : 0.0;
    if (!info.success && x0.size() == n) best = x0;
    return {best, info};
}

// ------------------------------------------------------------------------------------------ joint simulator
double end_stop_torque(double theta, double omega, const JointParams& j) {
    const double lo = j.mech_limits.first, hi = j.mech_limits.second;
    const double K = j.stop.K, B = j.stop.B;
    if (theta > hi) return -K * (theta - hi) - B * std::max(omega, 0.0);
    if (theta < lo) return -K * (theta - lo) - B * std::min(omega, 0.0);
    return 0.0;
}

double pd_torque(double theta, double omega, double th_d, const JointParams& j, const PdGains& pd) {
    const double Kp = j.M * (pd.wn * pd.wn);
    const double Kd = 2 * pd.zeta * pd.wn * j.M - j.b;
    return gravity(theta, j) + Kp * (th_d - theta) - Kd * omega;
}

JointLog simulate_joint(const JointParams& j, const std::string& experiment, const std::string& mode,
                        const std::string& tau_h_mode, std::optional<std::pair<double, double>> alphas) {
    const bool step = experiment == "step";
    if (!step && !(experiment == "spasm" && j.has_spasm))
        throw std::invalid_argument("simulate_joint: unknown experiment for this joint: " + experiment);
    if (mode != "none" && mode != "sat" && mode != "cbf") throw std::invalid_argument("simulate_joint: mode " + mode);
    if (tau_h_mode != "measured" && tau_h_mode != "zero" && tau_h_mode != "robust")
        throw std::invalid_argument("simulate_joint: tau_h_mode " + tau_h_mode);
    const double dt = j.dt;
    const int n_sub = j.n_sub;
    const double a1 = alphas ? alphas->first : j.cbf.alpha1;
    const double a2 = alphas ? alphas->second : j.cbf.alpha2;
    const double lo = j.rom.first, hi = j.rom.second;
    double th = step ? j.step.theta0 : j.spasm.theta0;
    double om = 0.0;
    double tau_prev = gravity(th, j);
    const double T = step ? j.step.T : j.spasm.T;
    const Eigen::Index n = static_cast<Eigen::Index>(std::nearbyint(T / dt));
    JointLog log;
    for (Eigen::VectorXd* v : {&log.t, &log.theta, &log.omega, &log.tau, &log.tau_nom, &log.tau_h, &log.th_d})
        *v = Eigen::VectorXd::Zero(n);
    log.active.assign(static_cast<std::size_t>(n), false);
    log.feasible.assign(static_cast<std::size_t>(n), true);
    for (Eigen::Index k = 0; k < n; ++k) {
        const double t = static_cast<double>(k) * dt;
        double th_d = 0.0, tau_nom = 0.0, tau_h_meas = 0.0, tau_h_bound = 0.0;
        if (step) {
            th_d = step_setpoint(t, j.step);
            if (mode == "sat") th_d = std::min(std::max(th_d, lo), hi);
            tau_nom = pd_torque(th, om, th_d, j, j.step.pd);
        } else {
            double om_d = 0.0;
            std::tie(th_d, om_d) = therapy_reference(t, j.spasm.ref);
            if (mode == "sat") th_d = std::min(std::max(th_d, lo), hi);
            const auto& a = j.spasm.aan;
            tau_nom = gravity(th, j) + aan_torque(th - th_d, om - om_d, a.K, a.B, a.delta);
            tau_h_meas = patient_torque(t, th, om, j.spasm);
            tau_h_bound = j.spasm.tau_h_max;
        }
        double act_lo = -j.tau_max, act_hi = j.tau_max;
        if (j.tau_rate != 0.0) {
            act_lo = std::max(act_lo, tau_prev - j.tau_rate * dt);
            act_hi = std::min(act_hi, tau_prev + j.tau_rate * dt);
        }
        bool active = false, feasible = true;
        double tau = 0.0;
        if (mode == "cbf") {
            std::pair<double, double> th_lo_hi;
            if (tau_h_mode == "measured") {
                th_lo_hi = rom_torque_bounds(th, om, j, a1, a2, tau_h_meas);
            } else if (tau_h_mode == "zero") {
                th_lo_hi = rom_torque_bounds(th, om, j, a1, a2, 0.0);
            } else {  // worst case for each barrier: the upper one with +bound, the lower one with -bound
                const double up = rom_torque_bounds(th, om, j, a1, a2, tau_h_bound).second;
                const double dn = rom_torque_bounds(th, om, j, a1, a2, -tau_h_bound).first;
                th_lo_hi = {dn, up};
            }
            const auto [tau_f, info] = joint_safety_filter(tau_nom, th_lo_hi, {act_lo, act_hi});
            tau = tau_f;
            active = info.active;
            feasible = info.feasible;
        } else {
            tau = tau_nom;
            if (mode == "sat") {
                const double g_hat = gravity(th, j);
                tau = g_hat + saturate_toward_limit(th, tau - g_hat, lo, hi);
            }
            tau = std::min(std::max(tau, act_lo), act_hi);
        }
        for (int s = 0; s < n_sub; ++s) {  // plant (true patient torque evaluated at every sub-step)
            const double tau_h = step ? 0.0 : patient_torque(t, th, om, j.spasm);
            const double acc = (tau + tau_h - j.b * om - gravity(th, j) + end_stop_torque(th, om, j)) / j.M;
            om += acc * dt / n_sub;
            th += om * dt / n_sub;
        }
        tau_prev = tau;
        log.t(k) = t + dt;
        log.theta(k) = th;
        log.omega(k) = om;
        log.tau(k) = tau;
        log.tau_nom(k) = tau_nom;
        log.tau_h(k) = tau_h_meas;
        log.th_d(k) = th_d;
        log.active[static_cast<std::size_t>(k)] = active;
        log.feasible[static_cast<std::size_t>(k)] = feasible;
    }
    return log;
}

// ------------------------------------------------------------------------------------------ Hapkit simulator
HapkitLog simulate_hapkit(const HapkitParams& hk, const std::string& hk_case, const std::string& mode,
                          std::optional<std::pair<double, double>> alphas) {
    const auto it = hk.cases.find(hk_case);
    if (it == hk.cases.end()) throw std::invalid_argument("simulate_hapkit: unknown case " + hk_case);
    if (mode != "none" && mode != "wall" && mode != "cbf") throw std::invalid_argument("simulate_hapkit: mode " + mode);
    const HapkitCase& c = it->second;
    const auto& op = hk.op;
    const double a1 = alphas ? alphas->first : hk.cbf.alpha1;
    const double a2 = alphas ? alphas->second : hk.cbf.alpha2;
    const double T = hk.T;
    const int n_sub = 4;
    double x = 0.0, v = 0.0;
    double x_meas_prev = 0.0;
    double v_est = 0.0;
    const double beta = std::exp(-2 * PI * hk.v_filter_hz * T);
    bool tripped = false;
    const Eigen::Index n = static_cast<Eigen::Index>(std::nearbyint(c.T / T));
    HapkitLog log;
    for (Eigen::VectorXd* e : {&log.t, &log.x, &log.v, &log.F, &log.F_h}) *e = Eigen::VectorXd::Zero(n);
    log.active.assign(static_cast<std::size_t>(n), false);
    log.feasible.assign(static_cast<std::size_t>(n), true);
    log.tripped.assign(static_cast<std::size_t>(n), false);
    const double m_tot = hk.m_h + op.m;
    for (Eigen::Index k = 0; k < n; ++k) {
        const double t = static_cast<double>(k) * T;
        const double x_meas = std::nearbyint(x / hk.x_res) * hk.x_res;  // Python round(): halves to even
        const double v_raw = k > 0 ? (x_meas - x_meas_prev) / T : 0.0;
        v_est = beta * v_est + (1 - beta) * v_raw;
        x_meas_prev = x_meas;
        bool active = false, feasible = true;
        double F = 0.0;
        if (mode == "none") {
            F = 0.0;
        } else if (mode == "wall") {
            F = virtual_wall_force(x_meas, v_est, hk.x_max, hk.wall.K, hk.wall.B);
        } else {
            const auto [F_f, info] = hapkit_safety_filter(0.0, x_meas, v_est, hk, a1, a2, c.F_h_max);
            F = F_f;
            active = info.active;
            feasible = info.feasible;
        }
        F = std::min(std::max(F, -hk.force_max), hk.force_max);
        if (std::abs(x_meas) > hk.x_range) tripped = true;
        if (tripped) F = 0.0;
        const double x_tgt = operator_target(t, op, c.x_int);
        double F_h = 0.0;
        for (int s = 0; s < n_sub; ++s) {
            F_h = op.k * (x_tgt - x) - op.b * v;
            const double acc = (F - hk.b_h * v - hk.k_g * x + F_h) / m_tot;
            v += acc * T / n_sub;
            x += v * T / n_sub;
        }
        log.t(k) = t + T;
        log.x(k) = x;
        log.v(k) = v;
        log.F(k) = F;
        log.F_h(k) = F_h;
        log.active[static_cast<std::size_t>(k)] = active;
        log.feasible[static_cast<std::size_t>(k)] = feasible;
        log.tripped[static_cast<std::size_t>(k)] = tripped;
    }
    return log;
}

}  // namespace lab
