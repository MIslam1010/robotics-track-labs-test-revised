// SBEG 46X Lab 3 -- Delayed bilateral teleoperation: instability, wave variables, time-domain passivity
// (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp, with the full documentation).
// The 'Provided' section (parameters, hand intent, DelayLine, network_delays, contact_force, simulate) is given:
// do not change it. simulate() calls YOUR follower_pd, wave_leader, wave_follower, tdpa_leader and tdpa_follower,
// so the simulation tests fail until those are written.
// Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 3 --cpp
//
// C++ reminders:  std::sqrt, std::abs, std::max; complex numbers: std::complex<double> z(re, im), std::arg(z)
//   (= np.angle), std::tanh(z), std::abs(z); a pair: return {a, b};  a struct: return {F_m, E_in_m, E_out_m, r};
//   infinity / NaN: std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN().
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace lab {

namespace {
[[maybe_unused]] constexpr double PI = 3.14159265358979323846;  // use PI, never M_PI (not standard C++)
}  // namespace

// ==========================================================================================
// Provided (do not edit): parameters, hand intent, network model, simulator
// ==========================================================================================

Params load_params(const std::string& path) {
    const sbeg::json j = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/teleop.json") : path);
    Params p;
    p.dt = j.at("dt").get<double>();
    p.substeps = j.at("substeps").get<int>();
    p.force_max = j.at("force_max").get<double>();
    p.x_range = j.at("x_range").get<double>();
    p.leader = Device{j.at("leader").at("m").get<double>(), j.at("leader").at("b").get<double>()};
    p.follower = Device{j.at("follower").at("m").get<double>(), j.at("follower").at("b").get<double>()};
    const sbeg::json& h = j.at("hand");
    p.hand = Hand{h.at("m").get<double>(), h.at("b").get<double>(), h.at("k").get<double>()};
    p.controller = Controller{j.at("controller").at("Kp").get<double>(), j.at("controller").at("Kd").get<double>()};
    for (const auto& item : j.at("environments").items()) {
        const sbeg::json& e = item.value();
        p.environments[item.key()] =
            Environment{e.at("k").get<double>(), e.at("b").get<double>(), e.at("x_wall").get<double>()};
    }
    const sbeg::json& rt = j.at("reach_task");
    p.reach_task = ReachTask{rt.at("x_start").get<double>(), rt.at("x_goal").get<double>(),
                             rt.at("t_move").get<double>(), rt.at("t_end").get<double>()};
    p.wave = WaveParams{j.at("wave").at("b").get<double>(), j.at("wave").at("tau_f").get<double>()};
    return p;
}

Eigen::VectorXd minjerk(const Eigen::VectorXd& t, double t0, double t1, double x0, double x1) {
    Eigen::VectorXd x(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        const double s = std::clamp((t(i) - t0) / (t1 - t0), 0.0, 1.0);
        x(i) = x0 + (x1 - x0) * (10 * std::pow(s, 3.0) - 15 * std::pow(s, 4.0) + 6 * std::pow(s, 5.0));
    }
    return x;
}

Eigen::VectorXd reach_intent(const Params& p, const Eigen::VectorXd& t) {
    const ReachTask& rt = p.reach_task;
    return minjerk(t, 0.0, rt.t_move, rt.x_start, rt.x_goal);
}

DelayLine::DelayLine(const std::vector<int>& delays, int n_steps, const Eigen::VectorXd& initial)
    : d_(delays), initial_(initial) {
    if (static_cast<long long>(d_.size()) < static_cast<long long>(n_steps))
        throw std::invalid_argument("need one delay per sample");
    for (int di : d_)
        if (di == 0) throw std::invalid_argument("delays must be >= 1 sample (or < 0 for a lost packet)");
}

DelayLine::DelayLine(int delay, int n_steps, const Eigen::VectorXd& initial)
    : DelayLine(std::vector<int>(static_cast<std::size_t>(std::max(n_steps, 0)), delay), n_steps, initial) {}

void DelayLine::send(int n, const Eigen::VectorXd& value) {
    if (n < 0 || static_cast<std::size_t>(n) >= d_.size()) throw std::invalid_argument("DelayLine::send: bad sample");
    values_[n] = value;
    const int dn = d_[static_cast<std::size_t>(n)];
    if (dn >= 0) arrivals_[n + dn].push_back(n);
}

Eigen::VectorXd DelayLine::receive(int n) {
    const auto it = arrivals_.find(n);
    if (it != arrivals_.end()) {
        for (int s : it->second) best_ = std::max(best_, s);
        arrivals_.erase(it);
    }
    return best_ < 0 ? initial_ : values_.at(best_);
}

std::vector<int> network_delays(double T, double dt, int n_steps, double jitter, double loss, int seed) {
    const std::size_t n = static_cast<std::size_t>(std::max(n_steps, 0));
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    auto uniform01 = [&gen]() { return static_cast<double>(gen()) / 4294967296.0; };  // [0, 1), portable
    std::vector<double> base(n, T / dt);
    if (jitter > 0)
        for (double& bi : base) bi += -jitter / dt + 2.0 * jitter / dt * uniform01();
    std::vector<int> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = std::max(1, static_cast<int>(std::nearbyint(base[i])));
    if (loss > 0)
        for (int& di : d)
            if (uniform01() < loss) di = -1;
    return d;
}

double contact_force(double x, double xdot, const Environment& env) {
    const double pen = x - env.x_wall;
    if (pen <= 0.0 || env.k == 0.0) return 0.0;
    return std::max(0.0, env.k * pen + env.b * xdot);
}

SimResult simulate(const std::string& arch, double T, const Params& p, const SimOptions& opt) {
    if (arch != "pp" && arch != "pf" && arch != "wave" && arch != "tdpa")
        throw std::invalid_argument("arch must be one of pp, pf, wave, tdpa");
    const double dt = p.dt;
    const int nsub = p.substeps;
    const double hsub = dt / nsub;
    const Device &L = p.leader, &F = p.follower;
    const Hand& H = p.hand;
    const Controller& C = p.controller;
    Environment E;
    if (opt.env_custom) {
        E = *opt.env_custom;
    } else {
        const auto it = p.environments.find(opt.env);
        if (it == p.environments.end()) throw std::invalid_argument("unknown environment: " + opt.env);
        E = it->second;
    }
    const double b = opt.b_wave ? *opt.b_wave : p.wave.b;
    const double t_end = opt.t_end ? *opt.t_end : p.reach_task.t_end;
    const int n = static_cast<int>(std::nearbyint(t_end / dt));
    SimResult out;
    out.t.resize(n);
    for (int k = 0; k < n; ++k) out.t(k) = k * dt;
    out.x_i = reach_intent(p, out.t);
    Eigen::VectorXd v_i(n);  // np.gradient(x_i, dt)
    if (n >= 2) {
        v_i(0) = (out.x_i(1) - out.x_i(0)) / dt;
        v_i(n - 1) = (out.x_i(n - 1) - out.x_i(n - 2)) / dt;
        for (int k = 1; k < n - 1; ++k) v_i(k) = (out.x_i(k + 1) - out.x_i(k - 1)) / (2.0 * dt);
    } else if (n == 1) {
        v_i(0) = 0.0;
    }
    const double mL = L.m + H.m;
    const double fmax = p.force_max;
    auto sat = [&](double f) { return opt.saturate ? std::clamp(f, -fmax, fmax) : f; };
    const Eigen::VectorXd zero3 = Eigen::VectorXd::Zero(3);
    DelayLine fwd(opt.fwd_delays.empty() ? network_delays(T, dt, n, opt.jitter, opt.loss, opt.seed) : opt.fwd_delays,
                  n, zero3);  // x_m, u_m, E_in,m
    DelayLine bwd(opt.bwd_delays.empty() ? network_delays(T, dt, n, opt.jitter, opt.loss, opt.seed + 1)
                                         : opt.bwd_delays,
                  n, zero3);  // F_back, v_s, E_in,s
    for (Eigen::VectorXd* v : {&out.x_m, &out.xdot_m, &out.x_s, &out.xdot_s, &out.x_d, &out.F_m, &out.F_c, &out.f_e,
                               &out.P_m, &out.P_s, &out.E_ch, &out.W_m, &out.W_s, &out.r_m, &out.r_s})
        *v = Eigen::VectorXd::Zero(n);
    double xm = 0.0, vm = 0.0, xs = 0.0, vs = 0.0, xd = 0.0;
    double x_recv_prev = 0.0;
    double E_in_m = 0.0, E_out_m = 0.0, E_in_s = 0.0, E_out_s = 0.0;
    double v_f = 0.0;
    const double a_f = dt / (p.wave.tau_f + dt);
    bool tripped = false;
    for (int k = 0; k < n; ++k) {
        const Eigen::VectorXd rf = fwd.receive(k);  // arrives at the follower
        const Eigen::VectorXd rb = bwd.receive(k);  // arrives at the leader
        const double fe = contact_force(xs, vs, E);
        // ------------------------------------------------------------------ follower side
        const double F_c = sat(follower_pd(xd, xs, vs, C.Kp, C.Kd));
        double r_s = 1.0, v_back = 0.0, dx = 0.0;
        if (arch == "wave") {
            const auto [xdot_d, v_s] = wave_follower(rf(1), F_c, b);
            v_back = v_s;
            dx = xdot_d * dt;
        } else {
            dx = rf(0) - x_recv_prev;
            x_recv_prev = rf(0);
            if (arch == "tdpa") {
                const TdpaFollowerResult tf = tdpa_follower(F_c, dx, E_in_s, E_out_s, rf(2));
                dx = tf.dx_d;
                E_in_s = tf.E_in_s;
                E_out_s = tf.E_out_s;
                r_s = tf.r;
            }
        }
        const double P_s = F_c * dx / dt;
        if (arch != "tdpa") {
            E_in_s += std::max(-P_s, 0.0) * dt;
            E_out_s += std::max(P_s, 0.0) * dt;
        }
        // ------------------------------------------------------------------ leader side
        double r_m = 1.0, u_m = 0.0, F_m = 0.0;
        if (arch == "wave") {
            v_f += a_f * (rb(1) - v_f);  // passive first-order low-pass on the arriving wave
            const auto [Fw, uw] = wave_leader(vm, v_f, b);
            F_m = sat(Fw);
            u_m = uw;
        } else {
            F_m = sat(rb(0));
            if (arch == "tdpa") {
                const TdpaLeaderResult tl = tdpa_leader(F_m, vm, dt, E_in_m, E_out_m, rb(2));
                F_m = tl.F_m;
                E_in_m = tl.E_in_m;
                E_out_m = tl.E_out_m;
                r_m = tl.r;
            }
        }
        const double P_m = F_m * vm;
        if (arch != "tdpa") {
            E_in_m += std::max(P_m, 0.0) * dt;
            E_out_m += std::max(-P_m, 0.0) * dt;
        }
        // ------------------------------------------------------------------ network
        Eigen::VectorXd sf(3), sb(3);
        sf << xm, u_m, E_in_m;
        sb << (arch == "pf" ? fe : F_c), v_back, E_in_s;
        fwd.send(k, sf);
        bwd.send(k, sb);
        // ------------------------------------------------------------------ watchdog and log
        if (opt.watchdog && (tripped || std::max(std::abs(xm), std::abs(xs)) > p.x_range)) tripped = true;
        out.x_m(k) = xm;
        out.xdot_m(k) = vm;
        out.x_s(k) = xs;
        out.xdot_s(k) = vs;
        out.x_d(k) = xd;
        out.F_m(k) = F_m;
        out.F_c(k) = F_c;
        out.f_e(k) = fe;
        out.P_m(k) = P_m;
        out.P_s(k) = P_s;
        out.E_ch(k) = E_in_m + E_in_s - E_out_m - E_out_s;
        out.W_m(k) = rb(2) - E_out_m;
        out.W_s(k) = rf(2) - E_out_s;
        out.r_m(k) = r_m;
        out.r_s(k) = r_s;
        const double am_cmd = tripped ? 0.0 : F_m, as_cmd = tripped ? 0.0 : F_c;
        // ------------------------------------------------------------------ mechanics (zero-order hold)
        for (int i = 0; i < nsub; ++i) {
            const double fh = H.k * (out.x_i(k) - xm) + H.b * (v_i(k) - vm);
            const double fe_c = contact_force(xs, vs, E);
            vm += hsub * (fh - am_cmd - L.b * vm) / mL;
            xm += hsub * vm;
            vs += hsub * (as_cmd - F.b * vs - fe_c) / F.m;
            xs += hsub * vs;
        }
        xd += dx;
    }
    out.tripped = tripped;
    out.N = std::max(1, static_cast<int>(std::nearbyint(T / dt)));
    out.arch = arch;
    out.T = T;
    return out;
}

SimResult simulate(const std::string& arch, double T, const SimOptions& opt) {
    return simulate(arch, T, load_params(), opt);
}

// ==========================================================================================
// Part A -- why force reflection becomes unstable: the delay margin
// ==========================================================================================

double contact_stiffness(double Kp, double k_e) {
    // TODO A1: two springs in series.
    throw std::logic_error("TODO: contact_stiffness");
}

std::pair<double, double> delay_margin(double m, double b, double k, double kp) {
    // TODO A2: solve the quadratic m^2 W^2 + (b^2 - 2 k m) W + k^2 - kp^2 = 0 in W = w^2 (no np.roots in C++:
    //       quadratic formula; a negative discriminant means complex roots = no crossing), keep the real positive
    //       roots, get tau from the phase condition with std::arg and the Python-style modulo
    //       ((-angle) % (2 pi) is in [0, 2 pi): std::fmod, then add 2 PI if negative); return the smallest (tau, w).
    //       Handle m = 0 (one root, (kp^2 - k^2) / b^2, only if b > 0). No root: {infinity, quiet_NaN}.
    throw std::logic_error("TODO: delay_margin");
}

// ==========================================================================================
// Part B -- wave variables
// ==========================================================================================

std::pair<double, double> wave_encode(double F, double xdot, double b) {
    // TODO B1: two lines.
    throw std::logic_error("TODO: wave_encode");
}

std::pair<double, double> wave_leader(double xdot_m, double v_m, double b) {
    // TODO B2: solve the wave definitions for F_m given xdot_m and v_m.
    throw std::logic_error("TODO: wave_leader");
}

std::pair<double, double> wave_follower(double u_s, double F_c, double b) {
    // TODO B3: solve the wave definitions for xdot_d given u_s and F_c.
    throw std::logic_error("TODO: wave_follower");
}

std::complex<double> wave_apparent_impedance(double b, double T, double omega, const std::string& termination) {
    // TODO B4: use std::tanh with a complex argument std::complex<double>(0.0, omega * T).
    throw std::logic_error("TODO: wave_apparent_impedance");
}

// ==========================================================================================
// Part C -- energy bookkeeping and time-domain passivity
// ==========================================================================================

double follower_pd(double x_d, double x_s, double xdot_s, double Kp, double Kd) {
    // TODO C1: one line.
    throw std::logic_error("TODO: follower_pd");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> port_energies(const Eigen::VectorXd& P, double h) {
    // TODO C2: running (cumulative) sums of the positive and negative parts (np.cumsum): size E_in and E_out
    //       first (Eigen::VectorXd E_in(P.size())), then one loop.
    throw std::logic_error("TODO: port_energies");
}

double passivity_ratio(double out_trial, double budget) {
    // TODO C3: three cases.
    throw std::logic_error("TODO: passivity_ratio");
}

TdpaLeaderResult tdpa_leader(double F_m, double xdot_m, double h, double E_in_m, double E_out_m,
                             double E_in_s_recv) {
    // TODO C4: observer (update E_in_m or E_out_m) + controller (passivity_ratio).
    throw std::logic_error("TODO: tdpa_leader");
}

TdpaFollowerResult tdpa_follower(double F_c, double dx_d, double E_in_s, double E_out_s, double E_in_m_recv) {
    // TODO C5: the same at the follower port, acting on the set-point step dx_d.
    throw std::logic_error("TODO: tdpa_follower");
}

// ==========================================================================================
// Part D -- performance metrics for the report
// ==========================================================================================

Metrics performance_metrics(const SimResult& sim, double t_from) {
    // TODO D1: loop over the samples with sim.t(i) >= t_from for the means / rms / standard deviation
    //       (population, as np.std); E_ch_min over the WHOLE run (sim.E_ch.minCoeff()).
    throw std::logic_error("TODO: performance_metrics");
}

std::pair<double, double> latency_budget(const std::vector<std::pair<std::string, double>>& components_ms,
                                         double limit_ms) {
    // TODO D2: two lines (for (const auto& c : components_ms) ... c.second).
    throw std::logic_error("TODO: latency_budget");
}

}  // namespace lab
