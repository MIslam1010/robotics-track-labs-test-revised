// SBEG 46X Hardware rotation 1 (Week 8) -- C++ track: GIVEN code (do not edit). The provided part of starter.py,
// translated:
//   data_path, load_task, load_record, write_log, load_log      data/hw1_task.json, the session record, the logs
//   minjerk, intent, tissue_force                               the operator intent and the tissue wall
//   Device, device_from, VelocityEstimator                      the device model and the firmware velocity
//   simulate_pp, simulate_follower, simulate_free_decay         the three simulators
//   rms, num_or (= _num), extrema (= _extrema)
// plus stand-ins for NumPy/SciPy: arange, interp, gradient, median, curve_fit (Levenberg-Marquardt).
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
double jd(const sbeg::json& j, const char* key) { return j.at(key).get<double>(); }
}  // namespace

// ------------------------------------------------------------------ files
std::string data_path(const std::string& name) { return sbeg::lab_path("../data/" + name); }

sbeg::json load_task(const std::string& path) {
    return sbeg::load_json(path.empty() ? data_path("hw1_task.json") : path);
}

sbeg::json load_record(const std::string& path) {
    return sbeg::load_json(path.empty() ? data_path("session_record_template.json") : path);
}

void write_log(const std::string& path, const VectorXd& t, const VectorXd& cmd, const VectorXd& meas,
               const VectorXd& cur, const VectorXd& force,
               const std::vector<std::pair<std::string, VectorXd>>& aux) {
    std::vector<std::string> header = {"t", "cmd_1", "meas_1"};
    std::vector<const VectorXd*> cols = {&t, &cmd, &meas};
    if (cur.size() > 0) {
        header.push_back("cur_1");
        cols.push_back(&cur);
    }
    if (force.size() > 0) {
        header.push_back("force");
        cols.push_back(&force);
    }
    for (const auto& kv : aux) {
        header.push_back("aux_" + kv.first);
        cols.push_back(&kv.second);
    }
    MatrixXd M(t.size(), static_cast<Index>(cols.size()));
    for (std::size_t k = 0; k < cols.size(); ++k) {
        if (cols[k]->size() != t.size()) throw std::invalid_argument("write_log: all columns must have len(t) samples");
        M.col(static_cast<Index>(k)) = *cols[k];
    }
    sbeg::write_csv(path, header, M);
}

Log load_log(const std::string& path_in) {
    std::string path = path_in;
    const std::filesystem::path p = sbeg::fs_path(path_in);
    if (!p.is_absolute() && !std::filesystem::exists(p)) path = data_path(path_in);
    const sbeg::CsvTable tab = sbeg::read_csv(path);
    const MatrixXd& D = tab.data;
    const Index ct = tab.col("t");
    // sort rows by time (stable), drop duplicate time stamps (keep the first)
    std::vector<Index> order(static_cast<std::size_t>(D.rows()));
    std::iota(order.begin(), order.end(), Index(0));
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return D(a, ct) < D(b, ct); });
    std::vector<Index> keep;
    for (std::size_t i = 0; i < order.size(); ++i)
        if (i == 0 || D(order[i], ct) != D(order[i - 1], ct)) keep.push_back(order[i]);
    auto column = [&](Index c) {
        VectorXd v(static_cast<Index>(keep.size()));
        for (std::size_t i = 0; i < keep.size(); ++i) v(static_cast<Index>(i)) = D(keep[i], c);
        return v;
    };
    Log out;
    out.t = column(ct);
    out.cmd = column(tab.col("cmd_1"));
    out.meas = column(tab.col("meas_1"));
    for (std::size_t c = 0; c < tab.header.size(); ++c) {
        const std::string& h = tab.header[c];
        if (h == "cur_1") out.cur = column(static_cast<Index>(c));
        if (h == "force") out.force = column(static_cast<Index>(c));
        if (h.rfind("aux_", 0) == 0) out.aux[h.substr(4)] = column(static_cast<Index>(c));
    }
    return out;
}

// ------------------------------------------------------------------ operator, tissue
VectorXd minjerk(const VectorXd& t, double t0, double T, double x0, double x1) {
    VectorXd y(t.size());
    for (Index i = 0; i < t.size(); ++i) {
        const double s = std::min(std::max((t(i) - t0) / T, 0.0), 1.0);
        y(i) = x0 + (x1 - x0) * (10 * std::pow(s, 3) - 15 * std::pow(s, 4) + 6 * std::pow(s, 5));
    }
    return y;
}

VectorXd intent(const sbeg::json& task, const VectorXd& t) {
    const sbeg::json& tr = task.at("trial");
    const VectorXd up = minjerk(t, jd(tr, "t_on"), jd(tr, "t_reach"), jd(tr, "x_start"), jd(tr, "x_goal"));
    const VectorXd down = minjerk(t, jd(tr, "t_hold_end"), jd(tr, "t_back"), 0.0, jd(tr, "x_start") - jd(tr, "x_goal"));
    return up + down;
}

double tissue_force(double x, double v, const sbeg::json& env) {
    const double d = x - jd(env, "x_wall");
    return d > 0 ? jd(env, "k") * d + jd(env, "b") * v : 0.0;
}

// ------------------------------------------------------------------ device
double Device::measure() const { return std::nearbyint(x / x_res) * x_res; }

double Device::command(double F, double scale) {
    if (std::abs(x) > x_range) tripped = true;
    if (tripped) {
        F_motor = 0.0;
        return 0.0;
    }
    if (std::abs(F) > force_max) ++sat_count;
    F_motor = std::min(std::max(F, -force_max), force_max) * scale;
    return F_motor;
}

void Device::advance(double dt, double F_ext, double m_extra) {
    const double fr = f_c * std::tanh(v / 1e-3);
    const double a = (F_motor - b_h * v - k_g * x - fr + F_ext) / (m_h + m_extra);
    v += a * dt;  // semi-implicit Euler, as in the project starter
    x += v * dt;
}

Device device_from(const sbeg::json& params, const sbeg::json& overrides) {
    Device d;
    auto set = [&d](const std::string& k, double val, bool any_field) {
        if (k == "m_h") d.m_h = val;
        else if (k == "b_h") d.b_h = val;
        else if (k == "k_g") d.k_g = val;
        else if (k == "x_res") d.x_res = val;
        else if (k == "force_max") d.force_max = val;
        else if (k == "x_range") d.x_range = val;
        else if (k == "f_c") d.f_c = val;
        else if (any_field && k == "T") d.T = val;
        else if (any_field && k == "x") d.x = val;
        else if (any_field && k == "v") d.v = val;
        else if (any_field) throw std::invalid_argument("device_from: unknown Device field '" + k + "'");
    };
    for (const char* k : {"m_h", "b_h", "k_g", "x_res", "force_max", "x_range", "f_c"})
        if (params.contains(k)) set(k, params.at(k).get<double>(), false);
    for (auto it = overrides.begin(); it != overrides.end(); ++it) set(it.key(), it.value().get<double>(), true);
    return d;
}

VelocityEstimator::VelocityEstimator(double T_, double fc) : T(T_), alpha(1.0 - std::exp(-2 * PI * fc * T_)) {}

double VelocityEstimator::operator()(double x) {
    if (has_prev) v += alpha * ((x - prev) / T - v);
    prev = x;
    has_prev = true;
    return v;
}

// ------------------------------------------------------------------ simulators
PPSim simulate_pp(const sbeg::json& task, Device* leader, Device* follower, const sbeg::json* op_in,
                  std::optional<double> T_end_in, std::optional<double> t_stop, const VectorXd* x_int) {
    const double h = jd(task, "dt");
    const int sub = task.at("substeps").get<int>();
    const sbeg::json& pp = task.at("pp");
    const sbeg::json& env = task.at("tissue");
    const sbeg::json& op = op_in ? *op_in : task.at("operator_nominal");
    Device L_own = device_from(task.at("device_nominal")), F_own = device_from(task.at("device_nominal"));
    Device& L = leader ? *leader : L_own;
    Device& F = follower ? *follower : F_own;
    const double T_end = T_end_in ? *T_end_in : jd(task.at("trial"), "t_end");
    const Index n = static_cast<Index>(std::nearbyint(T_end / h));
    VectorXd t(n);
    for (Index k = 0; k < n; ++k) t(k) = static_cast<double>(k) * h;
    VectorXd xi = x_int ? *x_int : intent(task, t);
    if (xi.size() < n) throw std::invalid_argument("simulate_pp: x_int is shorter than the simulation");
    const double alpha = jd(pp, "alpha"), beta = jd(pp, "beta"), Kp = jd(pp, "Kp"), Kd = jd(pp, "Kd");
    const double op_k = jd(op, "k"), op_b = jd(op, "b"), op_m = jd(op, "m");
    L.x = L.v = F.v = 0.0;
    F.x = alpha * L.x;
    VelocityEstimator est_m(h), est_s(h);
    PPSim out;
    for (VectorXd* v : {&out.x_m, &out.x_s, &out.xm_meas, &out.xs_meas, &out.x_d, &out.F_m, &out.F_c, &out.f_env,
                        &out.F_s, &out.E_ch})
        *v = VectorXd::Zero(n);
    double pkt_x = L.measure(), pkt_F = 0.0;  // packets in flight (one sample)
    double x_d_prev = 0.0, E = 0.0;
    bool have_prev = false;
    const double fade = jd(task.at("stop"), "fade_s");
    for (Index k = 0; k < n; ++k) {
        const double tk = t(k);
        const double xm = L.measure(), xs = F.measure();
        est_m(xm);
        const double vs_est = est_s(xs);
        const double f_env = tissue_force(xs, vs_est, env);
        const double x_d = alpha * pkt_x;  // follower side: received leader position
        const double dx = have_prev ? x_d - x_d_prev : 0.0;
        x_d_prev = x_d;
        have_prev = true;
        const double F_c = Kp * (x_d - xs) - Kd * vs_est;
        const double F_m = beta * pkt_F;  // leader side: received follower force
        pkt_x = xm;                       // send (arrive next tick)
        pkt_F = F_c;
        double scale = 1.0;
        if (t_stop && tk > *t_stop)
            scale = (tk - *t_stop > h) ? std::max(0.0, 1.0 - (tk - *t_stop - h) / fade) : 1.0;
        const double F_m_app = -L.command(-F_m, scale);
        F.command(F_c - f_env, scale);
        E += (F_m_app * L.v - F_c * dx / h) * h;
        out.x_m(k) = L.x;
        out.x_s(k) = F.x;
        out.xm_meas(k) = xm;
        out.xs_meas(k) = xs;
        out.x_d(k) = x_d;
        out.F_m(k) = F_m_app;
        out.F_c(k) = F_c;
        out.f_env(k) = f_env;
        out.F_s(k) = F.F_motor;
        out.E_ch(k) = E;
        for (int s = 0; s < sub; ++s) {
            const double Fh = op_k * (xi(k) - L.x) - op_b * L.v;
            L.advance(h / sub, Fh, op_m);
            F.advance(h / sub);
        }
    }
    out.t = t;
    out.x_int = xi;
    out.sat_m = L.sat_count;
    out.sat_s = F.sat_count;
    out.tripped_m = L.tripped;
    out.tripped_s = F.tripped;
    return out;
}

FollowerSim simulate_follower(const VectorXd& t_log, const VectorXd& x_d_log, double Kp, double Kd, Device& dev,
                              const VectorXd* f_env_log, double h, int substeps) {
    if (t_log.size() < 2 || x_d_log.size() != t_log.size())
        throw std::invalid_argument("simulate_follower: t_log and x_d_log must have the same length (>= 2)");
    const VectorXd t = arange(t_log(0), t_log(t_log.size() - 1) + 0.5 * h, h);
    const VectorXd r = interp(t, t_log, x_d_log);
    const VectorXd fe = f_env_log ? interp(t, t_log, *f_env_log) : VectorXd::Zero(t.size()).eval();
    dev.x = r(0);
    dev.v = 0.0;
    VelocityEstimator est(h);
    VectorXd x = VectorXd::Zero(t.size());
    for (Index k = 0; k < t.size(); ++k) {
        const double xm = dev.measure();
        dev.command(Kp * (r(k) - xm) - Kd * est(xm) - fe(k));
        x(k) = dev.x;
        for (int s = 0; s < substeps; ++s) dev.advance(h / substeps);
    }
    return FollowerSim{t, x};
}

DecaySim simulate_free_decay(Device& dev, double K, double x0, double t_release, double T_end, double h,
                             int substeps) {
    const Index n = static_cast<Index>(std::nearbyint(T_end / h));
    DecaySim out;
    out.t.resize(n);
    out.x.resize(n);
    out.F.resize(n);
    dev.x = x0;
    dev.v = 0.0;
    for (Index k = 0; k < n; ++k) {
        out.t(k) = static_cast<double>(k) * h;
        const double xm = dev.measure();
        dev.command(-K * xm);
        out.x(k) = xm;
        out.F(k) = dev.F_motor;
        for (int s = 0; s < substeps; ++s) {
            if (out.t(k) < t_release) {
                dev.x = x0;
                dev.v = 0.0;
            } else {
                dev.advance(h / substeps);
            }
        }
    }
    return out;
}

double rms(const VectorXd& x) {
    if (x.size() == 0) return std::numeric_limits<double>::quiet_NaN();  // np.mean of an empty array
    return std::sqrt(x.squaredNorm() / static_cast<double>(x.size()));
}

double num_or(const sbeg::json& d, const std::string& key, double fallback) {
    if (!d.is_object() || !d.contains(key) || d.at(key).is_null()) return fallback;
    return d.at(key).get<double>();
}

bool flag_or(const sbeg::json& d, const std::string& key, bool fallback) {
    if (!d.is_object() || !d.contains(key)) return fallback;
    const sbeg::json& v = d.at(key);
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0.0;
    if (v.is_string()) return !v.get<std::string>().empty();
    return !v.empty();
}

std::string text_or(const sbeg::json& d, const std::string& key) {
    if (!d.is_object() || !d.contains(key) || !d.at(key).is_string()) return "";
    return d.at(key).get<std::string>();
}

sbeg::json sub_or_empty(const sbeg::json& d, const std::string& key) {
    if (!d.is_object() || !d.contains(key)) return sbeg::json::object();
    return d.at(key);
}

std::vector<Index> extrema(const VectorXd& tt, const VectorXd& y, double sign) {
    const Index n = y.size();
    std::vector<Index> start;  // first sample of each run of equal values
    for (Index i = 0; i < n; ++i)
        if (i == 0 || sign * y(i) != sign * y(i - 1)) start.push_back(i);
    std::vector<Index> idx;
    for (std::size_t r = 1; r + 1 < start.size(); ++r) {
        const double v = sign * y(start[r]), vp = sign * y(start[r - 1]), vn = sign * y(start[r + 1]);
        if (v > vp && v > vn) idx.push_back((start[r] + start[r + 1] - 1) / 2);
    }
    if (idx.empty()) throw std::invalid_argument("extrema: no local extremum in the signal");
    std::vector<Index> keep = {idx[0]};
    for (std::size_t k = 1; k < idx.size(); ++k)
        if (tt(idx[k]) - tt(keep.back()) > 0.02) keep.push_back(idx[k]);
    return keep;
}

// ------------------------------------------------------------------ NumPy / SciPy stand-ins
VectorXd arange(double start, double stop, double step) {
    const double len = std::ceil((stop - start) / step);
    const Index n = len > 0 ? static_cast<Index>(len) : 0;
    VectorXd v(n);
    if (n > 0) v(0) = start;
    if (n > 1) v(1) = start + step;
    const double delta = n > 1 ? v(1) - v(0) : step;
    for (Index i = 2; i < n; ++i) v(i) = start + static_cast<double>(i) * delta;
    return v;
}

VectorXd interp(const VectorXd& x, const VectorXd& xp, const VectorXd& fp) {
    if (xp.size() == 0 || xp.size() != fp.size()) throw std::invalid_argument("interp: xp and fp must have the same length");
    VectorXd y(x.size());
    const Index n = xp.size();
    for (Index i = 0; i < x.size(); ++i) {
        const double xi = x(i);
        if (xi <= xp(0)) {
            y(i) = fp(0);
        } else if (xi >= xp(n - 1)) {
            y(i) = fp(n - 1);
        } else {
            const Index k = static_cast<Index>(std::upper_bound(xp.data(), xp.data() + n, xi) - xp.data()) - 1;
            if (xp(k) == xi) {
                y(i) = fp(k);
            } else {
                const double slope = (fp(k + 1) - fp(k)) / (xp(k + 1) - xp(k));
                y(i) = slope * (xi - xp(k)) + fp(k);
            }
        }
    }
    return y;
}

VectorXd gradient(const VectorXd& y, const VectorXd& x) {
    const Index n = y.size();
    if (n < 2 || x.size() != n) throw std::invalid_argument("gradient: y and x must have the same length (>= 2)");
    VectorXd dx = x.tail(n - 1) - x.head(n - 1);
    const bool uniform = (dx.array() == dx(0)).all();
    VectorXd g(n);
    for (Index i = 1; i + 1 < n; ++i) {
        if (uniform) {
            g(i) = (y(i + 1) - y(i - 1)) / (2.0 * dx(0));
        } else {
            const double d1 = dx(i - 1), d2 = dx(i);
            const double a = -d2 / (d1 * (d1 + d2)), b = (d2 - d1) / (d1 * d2), c = d1 / (d2 * (d1 + d2));
            g(i) = a * y(i - 1) + b * y(i) + c * y(i + 1);
        }
    }
    g(0) = (y(1) - y(0)) / dx(0);
    g(n - 1) = (y(n - 1) - y(n - 2)) / dx(n - 2);
    return g;
}

double median(const VectorXd& v) {
    if (v.size() == 0) throw std::invalid_argument("median of an empty vector");
    std::vector<double> w(v.data(), v.data() + v.size());
    std::sort(w.begin(), w.end());
    const std::size_t m = w.size() / 2;
    return w.size() % 2 ? w[m] : 0.5 * (w[m - 1] + w[m]);
}

VectorXd curve_fit(const std::function<double(double, const VectorXd&)>& model, const VectorXd& xdata,
                   const VectorXd& ydata, const VectorXd& p0, int maxfev) {
    const Index m = xdata.size(), p = p0.size();
    if (ydata.size() != m || m < p) throw std::invalid_argument("curve_fit: need len(xdata) == len(ydata) >= len(p0)");
    int nfev = 0;
    auto resid = [&](const VectorXd& q) {
        ++nfev;
        VectorXd r(m);
        for (Index i = 0; i < m; ++i) r(i) = model(xdata(i), q) - ydata(i);
        return r;
    };
    const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
    VectorXd x = p0, r = resid(x);
    double cost = r.squaredNorm();
    if (!std::isfinite(cost)) throw std::runtime_error("curve_fit: the model is not finite at p0");
    double lambda = 1e-3;
    VectorXd D = VectorXd::Zero(p);  // Marquardt scaling (largest column norms so far, as MINPACK)
    while (nfev < maxfev) {
        MatrixXd J(m, p);
        for (Index k = 0; k < p; ++k) {
            double hk = eps * std::abs(x(k));
            if (hk == 0.0) hk = eps;
            VectorXd xk = x;
            xk(k) += hk;
            J.col(k) = (resid(xk) - r) / hk;
            D(k) = std::max(D(k), J.col(k).norm());
        }
        const MatrixXd A = J.transpose() * J;
        const VectorXd g = J.transpose() * r;
        if ((g.array().abs() / (D.array() * std::sqrt(cost) + 1e-300)).maxCoeff() < 1e-14) return x;  // gradient zero
        bool accepted = false;
        while (!accepted && nfev < maxfev) {
            MatrixXd Al = A;
            for (Index k = 0; k < p; ++k) Al(k, k) += lambda * D(k) * D(k);
            const VectorXd dp = Al.ldlt().solve(-g);
            const VectorXd xn = x + dp;
            const VectorXd rn = resid(xn);
            const double cn = rn.squaredNorm();
            if (std::isfinite(cn) && cn < cost) {
                accepted = true;
                const double rel_red = (cost - cn) / cost;
                const bool small_step = (D.cwiseProduct(dp)).norm() <= 1e-12 * (D.cwiseProduct(xn)).norm();
                x = xn;
                r = rn;
                cost = cn;
                lambda = std::max(lambda / 10.0, 1e-15);
                if (rel_red < 1e-15 || small_step) return x;
            } else {
                lambda *= 10.0;
                if (lambda > 1e16) return x;  // no further decrease possible: converged to machine precision
            }
        }
    }
    throw std::runtime_error("curve_fit: no convergence within maxfev = " + std::to_string(maxfev) + " evaluations");
}

}  // namespace lab
