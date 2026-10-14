// SBEG 46X Lab 4 -- C++ track: GIVEN code (do not edit).  The "Provided" functions of starter.py, translated:
//   load_device, load_tissue, load_csv      data/device.json, data/needle_tissue.json, data/*.csv
//   rk4_step, simulate_device, hand_ramp    the sampled-data device simulator and the hand model
//   hapkit                                  the shared course model (= course_models.py, numbers identical)
//   required_damping (one theta)            calls your vector version
//   ProxySlab (constructor, force), NeedleTissue (constructor)
//   needle_quasistatic_profile, render_needle
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

namespace {
constexpr double G0 = 9.81;

std::string data_path(const std::string& name) { return sbeg::lab_path("../data/" + name); }

// ---- helpers of the shared course robot models (Course resources)
sbeg::Vector6d screw_revolute(const Eigen::Vector3d& w, const Eigen::Vector3d& q, double h = 0.0) {
    sbeg::Vector6d S;
    S << w, -w.cross(q) + h * w;
    return S;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p) { return mr::rp_to_trans(Eigen::Matrix3d::Identity(), p); }

sbeg::Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib) {
    sbeg::Matrix6d G = sbeg::Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = Ib.asDiagonal();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

mr::TransformList mlist_from_absolute(const mr::TransformList& com_frames, const Eigen::Matrix4d& M_ee) {
    mr::TransformList frames;
    frames.push_back(Eigen::Matrix4d::Identity());
    for (const auto& F : com_frames) frames.push_back(F);
    frames.push_back(M_ee);
    mr::TransformList out;
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) out.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    return out;
}
}  // namespace

// ---------------------------------------------------------------------------
// Data files
// ---------------------------------------------------------------------------
Device load_device() {
    const sbeg::json j = sbeg::load_json(data_path("device.json"));
    Device d;
    d.T = j.at("T").get<double>();
    d.x_res = j.at("x_res").get<double>();
    d.force_max = j.at("force_max").get<double>();
    d.x_range = j.at("x_range").get<double>();
    d.k_t = j.at("k_t").get<double>();
    d.m_h = j.at("m_h").get<double>();
    d.b_h = j.at("b_h").get<double>();
    d.k_g = j.at("k_g").get<double>();
    d.f_c = j.value("f_c", 0.0);
    d.note = j.value("note", std::string());
    return d;
}

TissueParams load_tissue() {
    const sbeg::json j = sbeg::load_json(data_path("needle_tissue.json"));
    TissueParams p;
    p.skin.a1 = j.at("skin").at("a1").get<double>();
    p.skin.a2 = j.at("skin").at("a2").get<double>();
    p.skin.f_p = j.at("skin").at("f_p").get<double>();
    p.tissue.f_cut = j.at("tissue").at("f_cut").get<double>();
    p.tissue.mu = j.at("tissue").at("mu").get<double>();
    p.pleura.z_pl = j.at("pleura").at("z_pl").get<double>();
    p.pleura.k_pl = j.at("pleura").at("k_pl").get<double>();
    p.pleura.f_p = j.at("pleura").at("f_p").get<double>();
    p.v_eps = j.at("v_eps").get<double>();
    p.x_skin = j.at("x_skin").get<double>();
    p.note = j.value("note", std::string());
    return p;
}

Columns load_csv(const std::string& name) {
    const sbeg::CsvTable tab = sbeg::read_csv(data_path(name));
    Columns out;
    for (std::size_t k = 0; k < tab.header.size(); ++k) out[tab.header[k]] = tab.data.col(static_cast<Eigen::Index>(k));
    return out;
}

// ---------------------------------------------------------------------------
// The device simulator and the hand
// ---------------------------------------------------------------------------
Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& s, double h) {
    const Eigen::VectorXd k1 = f(s);
    const Eigen::VectorXd k2 = f(s + 0.5 * h * k1);
    const Eigen::VectorXd k3 = f(s + 0.5 * h * k2);
    const Eigen::VectorXd k4 = f(s + h * k3);
    return s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

DeviceSim simulate_device(const Device& dev, const ForceLaw& law, double T_end, double x0, double v0,
                          const ForceLaw& f_hand, bool quantize, bool friction, int substeps,
                          const StopCondition& stop) {
    const ForceLaw hand = f_hand ? f_hand : ForceLaw([](double, double, double) { return 0.0; });
    const double T = dev.T, m = dev.m_h, b = dev.b_h, kg = dev.k_g, Fmax = dev.force_max;
    const double fc = friction ? dev.f_c : 0.0;
    const int n = static_cast<int>(std::nearbyint(T_end / T));  // Python round(): halves to even
    Eigen::VectorXd t(n + 1), X0 = Eigen::VectorXd::Zero(n + 1), X1 = Eigen::VectorXd::Zero(n + 1);
    Eigen::VectorXd xm = Eigen::VectorXd::Zero(n + 1), ve = Eigen::VectorXd::Zero(n + 1);
    Eigen::VectorXd Fa = Eigen::VectorXd::Zero(n + 1);
    std::vector<bool> sat(static_cast<std::size_t>(n + 1), false);
    for (int k = 0; k <= n; ++k) t(k) = k * T;
    X0(0) = x0;
    X1(0) = v0;
    int last = n;
    for (int k = 0; k <= n; ++k) {
        const double x = X0(k), v = X1(k);
        xm(k) = quantize ? std::nearbyint(x / dev.x_res) * dev.x_res : x;
        ve(k) = k == 0 ? v0 : (xm(k) - xm(k - 1)) / T;
        const double F = law(t(k), xm(k), ve(k));
        sat[static_cast<std::size_t>(k)] = std::abs(F) > Fmax;
        Fa(k) = std::max(-Fmax, std::min(Fmax, F));
        if (k == n || (stop && stop(k, x, v))) {
            last = k;
            break;
        }
        const double Fk = Fa(k), tk = t(k);
        const StateFunction rhs = [&](const Eigen::VectorXd& s) {
            const double fr = fc != 0.0 ? fc * std::tanh(s(1) / 1e-3) : 0.0;
            Eigen::VectorXd ds(2);
            ds << s(1), (Fk + hand(tk, s(0), s(1)) - b * s(1) - kg * s(0) - fr) / m;
            return ds;
        };
        Eigen::VectorXd s(2);
        s << x, v;
        for (int i = 0; i < substeps; ++i) s = rk4_step(rhs, s, T / substeps);
        X0(k + 1) = s(0);
        X1(k + 1) = s(1);
    }
    DeviceSim out;
    out.t = t.head(last + 1);
    out.x = X0.head(last + 1);
    out.v = X1.head(last + 1);
    out.x_meas = xm.head(last + 1);
    out.v_est = ve.head(last + 1);
    out.F = Fa.head(last + 1);
    out.saturated.assign(sat.begin(), sat.begin() + (last + 1));
    return out;
}

ForceLaw hand_ramp(double x_start, double speed, double k_hand, double b_hand, double t_start) {
    return [=](double t, double x, double v) {
        const double tt = std::max(t - t_start, 0.0);
        return k_hand * (x_start + speed * tt - x) + b_hand * ((t >= t_start ? speed : 0.0) - v);
    };
}

// ---------------------------------------------------------------------------
// The shared course model (= course_models.hapkit())
// ---------------------------------------------------------------------------
HapkitModel hapkit() {
    const double rh = 0.090, rs = 0.073, rp = 0.0042;
    HapkitModel h;
    h.Slist = screw_revolute(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
    h.M = trans(Eigen::Vector3d(0, rh, 0));
    h.Mlist = mlist_from_absolute({trans(Eigen::Vector3d(0, -0.010, 0))}, h.M);  // sector mass below the pivot
    h.Glist = {body_inertia(0.10, Eigen::Vector3d(2.0e-4, 1.0e-4, 1.5e-4))};
    h.g = Eigen::Vector3d(0, -G0, 0);
    const double th_max = 0.040 / rh;
    h.limits.resize(1, 2);
    h.limits << -th_max, th_max;
    h.qdot_max = Eigen::VectorXd::Constant(1, 20.0);
    h.tau_max = Eigen::VectorXd::Constant(1, 2.0 * rh);
    h.params.rh = rh;
    h.params.rs = rs;
    h.params.rp = rp;
    h.params.ratio = rs / rp;
    h.params.rotor_inertia = 1.0e-6;
    h.params.reflected_rotor_inertia = 1.0e-6 * (rs / rp) * (rs / rp);
    h.params.force_max = 2.0;
    h.params.x_range = 0.040;
    h.params.loop_rate_hz = 1000.0;
    return h;
}

// ---------------------------------------------------------------------------
// Given parts of Parts A, C and D
// ---------------------------------------------------------------------------
double required_damping(double K, double B, double T, double theta) {
    const Eigen::VectorXd r = required_damping(K, B, T, Eigen::VectorXd::Constant(1, theta));
    return r(0);
}

ProxySlab::ProxySlab(double a_, double w_, double K_, double x0) : a(a_), w(w_), K(K_), p(x0) {
    if (a_ < x0 && x0 < a_ + w_) throw std::invalid_argument("start the proxy outside the slab");
}

double ProxySlab::force(double x) { return K * (update(x) - x); }

NeedleTissue::NeedleTissue(const TissueParams& p_) : p(p_) {}

Eigen::VectorXd needle_quasistatic_profile(const TissueParams& p, const Eigen::VectorXd& z) {
    NeedleTissue m(p);
    Eigen::VectorXd f(z.size());
    for (Eigen::Index i = 0; i < z.size(); ++i) f(i) = m.force(z(i), 1.0);
    return f;
}

NeedleSim render_needle(const Device& dev, const TissueParams& p, double x_skin, double speed, double T_end,
                        double x0, int n_vel, bool quantize, bool friction) {
    NeedleTissue model(p);
    std::vector<double> hist;
    const ForceLaw law = [&](double, double x, double) {
        hist.push_back(x);
        const int n = std::min(n_vel, static_cast<int>(hist.size()) - 1);
        const double v_f = n > 0 ? (hist.back() - hist[hist.size() - 1 - static_cast<std::size_t>(n)]) / (n * dev.T)
                                 : 0.0;
        return -model.force(x - x_skin, v_f);
    };
    NeedleSim sim;
    static_cast<DeviceSim&>(sim) =
        simulate_device(dev, law, T_end, x0, 0.0, hand_ramp(x0, speed), quantize, friction);
    sim.z = sim.x.array() - x_skin;
    sim.f_tissue = -sim.F;
    sim.events = model.events;
    return sim;
}

}  // namespace lab
