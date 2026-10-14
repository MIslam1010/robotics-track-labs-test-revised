// SBEG 46X Lab 2 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_params, load_log                   data/teleop_params.json and data/leader_log.csv
//   minjerk, reach_intent                   hand-intent trajectories
//   simulate_teleop                         the sampled-data hand + leader + follower + environment simulator
//   jacobian_body                           MR Eq. 5.18
//   hapkit, eod_6r_ugv                      the two course robot models (Python: course_models.py)
//   saturate_force (scalar), hybrid_matrix (vector of s)   call your functions of starter.cpp
//
// These are the provided functions of starter.py and course_models.py, translated to C++.
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
constexpr double PI = 3.14159265358979323846;
constexpr double DEG = PI / 180.0;
constexpr double G0 = 9.81;

Eigen::Matrix<double, 6, 1> screw_revolute(const Eigen::Vector3d& w, const Eigen::Vector3d& q) {
    Eigen::Matrix<double, 6, 1> S;
    S << w, -w.cross(q);
    return S;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = p;
    return T;
}

/// np.gradient(x, dt): central differences inside, one-sided first differences at the two ends.
Eigen::VectorXd gradient(const Eigen::VectorXd& x, double dt) {
    const Eigen::Index N = x.size();
    Eigen::VectorXd g = Eigen::VectorXd::Zero(N);
    if (N < 2) throw std::invalid_argument("simulate_teleop: x_i needs at least 2 samples");
    g(0) = (x(1) - x(0)) / dt;
    g(N - 1) = (x(N - 1) - x(N - 2)) / dt;
    for (Eigen::Index k = 1; k + 1 < N; ++k) g(k) = (x(k + 1) - x(k - 1)) / (2.0 * dt);
    return g;
}
}  // namespace

// ------------------------------------------------------------------ data
Params load_params(const std::string& path) {
    const sbeg::json P = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/teleop_params.json") : path);
    Params p;
    p.m_m = P.at("leader").at("m").get<double>();
    p.b_m = P.at("leader").at("b").get<double>();
    p.m_s = P.at("follower").at("m").get<double>();
    p.b_s = P.at("follower").at("b").get<double>();
    p.m_h = P.at("hand").at("m").get<double>();
    p.b_h = P.at("hand").at("b").get<double>();
    p.k_h = P.at("hand").at("k").get<double>();
    p.kc = P.at("controller").at("kc").get<double>();
    p.bc = P.at("controller").at("bc").get<double>();
    p.alpha = P.at("alpha").get<double>();
    p.beta = P.at("beta").get<double>();
    p.f_max = P.at("force_max").get<double>();
    p.x_range = P.at("x_range").get<double>();
    p.dt = P.at("dt").get<double>();
    p.substeps = P.at("substeps").get<int>();
    for (const auto& item : P.at("environments").items()) {
        const sbeg::json& e = item.value();
        p.environments[item.key()] =
            Environment{e.at("k").get<double>(), e.at("b").get<double>(), e.at("x_wall").get<double>()};
    }
    const sbeg::json& rt = P.at("reach_task");
    p.reach_task = ReachTask{rt.at("x_start").get<double>(), rt.at("x_goal").get<double>(),
                             rt.at("t_move").get<double>(), rt.at("t_end").get<double>()};
    const sbeg::json& lp = P.at("laparoscopy");
    p.laparoscopy = Laparoscopy{lp.at("alpha").get<double>(), lp.at("insertion_m").get<double>(),
                                lp.at("stroke_m").get<double>()};
    const sbeg::json& tr = P.at("tremor");
    p.tremor = Tremor{tr.at("f_tremor_hz").get<double>(), tr.at("band_hz").at(0).get<double>(),
                      tr.at("band_hz").at(1).get<double>(), tr.at("voluntary_max_hz").get<double>()};
    const sbeg::json& eo = P.at("eod");
    p.eod = EodTask{eo.at("alpha").get<double>(), eo.at("approach_m").get<double>(), eo.at("stroke_m").get<double>(),
                    eo.at("speed_fraction").get<double>()};
    return p;
}

LeaderLog load_log(const std::string& path) {
    const sbeg::CsvTable T = sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/leader_log.csv") : path);
    LeaderLog L;
    L.t = T.data.col(T.col("t"));
    L.x_m = T.data.col(T.col("x_m"));
    const Eigen::VectorXd c = T.data.col(T.col("clutch"));
    L.clutch.resize(static_cast<std::size_t>(c.size()));
    for (Eigen::Index k = 0; k < c.size(); ++k) L.clutch[static_cast<std::size_t>(k)] = c(k) != 0.0;
    return L;
}

// ------------------------------------------------------------------ trajectories
Eigen::VectorXd minjerk(const Eigen::VectorXd& t, double t0, double t1, double x0, double x1) {
    Eigen::VectorXd x(t.size());
    for (Eigen::Index k = 0; k < t.size(); ++k) {
        const double s = std::clamp((t(k) - t0) / (t1 - t0), 0.0, 1.0);
        x(k) = x0 + (x1 - x0) * (10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s);
    }
    return x;
}

Eigen::VectorXd reach_intent(const Params& p, const Eigen::VectorXd& t) {
    const ReachTask& rt = p.reach_task;
    return minjerk(t, 0.0, rt.t_move, rt.x_start, rt.x_goal);
}

// ------------------------------------------------------------------ simulator
TeleopSim simulate_teleop(const std::string& arch, const Params& p, const Environment& env, const Eigen::VectorXd& x_i,
                          double dt, int substeps, bool saturate, bool watchdog, const ControlLaw& control) {
    const ControlLaw law = control ? control : ControlLaw(control_forces);
    if (dt == 0.0) dt = p.dt;
    if (substeps == 0) substeps = p.substeps;
    const Eigen::Index N = x_i.size();
    const Eigen::VectorXd v_i = gradient(x_i, dt);
    const double h = dt / substeps;
    const double mL = p.m_m + p.m_h;
    double xm = x_i(0), vm = 0.0, xs = p.alpha * x_i(0), vs = 0.0;
    TeleopSim out;
    for (Eigen::VectorXd* v : {&out.x_m, &out.v_m, &out.x_s, &out.v_s, &out.f_h, &out.f_e, &out.a_m, &out.a_s})
        *v = Eigen::VectorXd::Zero(N);
    bool tripped = false;

    auto fenv = [&env](double xs_, double vs_) {
        const double pen = xs_ - env.x_wall;
        if (pen <= 0.0 || env.k == 0.0) return 0.0;
        return std::max(0.0, env.k * pen + env.b * vs_);
    };

    for (Eigen::Index k = 0; k < N; ++k) {
        const double fe = fenv(xs, vs);
        auto [am, as] = law(arch, xm, vm, xs, vs, fe, p);
        if (saturate) {
            am = std::clamp(am, -p.f_max, p.f_max);
            as = std::clamp(as, -p.f_max, p.f_max);
        }
        if (watchdog && (tripped || std::abs(xm) > p.x_range)) {
            tripped = true;
            am = as = 0.0;
        }
        for (int i = 0; i < substeps; ++i) {
            const double fh = p.k_h * (x_i(k) - xm) + p.b_h * (v_i(k) - vm);
            const double fe_c = fenv(xs, vs);
            vm += h * (fh + am - p.b_m * vm) / mL;
            xm += h * vm;
            vs += h * (as - p.b_s * vs - fe_c) / p.m_s;
            xs += h * vs;
        }
        out.x_m(k) = xm;
        out.v_m(k) = vm;
        out.x_s(k) = xs;
        out.v_s(k) = vs;
        out.f_h(k) = p.k_h * (x_i(k) - xm) + p.b_h * (v_i(k) - vm);
        out.f_e(k) = fenv(xs, vs);
        out.a_m(k) = am;
        out.a_s(k) = as;
    }
    out.t = Eigen::VectorXd::LinSpaced(N, 0.0, static_cast<double>(N - 1)) * dt;
    out.tripped = tripped;
    return out;
}

// ------------------------------------------------------------------ kinematics
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& q) {
    const Eigen::Index n = Blist.cols();
    if (q.size() != n) throw std::invalid_argument("jacobian_body: q has the wrong length");
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    J.col(n - 1) = Blist.col(n - 1);
    for (Eigen::Index i = n - 2; i >= 0; --i) {
        const mr::Vector6d B = Blist.col(i + 1);
        T = (T * mr::matrix_exp6(mr::vec_to_se3(-B * q(i + 1)))).eval();
        J.col(i) = mr::adjoint(T) * Blist.col(i);
    }
    return J;
}

// ------------------------------------------------------------------ course robot models (course_models.py)
HapkitModel hapkit() {
    const double rh = 0.090, rs = 0.073, rp = 0.0042;
    HapkitModel h;
    h.Slist = screw_revolute(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d::Zero());
    h.M = trans(Eigen::Vector3d(0, rh, 0));
    h.mass = 0.10;
    h.com = Eigen::Vector3d(0.0, -0.010, 0.0);
    h.I_com = Eigen::Vector3d(2.0e-4, 1.0e-4, 1.5e-4).asDiagonal();
    h.g = Eigen::Vector3d(0.0, -G0, 0.0);
    h.limits.resize(1, 2);
    h.limits << -0.040 / rh, 0.040 / rh;
    h.params = HapkitParams{rh, rs, rp, rs / rp, 1.0e-6, 1.0e-6 * (rs / rp) * (rs / rp), 2.0, 0.040, 1000.0};
    return h;
}

EodModel eod_6r_ugv(double shoulder_a, double shoulder_h, double L1, double L2, double Lg) {
    const Eigen::Vector3d sh(shoulder_a, 0, shoulder_h);
    const Eigen::Vector3d el = sh + Eigen::Vector3d(L1, 0, 0);
    const Eigen::Vector3d wr = el + Eigen::Vector3d(L2, 0, 0);
    const Eigen::Vector3d w[6] = {{0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}};
    const Eigen::Vector3d q[6] = {Eigen::Vector3d::Zero(), sh, el, wr, wr, wr};
    EodModel m;
    m.Slist.resize(6, 6);
    for (int i = 0; i < 6; ++i) m.Slist.col(i) = screw_revolute(w[i], q[i]);
    m.M = trans(wr + Eigen::Vector3d(Lg, 0, 0));
    m.Blist = mr::adjoint(mr::trans_inv(m.M)) * m.Slist;
    m.limits.resize(6, 2);
    m.limits << -180, 180, -60, 110, -160, 160, -180, 180, -115, 115, -180, 180;
    m.limits *= DEG;
    m.qdot_max = Eigen::VectorXd::Constant(6, 30 * DEG);
    m.params.shoulder_a = shoulder_a;
    m.params.shoulder_h = shoulder_h;
    m.params.L1 = L1;
    m.params.L2 = L2;
    m.params.Lg = Lg;
    m.params.q_stow.resize(6);
    m.params.q_stow << -90, 70, -150, 0, 50, 0;
    m.params.q_stow *= DEG;
    m.params.q_ready.resize(6);
    m.params.q_ready << 0, 30, -70, 0, 10, 0;
    m.params.q_ready *= DEG;
    m.params.qddot_max = 60 * DEG;
    return m;
}

// ------------------------------------------------------------------ overloads that call your functions
double saturate_force(double f, double f_max) {
    const Eigen::VectorXd y = saturate_force(Eigen::VectorXd::Constant(1, f), f_max);
    if (y.size() != 1) throw std::invalid_argument("saturate_force must return one value per input value");
    return y(0);
}

std::vector<Eigen::Matrix2cd> hybrid_matrix(const std::string& arch, const Eigen::VectorXcd& s, const Params& p) {
    std::vector<Eigen::Matrix2cd> H;
    H.reserve(static_cast<std::size_t>(s.size()));
    for (Eigen::Index k = 0; k < s.size(); ++k) H.push_back(hybrid_matrix(arch, s(k), p));
    return H;
}

}  // namespace lab
