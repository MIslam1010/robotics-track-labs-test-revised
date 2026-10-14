// SBEG 461 Lab 8 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   hapkit, rehab_2r, rehab                      the two robot models (= course_models.py, numbers identical)
//   load_device, load_free_decay                 data/device.json, data/free_decay_K*.csv
//   rk4_step                                     one RK4 step of x' = f(x)
//   rehab_fk, rehab_jacobian, rehab_jacobian_dot, rehab_ik, rehab_dynamics, simulate_rehab
//   spectral_radius, oscillation_growth, pulse_force, one_dof_pulse_response
//   expm, find_peaks, linear_fit                 C++ only: scipy.linalg.expm, scipy.signal.find_peaks, np.polyfit(.., 1)
//   the scalar overloads of handle_force_to_current, impedance_control_torque, admittance_control_torque
//
// These are the "Provided" functions of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double G0 = 9.81;

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

/// Uniform solid cylinder (radius r) of length L along the CoM-frame x axis.
sbeg::Matrix6d rod_inertia_x(double m, double L, double r = 0.03) {
    const double Ia = 0.5 * m * r * r;
    const double It = m * (3.0 * r * r + L * L) / 12.0;
    return body_inertia(m, Eigen::Vector3d(Ia, It, It));
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

/// (m1, c1, I1, m2, c2, I2) of the rehab robot from its Glist (I about the CoM, z axis).
struct RehabInertial {
    double m1, c1, I1, m2, c2, I2;
};

RehabInertial rehab_inertial() {
    const RehabModel& R = rehab();
    return {R.Glist[0](3, 3), R.params.L1 / 2.0, R.Glist[0](2, 2),
            R.Glist[1](3, 3), R.params.L2 / 2.0, R.Glist[1](2, 2)};
}
}  // namespace

// ---------------------------------------------------------------------------
// Models and data
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

RehabModel rehab_2r() {
    const double L1 = 0.35, L2 = 0.35, m1 = 2.4, m2 = 1.6;
    RehabModel r;
    r.Slist.resize(6, 2);
    r.Slist.col(0) = screw_revolute(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
    r.Slist.col(1) = screw_revolute(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(L1, 0, 0));
    r.M = trans(Eigen::Vector3d(L1 + L2, 0, 0));
    r.Mlist = mlist_from_absolute({trans(Eigen::Vector3d(L1 / 2, 0, 0)), trans(Eigen::Vector3d(L1 + L2 / 2, 0, 0))},
                                  r.M);
    r.Glist = {rod_inertia_x(m1, L1), rod_inertia_x(m2, L2)};
    r.g = Eigen::Vector3d(0, 0, -G0);
    r.limits.resize(2, 2);
    r.limits << 0.0, PI, 0.2, 2.8;
    r.qdot_max = Eigen::Vector2d(3.0, 3.0);
    r.tau_max = Eigen::Vector2d(20.0, 12.0);
    r.q_home = Eigen::Vector2d(0.9, 1.6);
    r.params.L1 = L1;
    r.params.L2 = L2;
    r.params.m1 = m1;
    r.params.m2 = m2;
    r.params.path_A = Eigen::Vector2d(0.0, 0.35);
    r.params.path_B = Eigen::Vector2d(0.0, 0.55);
    return r;
}

const RehabModel& rehab() {
    static const RehabModel R = rehab_2r();
    return R;
}

Device load_device() {
    const sbeg::json j = sbeg::load_json(sbeg::lab_path("../data/device.json"));
    Device d;
    d.T = j.at("T").get<double>();
    d.x_res = j.at("x_res").get<double>();
    d.force_max = j.at("force_max").get<double>();
    d.x_range = j.at("x_range").get<double>();
    d.k_t = j.at("k_t").get<double>();
    d.i_max = j.at("i_max").get<double>();
    d.m_h = j.at("m_h").get<double>();
    d.b_h = j.at("b_h").get<double>();
    d.k_g = j.at("k_g").get<double>();
    if (j.contains("note")) d.note = j.at("note").get<std::string>();
    return d;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> load_free_decay(int K) {
    const sbeg::CsvTable tab = sbeg::read_csv(sbeg::lab_path("../data/free_decay_K" + std::to_string(K) + ".csv"));
    return {tab.data.col(tab.col("t")), tab.data.col(tab.col("x"))};
}

Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& x, double h) {
    const Eigen::VectorXd k1 = f(x);
    const Eigen::VectorXd k2 = f(x + 0.5 * h * k1);
    const Eigen::VectorXd k3 = f(x + 0.5 * h * k2);
    const Eigen::VectorXd k4 = f(x + h * k3);
    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ---------------------------------------------------------------------------
// Rehab robot kinematics, dynamics and simulator
// ---------------------------------------------------------------------------

Eigen::Vector2d rehab_fk(const Eigen::Vector2d& q) {
    const double L1 = rehab().params.L1, L2 = rehab().params.L2;
    return Eigen::Vector2d(L1 * std::cos(q(0)) + L2 * std::cos(q(0) + q(1)),
                           L1 * std::sin(q(0)) + L2 * std::sin(q(0) + q(1)));
}

Eigen::Matrix2d rehab_jacobian(const Eigen::Vector2d& q) {
    const double L1 = rehab().params.L1, L2 = rehab().params.L2;
    const double s1 = std::sin(q(0)), c1 = std::cos(q(0)), s12 = std::sin(q(0) + q(1)), c12 = std::cos(q(0) + q(1));
    Eigen::Matrix2d J;
    J << -L1 * s1 - L2 * s12, -L2 * s12,
         L1 * c1 + L2 * c12, L2 * c12;
    return J;
}

Eigen::Matrix2d rehab_jacobian_dot(const Eigen::Vector2d& q, const Eigen::Vector2d& dq) {
    const double L1 = rehab().params.L1, L2 = rehab().params.L2;
    const double w1 = dq(0), w12 = dq(0) + dq(1);
    const double s1 = std::sin(q(0)), c1 = std::cos(q(0)), s12 = std::sin(q(0) + q(1)), c12 = std::cos(q(0) + q(1));
    Eigen::Matrix2d Jd;
    Jd << -L1 * c1 * w1 - L2 * c12 * w12, -L2 * c12 * w12,
          -L1 * s1 * w1 - L2 * s12 * w12, -L2 * s12 * w12;
    return Jd;
}

Eigen::Vector2d rehab_ik(const Eigen::Vector2d& p) {
    const double L1 = rehab().params.L1, L2 = rehab().params.L2;
    const double x = p(0), y = p(1);
    const double c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2);
    const double q2 = std::acos(std::max(-1.0, std::min(1.0, c2)));
    const double q1 = std::atan2(y, x) - std::atan2(L2 * std::sin(q2), L1 + L2 * std::cos(q2));
    return Eigen::Vector2d(q1, q2);
}

std::pair<Eigen::Matrix2d, Eigen::Vector2d> rehab_dynamics(const Eigen::Vector2d& q, const Eigen::Vector2d& dq) {
    const RehabInertial r = rehab_inertial();
    const double L1 = rehab().params.L1;
    const double cq2 = std::cos(q(1)), sq2 = std::sin(q(1));
    const double M11 = r.I1 + r.m1 * r.c1 * r.c1 + r.I2 + r.m2 * (L1 * L1 + r.c2 * r.c2 + 2.0 * L1 * r.c2 * cq2);
    const double M12 = r.I2 + r.m2 * (r.c2 * r.c2 + L1 * r.c2 * cq2);
    const double M22 = r.I2 + r.m2 * r.c2 * r.c2;
    const double h = r.m2 * L1 * r.c2 * sq2;
    Eigen::Matrix2d M;
    M << M11, M12, M12, M22;
    const Eigen::Vector2d c(-h * (2.0 * dq(0) * dq(1) + dq(1) * dq(1)), h * dq(0) * dq(0));
    return {M, c};
}

RehabSim simulate_rehab(const RehabController& controller, const Eigen::Vector2d& q0, double T_end, double dt,
                        const HandForce& f_ext, const Eigen::Vector2d& dq0, int substeps, double force_lag_hz) {
    const auto hand = [&f_ext](double t, const Eigen::Vector2d& p, const Eigen::Vector2d& v) -> Eigen::Vector2d {
        return f_ext ? f_ext(t, p, v) : Eigen::Vector2d::Zero().eval();
    };
    const Eigen::Vector2d tau_max = rehab().tau_max;
    const auto n = static_cast<Eigen::Index>(std::llround(T_end / dt));
    RehabSim s;
    s.t.resize(n + 1);
    for (Eigen::Index k = 0; k <= n; ++k) s.t(k) = static_cast<double>(k) * dt;
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n + 1, 4);
    X.row(0) << q0.transpose(), dq0.transpose();
    s.tau = Eigen::MatrixXd::Zero(n + 1, 2);
    s.f = Eigen::MatrixXd::Zero(n + 1, 2);
    const bool lag = force_lag_hz > 0.0;
    const double alpha = lag ? 1.0 - std::exp(-2.0 * PI * force_lag_hz * dt) : 0.0;
    Eigen::Vector2d f_filt = Eigen::Vector2d::Zero();
    bool have_filt = false;
    for (Eigen::Index k = 0; k <= n; ++k) {
        const Eigen::Vector2d q = X.block<1, 2>(k, 0).transpose(), dq = X.block<1, 2>(k, 2).transpose();
        const Eigen::Vector2d p = rehab_fk(q), v = rehab_jacobian(q) * dq;
        const Eigen::Vector2d fk = hand(s.t(k), p, v);
        s.f.row(k) = fk.transpose();
        Eigen::Vector2d f_meas = fk;
        if (lag) {
            f_filt = have_filt ? (f_filt + alpha * (fk - f_filt)).eval() : fk;
            have_filt = true;
            f_meas = f_filt;
        }
        const Eigen::Vector2d raw = controller(s.t(k), q, dq, f_meas);
        const Eigen::Vector2d tau = raw.cwiseMax(-tau_max).cwiseMin(tau_max);
        s.tau.row(k) = tau.transpose();
        if (k == n) break;
        const double h = dt / substeps, tk = s.t(k);
        const StateFunction rhs = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            const Eigen::Vector2d qq = x.head<2>(), ww = x.tail<2>();
            const auto [Mq, cq] = rehab_dynamics(qq, ww);
            const Eigen::Matrix2d Jq = rehab_jacobian(qq);
            const Eigen::Vector2d fe = hand(tk, rehab_fk(qq), Jq * ww);
            Eigen::VectorXd out(4);
            out << ww, Mq.partialPivLu().solve(tau + Jq.transpose() * fe - cq);
            return out;
        };
        Eigen::VectorXd x = X.row(k).transpose();
        for (int i = 0; i < substeps; ++i) x = rk4_step(rhs, x, h);
        X.row(k + 1) = x.transpose();
    }
    s.q = X.leftCols(2);
    s.dq = X.rightCols(2);
    s.p.resize(n + 1, 2);
    s.v.resize(n + 1, 2);
    for (Eigen::Index k = 0; k <= n; ++k) {
        const Eigen::Vector2d q = s.q.row(k).transpose(), dq = s.dq.row(k).transpose();
        s.p.row(k) = rehab_fk(q).transpose();
        s.v.row(k) = (rehab_jacobian(q) * dq).transpose();
    }
    return s;
}

// ---------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------

double spectral_radius(const Eigen::MatrixXd& A) {
    const Eigen::EigenSolver<Eigen::MatrixXd> es(A, false);
    return es.eigenvalues().cwiseAbs().maxCoeff();
}

double oscillation_growth(const DeviceSim& sim, double t_from) {
    const Eigen::VectorXd& t = sim.t;
    if (t.size() == 0 || sim.x.size() != t.size()) throw std::invalid_argument("oscillation_growth: empty or bad sim");
    const double tl = t(t.size() - 1), q = (tl - t_from) / 4.0;
    double a = -1.0, z = -1.0;
    for (Eigen::Index k = 0; k < t.size(); ++k) {
        const double ax = std::abs(sim.x(k));
        if (t(k) >= t_from && t(k) < t_from + q) a = std::max(a, ax);
        if (t(k) >= tl - q) z = std::max(z, ax);
    }
    if (a < 0.0 || z < 0.0) throw std::invalid_argument("oscillation_growth: run too short");
    return a > 0.0 ? z / a : 0.0;
}

HandForce pulse_force(const Eigen::Vector2d& direction, double amplitude, double t_on, double t_off) {
    const Eigen::Vector2d u = direction / direction.norm();
    return [u, amplitude, t_on, t_off](double t, const Eigen::Vector2d&, const Eigen::Vector2d&) -> Eigen::Vector2d {
        return (t_on <= t && t < t_off) ? (amplitude * u).eval() : Eigen::Vector2d::Zero().eval();
    };
}

Eigen::VectorXd one_dof_pulse_response(double m, double b, double k, double amplitude, double t_on, double t_off,
                                       const Eigen::VectorXd& t) {
    Eigen::MatrixXd Aug = Eigen::MatrixXd::Zero(3, 3);
    Aug(0, 1) = 1.0;
    Aug(1, 0) = -k / m;
    Aug(1, 1) = -b / m;
    Aug(1, 2) = amplitude / m;
    const auto step = [&Aug](double tau) {  // step response of the given amplitude, from rest
        return tau <= 0.0 ? 0.0 : expm(Aug * tau)(0, 2);
    };
    Eigen::VectorXd xs(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) xs(i) = step(t(i) - t_on) - step(t(i) - t_off);
    return xs;
}

// ---------------------------------------------------------------------------
// C++ only: expm, find_peaks, linear_fit
// ---------------------------------------------------------------------------

Eigen::MatrixXd expm(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) throw std::invalid_argument("expm: matrix must be square");
    const Eigen::Index n = A.rows();
    // Pade(13) scaling and squaring (Higham 2005), the method of scipy.linalg.expm for larger norms.
    static const double b[] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
                               129060195264000.0,   10559470521600.0,    670442572800.0,     33522128640.0,
                               1323241920.0,        40840800.0,          960960.0,           16380.0,
                               182.0,               1.0};
    const double norm1 = A.cwiseAbs().colwise().sum().maxCoeff();
    int s = 0;
    if (norm1 > 5.371920351148152) s = static_cast<int>(std::ceil(std::log2(norm1 / 5.371920351148152)));
    const Eigen::MatrixXd As = A / std::ldexp(1.0, s);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd A2 = As * As, A4 = A2 * A2, A6 = A4 * A2;
    const Eigen::MatrixXd U =
        As * (A6 * (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I);
    const Eigen::MatrixXd V = A6 * (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I;
    Eigen::MatrixXd R = (V - U).partialPivLu().solve(V + U);
    for (int i = 0; i < s; ++i) R = (R * R).eval();
    return R;
}

std::vector<Eigen::Index> find_peaks(const Eigen::VectorXd& x, int distance) {
    const Eigen::Index n = x.size();
    std::vector<Eigen::Index> peaks;
    Eigen::Index i = 1;
    while (i < n - 1) {  // local maxima; a flat peak gives its middle sample (scipy _local_maxima_1d)
        if (x(i - 1) < x(i)) {
            Eigen::Index ahead = i + 1;
            while (ahead < n - 1 && x(ahead) == x(i)) ++ahead;
            if (x(ahead) < x(i)) {
                peaks.push_back((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        ++i;
    }
    // remove peaks closer than `distance`, keeping the higher ones (scipy _select_by_peak_distance)
    const std::size_t np_ = peaks.size();
    const Eigen::Index d = std::max(1, distance);
    std::vector<std::size_t> order(np_);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t c) { return x(peaks[a]) < x(peaks[c]); });
    std::vector<bool> keep(np_, true);
    for (std::size_t r = np_; r-- > 0;) {
        const std::size_t j = order[r];
        if (!keep[j]) continue;
        for (std::size_t k = j; k-- > 0 && peaks[j] - peaks[k] < d;) keep[k] = false;
        for (std::size_t k = j + 1; k < np_ && peaks[k] - peaks[j] < d; ++k) keep[k] = false;
    }
    std::vector<Eigen::Index> out;
    for (std::size_t k = 0; k < np_; ++k)
        if (keep[k]) out.push_back(peaks[k]);
    return out;
}

std::pair<double, double> linear_fit(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    if (x.size() != y.size() || x.size() < 2) throw std::invalid_argument("linear_fit: need >= 2 points of equal length");
    const double mx = x.mean(), my = y.mean();
    const Eigen::VectorXd dx = x.array() - mx, dy = y.array() - my;
    const double slope = dx.dot(dy) / dx.squaredNorm();
    return {slope, my - slope * mx};
}

// ---------------------------------------------------------------------------
// Scalar overloads (given: they call your functions)
// ---------------------------------------------------------------------------

double handle_force_to_current(double F, const HapkitModel& model, double k_t) {
    return handle_force_to_current(Eigen::VectorXd::Constant(1, F), model, k_t)(0);
}

Eigen::Vector2d impedance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                         const Eigen::Vector2d& p_d, const Eigen::Vector2d& v_d, double K, double B) {
    const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    return impedance_control_torque(q, dq, p_d, v_d, (K * I).eval(), (B * I).eval());
}

Eigen::Vector2d admittance_control_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& dq,
                                          const Eigen::Vector2d& f_meas, const Eigen::Vector2d& p0, double M_d,
                                          double B_d, double K_d) {
    const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    return admittance_control_torque(q, dq, f_meas, p0, (M_d * I).eval(), (B_d * I).eval(), (K_d * I).eval());
}

}  // namespace lab
