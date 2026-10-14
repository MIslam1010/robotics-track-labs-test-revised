// SBEG 461 enrichment lab (CBF safety filter) -- provided functions (C++ track). Given code: do not edit; use
// these functions freely.
//
//   rehab_2r, robot                               the shared robot model (= course_models.py, numbers identical)
//   load_scenario                                 data/scenario.json
//   fk, jacobian, jdot_qdot, dynamics, ik         kinematics and closed-form dynamics of rehab_2r
//   quintic, person_state, reference             time scaling and the two motion profiles
//   peak_contact_force, metrics                   evaluation of a run
//   solve_qp                                      C++ only: small exact QP solver (replaces scipy.optimize.minimize)
//
// These are the "provided" functions of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double G0 = 9.81;
constexpr double R_ROD = 0.03;  // rod radius used by rod_inertia_x

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
sbeg::Matrix6d rod_inertia_x(double m, double L, double r = R_ROD) {
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

Eigen::Vector2d vec2(const sbeg::json& j) { return sbeg::as<Eigen::Vector2d>(j); }
}  // namespace

// ---------------------------------------------------------------------------
// Model and data
// ---------------------------------------------------------------------------

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
    r.limits << 0.0, PI, 0.2, 2.8;
    r.qdot_max = Eigen::Vector2d(3.0, 3.0);
    r.tau_max = Eigen::Vector2d(20.0, 12.0);
    r.q_home = Eigen::Vector2d(0.9, 1.6);
    r.params.L1 = L1;
    r.params.L2 = L2;
    r.params.m1 = m1;
    r.params.m2 = m2;
    return r;
}

const RehabModel& robot() {
    static const RehabModel R = rehab_2r();
    return R;
}

Scenario load_scenario(const std::string& name) {
    const sbeg::json data = sbeg::load_json(sbeg::lab_path("../data/scenario.json"));
    if (!data["scenarios"].contains(name)) throw std::invalid_argument("load_scenario: unknown scenario " + name);
    sbeg::json m = data["common"];
    for (const auto& item : data["scenarios"][name].items()) m[item.key()] = item.value();
    Scenario s;
    s.name = name;
    s.dt = m.at("dt").get<double>();
    s.T_move = m.at("T_move").get<double>();
    s.t_end = m.at("t_end").get<double>();
    s.p_A = vec2(m.at("p_A"));
    s.p_B = vec2(m.at("p_B"));
    s.Kp = m.at("Kp").get<double>();
    s.Kd = m.at("Kd").get<double>();
    s.r_keepout = m.at("r_keepout").get<double>();
    s.alpha1 = m.at("alpha1").get<double>();
    s.alpha2 = m.at("alpha2").get<double>();
    s.gamma = m.at("gamma").get<double>();
    s.v_max = m.at("v_max").get<double>();
    s.m_H = m.at("m_H").get<double>();
    s.k_contact = m.at("k_contact").get<double>();
    s.F_max = m.at("F_max").get<double>();
    s.person_p0 = vec2(m.at("person_p0"));
    s.person_p1 = vec2(m.at("person_p1"));
    s.person_t0 = m.at("person_t0").get<double>();
    s.person_t1 = m.at("person_t1").get<double>();
    return s;
}

// ---------------------------------------------------------------------------
// Kinematics and dynamics of rehab_2r
// ---------------------------------------------------------------------------

Eigen::Vector2d fk(const Eigen::Vector2d& q) {
    const double L1 = robot().params.L1, L2 = robot().params.L2;
    return Eigen::Vector2d(L1 * std::cos(q(0)) + L2 * std::cos(q(0) + q(1)),
                           L1 * std::sin(q(0)) + L2 * std::sin(q(0) + q(1)));
}

Eigen::Matrix2d jacobian(const Eigen::Vector2d& q) {
    const double L1 = robot().params.L1, L2 = robot().params.L2;
    const double s1 = std::sin(q(0)), c1 = std::cos(q(0)), s12 = std::sin(q(0) + q(1)), c12 = std::cos(q(0) + q(1));
    Eigen::Matrix2d J;
    J << -L1 * s1 - L2 * s12, -L2 * s12,
         L1 * c1 + L2 * c12, L2 * c12;
    return J;
}

Eigen::Vector2d jdot_qdot(const Eigen::Vector2d& q, const Eigen::Vector2d& qd) {
    const double L1 = robot().params.L1, L2 = robot().params.L2;
    const double w1 = qd(0), w12 = qd(0) + qd(1);
    return Eigen::Vector2d(-L1 * std::cos(q(0)) * w1 * w1 - L2 * std::cos(q(0) + q(1)) * w12 * w12,
                           -L1 * std::sin(q(0)) * w1 * w1 - L2 * std::sin(q(0) + q(1)) * w12 * w12);
}

std::pair<Eigen::Matrix2d, Eigen::Vector2d> dynamics(const Eigen::Vector2d& q, const Eigen::Vector2d& qd) {
    const RehabParams& P = robot().params;
    const double LC1 = P.L1 / 2, LC2 = P.L2 / 2;
    const double IZ1 = P.m1 * (3 * R_ROD * R_ROD + P.L1 * P.L1) / 12.0;
    const double IZ2 = P.m2 * (3 * R_ROD * R_ROD + P.L2 * P.L2) / 12.0;
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));
    const double m22 = IZ2 + P.m2 * LC2 * LC2;
    const double m12 = m22 + P.m2 * P.L1 * LC2 * c2;
    const double m11 = IZ1 + P.m1 * LC1 * LC1 + IZ2 + P.m2 * (P.L1 * P.L1 + LC2 * LC2 + 2 * P.L1 * LC2 * c2);
    const double h = P.m2 * P.L1 * LC2 * s2;
    Eigen::Matrix2d M;
    M << m11, m12, m12, m22;
    const Eigen::Vector2d c(-h * (2 * qd(0) * qd(1) + qd(1) * qd(1)), h * qd(0) * qd(0));
    return {M, c};
}

Eigen::Vector2d ik(const Eigen::Vector2d& p, double elbow) {
    const double L1 = robot().params.L1, L2 = robot().params.L2;
    const double x = p(0), y = p(1);
    const double c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    const double q2 = elbow * std::acos(std::min(1.0, std::max(-1.0, c2)));
    const double q1 = std::atan2(y, x) - std::atan2(L2 * std::sin(q2), L1 + L2 * std::cos(q2));
    return Eigen::Vector2d(q1, q2);
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

Quintic quintic(double t, double T) {
    if (t <= 0.0) return {0.0, 0.0, 0.0};
    if (t >= T) return {1.0, 0.0, 0.0};
    const double r = t / T;
    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
    return {10 * r3 - 15 * r4 + 6 * r5, (30 * r2 - 60 * r3 + 30 * r4) / T, (60 * r - 180 * r2 + 120 * r3) / (T * T)};
}

Profile person_state(double t, const Scenario& scn) {
    const Quintic s = quintic(t - scn.person_t0, scn.person_t1 - scn.person_t0);
    const Eigen::Vector2d D = scn.person_p1 - scn.person_p0;
    return {scn.person_p0 + s.s * D, s.sd * D, s.sdd * D};
}

Profile reference(double t, const Scenario& scn) {
    const Quintic s = quintic(t, scn.T_move);
    const Eigen::Vector2d D = scn.p_B - scn.p_A;
    return {scn.p_A + s.s * D, s.sd * D, s.sdd * D};
}

// ---------------------------------------------------------------------------
// Evaluation of a run
// ---------------------------------------------------------------------------

Eigen::VectorXd peak_contact_force(const SimLog& log, const Scenario& scn) {
    const Eigen::Index N = log.t.size();
    if (log.p.rows() != N || log.p_o.rows() != N || log.q.rows() != N || log.approach.size() != N)
        throw std::invalid_argument("peak_contact_force: the log arrays must all have one row per sample");
    Eigen::VectorXd F = Eigen::VectorXd::Zero(N);
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::Vector2d e = (log.p.row(k) - log.p_o.row(k)).transpose();
        const Eigen::Vector2d n = e / e.norm();
        const double mR = apparent_mass(log.q.row(k).transpose(), n);
        const double mu = 1.0 / (1.0 / mR + 1.0 / scn.m_H);
        F(k) = std::max(0.0, log.approach(k)) * std::sqrt(mu * scn.k_contact);
    }
    return F;
}

Metrics metrics(const SimLog& log, const Scenario& scn) {
    const Eigen::Index N = log.t.size();
    if (N == 0 || log.p.rows() != N || log.tau.rows() != N || log.d.size() != N || log.approach.size() != N ||
        static_cast<Eigen::Index>(log.active.size()) != N)
        throw std::invalid_argument("metrics: the log arrays must all have one row per sample (and N > 0)");
    Metrics m;
    m.min_d = log.d.minCoeff();
    m.max_approach = log.approach.maxCoeff();
    const Eigen::Vector2d tau_max = robot().tau_max;
    m.max_tau_ratio = (log.tau.cwiseAbs().array().rowwise() / tau_max.transpose().array()).maxCoeff();
    m.active_fraction =
        static_cast<double>(std::count(log.active.begin(), log.active.end(), true)) / static_cast<double>(N);
    m.final_error = (log.p.row(N - 1).transpose() - scn.p_B).norm();
    m.max_dev = 0.0;
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::Vector2d pd = reference(log.t(k), scn).p;
        m.max_dev = std::max(m.max_dev, (log.p.row(k).transpose() - pd).norm());
    }
    return m;
}

// ---------------------------------------------------------------------------
// Small QP solver (C++ only)
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, QPInfo> solve_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                                            const Eigen::MatrixXd& A_ub, const Eigen::VectorXd& b_ub) {
    const Eigen::Index n = f.size();
    if (n == 0 || H.rows() != n || H.cols() != n) throw std::invalid_argument("solve_qp: H must be n x n, f of length n");
    if (A_ub.size() > 0 && (A_ub.cols() != n || A_ub.rows() != b_ub.size()))
        throw std::invalid_argument("solve_qp: A_ub must be m x n and b_ub of length m");
    const int m = static_cast<int>(A_ub.size() > 0 ? A_ub.rows() : 0);
    if (n > 8 || m > 20) throw std::invalid_argument("solve_qp: problem too large for the lab solver (n <= 8, m <= 20)");
    QPInfo info;
    Eigen::VectorXd best = Eigen::VectorXd::Zero(n);
    double best_cost = std::numeric_limits<double>::infinity();
    // enumerate the active sets of size 0 ... min(n, m) (bit masks); keep the KKT point of lowest cost
    const int kmax = static_cast<int>(std::min<Eigen::Index>(n, m));
    for (unsigned mask = 0; mask < (1u << m); ++mask) {
        std::vector<Eigen::Index> act;
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
            const Eigen::Index row = act[static_cast<std::size_t>(a)];
            K.block(0, n + a, n, 1) = A_ub.row(row).transpose();
            K.block(n + a, 0, 1, n) = A_ub.row(row);
            r(n + a) = b_ub(row);
        }
        Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
        if (!lu.isInvertible()) continue;
        const Eigen::VectorXd sol = lu.solve(r);
        const Eigen::VectorXd z = sol.head(n);
        if (k > 0 && sol.tail(k).minCoeff() < -1e-10) continue;  // multipliers of active rows must be >= 0
        bool feasible = true;
        for (int i = 0; i < m && feasible; ++i) feasible = A_ub.row(i).dot(z) <= b_ub(i) + 1e-9;
        if (!feasible) continue;
        const double cost = 0.5 * z.dot(H * z) + f.dot(z);
        if (cost < best_cost) {
            best_cost = cost;
            best = z;
            info.success = true;
        }
    }
    info.fun = info.success ? best_cost : 0.0;
    return {best, info};
}

}  // namespace lab
