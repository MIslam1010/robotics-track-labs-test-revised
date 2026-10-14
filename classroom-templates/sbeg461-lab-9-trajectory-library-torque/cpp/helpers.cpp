// SBEG 461 Lab 9 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   skew, trans_inv, adjoint, ad, matrix_exp6, rnea      the small SE(3) helpers and the RNEA of Labs 2/4
//   load_course_arm, course_arm_model, load_tasks,        data/course_arm.json, data/tasks.json,
//   load_topp_task, data_path, export_csv, write_log      data/topp_task.json; CSV output
//   trapezoid_from_limits, heuristic_via_velocities,      the provided functions of Parts A-C
//   bspline_basis, check_limits, scalar overloads of the three time scalings
//   paths, models, inverse_dynamics, quintic_traj,        the provided functions of Parts D-E
//   peak_usage, command_table, check_command
//
// These are the "Provided" functions of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;

/// np.linspace(a, b, num): a + i * (b - a) / (num - 1), the last point exactly b.
Eigen::VectorXd linspace(double a, double b, Eigen::Index num) {
    Eigen::VectorXd y(num);
    const double step = num > 1 ? (b - a) / static_cast<double>(num - 1) : 0.0;
    for (Eigen::Index i = 0; i < num; ++i) y(i) = static_cast<double>(i) * step + a;
    if (num > 1) y(num - 1) = b;
    return y;
}

/// Natural cubic spline through (x_k, Y.row(k)) (scipy CubicSpline(bc_type="natural")): value and derivatives.
struct NaturalSpline {
    Eigen::VectorXd x;
    Eigen::MatrixXd Y, M2;  // M2: second derivatives at the knots (one row per knot)
    Eigen::VectorXd operator()(double s, int order) const {
        const Eigen::Index K = x.size();
        Eigen::Index k = static_cast<Eigen::Index>(std::upper_bound(x.data(), x.data() + K, s) - x.data()) - 1;
        k = std::min<Eigen::Index>(std::max<Eigen::Index>(k, 0), K - 2);
        const double h = x(k + 1) - x(k), a = x(k + 1) - s, b = s - x(k);
        const Eigen::VectorXd m0 = M2.row(k).transpose(), m1 = M2.row(k + 1).transpose();
        const Eigen::VectorXd y0 = Y.row(k).transpose(), y1 = Y.row(k + 1).transpose();
        if (order == 0)
            return m0 * (a * a * a / (6 * h)) + m1 * (b * b * b / (6 * h)) + (y0 / h - m0 * (h / 6)) * a +
                   (y1 / h - m1 * (h / 6)) * b;
        if (order == 1)
            return -m0 * (a * a / (2 * h)) + m1 * (b * b / (2 * h)) - (y0 / h - m0 * (h / 6)) +
                   (y1 / h - m1 * (h / 6));
        if (order == 2) return m0 * (a / h) + m1 * (b / h);
        throw std::invalid_argument("path order must be 0, 1 or 2");
    }
};

Eigen::Matrix4d trans_p(double x, double y, double z) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
}

sbeg::Matrix6d diag6(double a, double b, double c, double d, double e, double f) {
    sbeg::Vector6d v;
    v << a, b, c, d, e, f;
    return sbeg::Matrix6d(v.asDiagonal());
}

Robot arm_from_json(const sbeg::json& d, const char* glist_key, double kf, double tf) {
    Robot r;
    r.name = "course arm";
    r.n = 6;
    r.Slist = sbeg::as_mat(d["Slist"]);
    r.M = sbeg::as<Eigen::Matrix4d>(d["M"]);
    r.Mlist = sbeg::as_list<Eigen::Matrix4d>(d["Mlist"]);
    r.Glist = sbeg::as_list<sbeg::Matrix6d>(d[glist_key]);
    r.g = sbeg::as<Eigen::Vector3d>(d["g"]);
    r.qdot_max = kf * sbeg::as_vec(d["qdot_max"]);
    r.tau_max = tf * sbeg::as_vec(d["tau_max"]);
    r.limits = sbeg::as_mat(d["limits"]);
    r.q_home = sbeg::as_vec(d["q_home"]);
    return r;
}
}  // namespace

// ---------------------------------------------------------------------------
// SE(3) helpers and the RNEA
// ---------------------------------------------------------------------------

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d m;
    m << 0.0, -w(2), w(1),
         w(2), 0.0, -w(0),
         -w(1), w(0), 0.0;
    return m;
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.topLeftCorner<3, 3>() = R.transpose();
    out.block<3, 1>(0, 3) = -R.transpose() * p;
    return out;
}

sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    sbeg::Matrix6d Ad = sbeg::Matrix6d::Zero();
    Ad.topLeftCorner<3, 3>() = R;
    Ad.bottomLeftCorner<3, 3>() = skew(p) * R;
    Ad.bottomRightCorner<3, 3>() = R;
    return Ad;
}

sbeg::Matrix6d ad(const sbeg::Vector6d& V) {
    sbeg::Matrix6d out = sbeg::Matrix6d::Zero();
    const Eigen::Matrix3d W = skew(V.head<3>());
    out.topLeftCorner<3, 3>() = W;
    out.bottomLeftCorner<3, 3>() = skew(V.tail<3>());
    out.bottomRightCorner<3, 3>() = W;
    return out;
}

Eigen::Matrix4d matrix_exp6(const sbeg::Vector6d& S, double theta) {
    const Eigen::Vector3d w = S.head<3>(), v = S.tail<3>();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (w.norm() < 1e-12) {
        T.block<3, 1>(0, 3) = v * theta;
        return T;
    }
    const Eigen::Matrix3d W = skew(w);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R = I + std::sin(theta) * W + (1.0 - std::cos(theta)) * W * W;
    const Eigen::Matrix3d G = I * theta + (1.0 - std::cos(theta)) * W + (theta - std::sin(theta)) * W * W;
    T.topLeftCorner<3, 3>() = R;
    T.block<3, 1>(0, 3) = G * v;
    return T;
}

Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const sbeg::Vector6d& Ftip,
                     const mr::TransformList& Mlist, const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist) {
    const Eigen::Index n = thetalist.size();
    const std::size_t nn = static_cast<std::size_t>(n);
    if (dthetalist.size() != n || ddthetalist.size() != n || Slist.rows() != 6 || Slist.cols() != n ||
        Mlist.size() != nn + 1 || Glist.size() != nn)
        throw std::invalid_argument("rnea: inconsistent sizes of thetalist, dthetalist, ddthetalist, Mlist, Glist, Slist");
    Eigen::Matrix4d Mi = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd Ai = Eigen::MatrixXd::Zero(6, n);
    std::vector<sbeg::Matrix6d> AdTi(nn + 1);
    Eigen::MatrixXd Vi = Eigen::MatrixXd::Zero(6, n + 1);
    Eigen::MatrixXd Vdi = Eigen::MatrixXd::Zero(6, n + 1);
    Vdi.block<3, 1>(3, 0) = -g;
    AdTi[nn] = adjoint(trans_inv(Mlist[nn]));
    sbeg::Vector6d Fi = Ftip;
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        Mi = Mi * Mlist[k];
        Ai.col(i) = adjoint(trans_inv(Mi)) * Slist.col(i);
        const sbeg::Vector6d A = Ai.col(i);
        const Eigen::Matrix4d Ti = matrix_exp6(-A, thetalist(i)) * trans_inv(Mlist[k]);
        AdTi[k] = adjoint(Ti);
        Vi.col(i + 1) = AdTi[k] * Vi.col(i) + A * dthetalist(i);
        const sbeg::Vector6d V = Vi.col(i + 1);
        Vdi.col(i + 1) = AdTi[k] * Vdi.col(i) + A * ddthetalist(i) + ad(V) * A * dthetalist(i);
    }
    for (Eigen::Index i = n - 1; i >= 0; --i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const sbeg::Vector6d V = Vi.col(i + 1), Vd = Vdi.col(i + 1);
        const sbeg::Vector6d F = AdTi[k + 1].transpose() * Fi + Glist[k] * Vd - ad(V).transpose() * (Glist[k] * V);
        Fi = F;
        tau(i) = Fi.dot(Ai.col(i));
    }
    return tau;
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

Robot load_course_arm(bool payload, bool derated) {
    const sbeg::json d = sbeg::load_json(sbeg::lab_path("../data/course_arm.json"));
    const double kf = derated ? d["derated_speed_fraction"].get<double>() : 1.0;
    const double tf = derated ? d["derated_current_fraction"].get<double>() : 1.0;
    return arm_from_json(d, payload ? "Glist_payload" : "Glist", kf, tf);
}

Robot course_arm_model(double payload_kg) {
    const sbeg::json d = sbeg::load_json(sbeg::lab_path("../data/course_arm.json"));
    Robot r = arm_from_json(d, "Glist", 1.0, 1.0);
    if (payload_kg != 0.0) {
        const sbeg::Matrix6d Ad = adjoint(trans_inv(r.Mlist.back()));
        const sbeg::Matrix6d Gp = Ad.transpose() * diag6(0, 0, 0, payload_kg, payload_kg, payload_kg) * Ad;
        r.Glist.back() = r.Glist.back() + Gp;
    }
    return r;
}

Tasks load_tasks() {
    const sbeg::json t = sbeg::load_json(sbeg::lab_path("../data/tasks.json"));
    Tasks out;
    out.A = sbeg::as_vec(t["A"]);
    out.B = sbeg::as_vec(t["B"]);
    out.C = sbeg::as_vec(t["C"]);
    out.D = sbeg::as_vec(t["D"]);
    out.W = sbeg::as_vec(t["W"]);
    out.via_times = sbeg::as_vec(t["via_times"]);
    out.qddot_lim = sbeg::as_vec(t["qddot_lim"]);
    out.payload_kg = t["payload_kg"].get<double>();
    out.description = t["description"].get<std::string>();
    return out;
}

const Eigen::VectorXd& Tasks::config(const std::string& name) const {
    if (name == "A") return A;
    if (name == "B") return B;
    if (name == "C") return C;
    if (name == "D") return D;
    if (name == "W") return W;
    throw std::invalid_argument("Tasks::config: no task configuration named '" + name + "'");
}

std::string data_path(const std::string& name) { return sbeg::lab_path("../data/" + name); }

ToppTask load_topp_task(const std::string& name) {
    const sbeg::json j = sbeg::load_json(data_path(name));
    ToppTask t;
    t.description = j.value("description", std::string());
    t.waypoints = sbeg::as_mat(j.at("waypoints"));
    t.payload_kg = j.at("payload_kg").get<double>();
    t.torque_fraction = j.at("torque_fraction").get<double>();
    t.sim_speed_fraction = j.at("sim_speed_fraction").get<double>();
    t.hw_speed_fraction = j.at("hw_speed_fraction").get<double>();
    t.first_run_speed_fraction = j.at("first_run_speed_fraction").get<double>();
    t.hw_qddot_max = j.at("hw_qddot_max").get<double>();
    t.servo_period_s = j.at("servo_period_s").get<double>();
    t.joint_margin_deg = j.at("joint_margin_deg").get<double>();
    t.grid_N = j.at("grid_N").get<int>();
    return t;
}

void export_csv(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& Q,
                const Eigen::MatrixXd& Qd) {
    if (Q.rows() != t.size() || Qd.rows() != t.size() || Qd.cols() != Q.cols())
        throw std::invalid_argument("export_csv: t, Q and Qd must have the same number of rows");
    const Eigen::Index n = Q.cols();
    std::vector<std::string> header{"t"};
    for (Eigen::Index i = 0; i < n; ++i) header.push_back("q" + std::to_string(i + 1));
    for (Eigen::Index i = 0; i < n; ++i) header.push_back("qd" + std::to_string(i + 1));
    Eigen::MatrixXd data(t.size(), 1 + 2 * n);
    data << t, Q, Qd;
    sbeg::write_csv(path, header, data);
}

void write_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd) {
    if (cmd.rows() != t.size()) throw std::invalid_argument("write_log: one row of cmd per time");
    std::vector<std::string> header{"t"};
    for (Eigen::Index j = 0; j < cmd.cols(); ++j) header.push_back("cmd_" + std::to_string(j + 1));
    Eigen::MatrixXd data(t.size(), 1 + cmd.cols());
    data << t, cmd;
    sbeg::write_csv(path, header, data);
}

// ---------------------------------------------------------------------------
// Parts A-C: provided functions and the scalar / vector overloads (they call the functions you write)
// ---------------------------------------------------------------------------

namespace {
TimeScalingPoint first_sample(const TimeScaling& r) {
    if (r.s.size() != 1 || r.sd.size() != 1 || r.sdd.size() != 1)
        throw std::invalid_argument("time scaling: one time in, but not one (s, sdot, sddot) out");
    return {r.s(0), r.sd(0), r.sdd(0)};
}
Eigen::VectorXd one(double t) { return Eigen::VectorXd::Constant(1, t); }
}  // namespace

TimeScalingPoint cubic_time_scaling(double T, double t) { return first_sample(cubic_time_scaling(T, one(t))); }

TimeScalingPoint quintic_time_scaling(double T, double t) { return first_sample(quintic_time_scaling(T, one(t))); }

TimeScalingPoint trapezoid_time_scaling(double v, double a, double t) {
    return first_sample(trapezoid_time_scaling(v, a, one(t)));
}

TrapezoidLimits trapezoid_from_limits(const Eigen::VectorXd& theta_start, const Eigen::VectorXd& theta_end,
                                      const Eigen::VectorXd& qdot_lim, const Eigen::VectorXd& qddot_lim) {
    const Eigen::VectorXd D = (theta_end - theta_start).cwiseAbs();
    if (qdot_lim.size() != D.size() || qddot_lim.size() != D.size())
        throw std::invalid_argument("trapezoid_from_limits: one limit per joint");
    double v = std::numeric_limits<double>::infinity(), a = v;
    bool any = false;
    for (Eigen::Index i = 0; i < D.size(); ++i) {
        if (D(i) > 1e-12) {
            v = std::min(v, qdot_lim(i) / D(i));
            a = std::min(a, qddot_lim(i) / D(i));
            any = true;
        }
    }
    if (!any) throw std::invalid_argument("trapezoid_from_limits: no joint moves");
    return {v, a, trapezoid_params(v, a).T};
}

Eigen::MatrixXd heuristic_via_velocities(const Eigen::MatrixXd& points, const Eigen::VectorXd& times) {
    const Eigen::Index k = points.rows();
    if (times.size() != k) throw std::invalid_argument("heuristic_via_velocities: one time per via");
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(k, points.cols());
    for (Eigen::Index i = 1; i + 1 < k; ++i) {
        for (Eigen::Index c = 0; c < points.cols(); ++c) {
            const double m1 = (points(i, c) - points(i - 1, c)) / (times(i) - times(i - 1));
            const double m2 = (points(i + 1, c) - points(i, c)) / (times(i + 1) - times(i));
            V(i, c) = m1 * m2 > 0.0 ? 0.5 * (m1 + m2) : 0.0;
        }
    }
    return V;
}

Eigen::Vector4d bspline_basis(double u) {
    const double w = 1.0 - u;
    return Eigen::Vector4d(w * w * w, 3.0 * u * u * u - 6.0 * u * u + 4.0, -3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0,
                           u * u * u) / 6.0;
}

Eigen::MatrixXd bspline_basis(const Eigen::VectorXd& u) {
    Eigen::MatrixXd b(4, u.size());
    for (Eigen::Index k = 0; k < u.size(); ++k) b.col(k) = bspline_basis(u(k));
    return b;
}

LimitCheck check_limits(const Robot& robot, const Eigen::MatrixXd& Qd, const Eigen::MatrixXd& Tau, double tol) {
    if (Qd.cols() != robot.qdot_max.size() || Tau.cols() != robot.tau_max.size())
        throw std::invalid_argument("check_limits: one column per joint (Qd, Tau) and one limit per joint");
    LimitCheck r;
    r.vel_ratio = (Qd.cwiseAbs().array().rowwise() / robot.qdot_max.transpose().array()).colwise().maxCoeff().transpose();
    r.tau_ratio = (Tau.cwiseAbs().array().rowwise() / robot.tau_max.transpose().array()).colwise().maxCoeff().transpose();
    r.ok = (r.vel_ratio.array() <= 1.0 + tol).all() && (r.tau_ratio.array() <= 1.0 + tol).all();
    return r;
}

// ---------------------------------------------------------------------------
// Part D: paths, models, inverse_dynamics, quintic_traj, peak_usage
// ---------------------------------------------------------------------------

Path straight_line_path(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1) {
    if (q0.size() != q1.size()) throw std::invalid_argument("straight_line_path: q0 and q1 differ in size");
    const Eigen::VectorXd d = q1 - q0;
    return [q0, d](double s, int order) -> Eigen::VectorXd {
        if (order == 0) return q0 + s * d;
        if (order == 1) return d;
        return Eigen::VectorXd::Zero(d.size());
    };
}

Path semicircle_path(double R, double xc, double yc) {
    return [R, xc, yc](double s, int order) -> Eigen::VectorXd {
        const double a = PI * s;
        Eigen::VectorXd out(2);
        if (order == 0) out << xc + R * std::cos(a), yc - R * std::sin(a);
        else if (order == 1) out << -PI * R * std::sin(a), -PI * R * std::cos(a);
        else out << -PI * PI * R * std::cos(a), PI * PI * R * std::sin(a);
        return out;
    };
}

Path waypoint_path(const Eigen::MatrixXd& waypoints) {
    const Eigen::Index K = waypoints.rows();
    if (K < 2) throw std::invalid_argument("waypoint_path: at least two waypoints are needed");
    NaturalSpline sp;
    sp.x = linspace(0.0, 1.0, K);
    sp.Y = waypoints;
    sp.M2 = Eigen::MatrixXd::Zero(K, waypoints.cols());
    if (K > 2) {  // tridiagonal system for the interior second derivatives (natural ends: M2 = 0)
        const Eigen::Index m = K - 2;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, m), rhs(m, waypoints.cols());
        for (Eigen::Index i = 0; i < m; ++i) {
            const Eigen::Index k = i + 1;
            const double h0 = sp.x(k) - sp.x(k - 1), h1 = sp.x(k + 1) - sp.x(k);
            A(i, i) = 2.0 * (h0 + h1);
            if (i > 0) A(i, i - 1) = h0;
            if (i + 1 < m) A(i, i + 1) = h1;
            rhs.row(i) = 6.0 * ((waypoints.row(k + 1) - waypoints.row(k)) / h1 -
                                (waypoints.row(k) - waypoints.row(k - 1)) / h0);
        }
        sp.M2.middleRows(1, m) = A.partialPivLu().solve(rhs);
    }
    return sp;
}

Robot mr_2r_model(double L1, double L2, double m1, double m2, double g) {
    Robot r;
    r.name = "MR 2R";
    r.n = 2;
    r.Mlist = {trans_p(L1, 0, 0), trans_p(L2, 0, 0), Eigen::Matrix4d::Identity()};
    r.Glist = {diag6(0, 0, 0, m1, m1, m1), diag6(0, 0, 0, m2, m2, m2)};
    r.Slist.resize(6, 2);
    r.Slist << 0, 0, 0, 0, 1, 1, 0, 0, 0, -L1, 0, 0;  // columns (0,0,1,0,0,0) and (0,0,1,0,-L1,0)
    r.g = Eigen::Vector3d(0.0, -g, 0.0);
    return r;
}

Robot elbow_exo_model() {
    const double m = 1.6 + 0.4 + 1.0;
    const double mr_sum = 1.6 * 0.18 + 0.4 * 0.15 + 1.0 * 0.35;
    const double I_elbow = 0.012 + 1.6 * 0.18 * 0.18 + 0.002 + 0.4 * 0.15 * 0.15 + 1.0 * 0.35 * 0.35;
    const double rc = mr_sum / m;
    const double Ic = I_elbow - m * rc * rc;
    Robot r;
    r.name = "elbow exoskeleton";
    r.n = 1;
    r.Mlist = {trans_p(0, -rc, 0), trans_p(0, -(0.35 - rc), 0)};
    r.Glist = {diag6(Ic, Ic, Ic, m, m, m)};
    r.Slist.resize(6, 1);
    r.Slist << 0, 0, 1, 0, 0, 0;
    r.g = Eigen::Vector3d(0.0, -9.81, 0.0);
    return r;
}

Robot gantry_model(double Mx, double my) {
    Robot r;
    r.name = "gantry";
    r.n = 2;
    r.Mlist = {Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity()};
    r.Glist = {diag6(0, 0, 0, Mx - my, Mx - my, Mx - my), diag6(0, 0, 0, my, my, my)};
    r.Slist = Eigen::MatrixXd::Zero(6, 2);
    r.Slist(3, 0) = 1.0;
    r.Slist(4, 1) = 1.0;
    r.g = Eigen::Vector3d::Zero();
    return r;
}

Eigen::VectorXd inverse_dynamics(const Robot& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& ddq, bool gravity) {
    const Eigen::Vector3d g = gravity ? model.g : Eigen::Vector3d::Zero();
    return rnea(q, dq, ddq, g, sbeg::Vector6d::Zero(), model.Mlist, model.Glist, model.Slist);
}

TrajectoryFn quintic_traj(const Path& path, double T) {
    return [path, T](double t, int order) -> Eigen::VectorXd {
        const TimeScalingPoint q = quintic_time_scaling(T, t);
        if (order == 0) return path(q.s, 0);
        const Eigen::VectorXd d1 = path(q.s, 1);
        if (order == 1) return d1 * q.sd;
        return d1 * q.sdd + path(q.s, 2) * (q.sd * q.sd);
    };
}

PeakUsage peak_usage(const Robot& model, const TrajectoryFn& traj, double T, const Eigen::VectorXd& tau_max,
                     const Eigen::VectorXd& qdot_max, int N) {
    PeakUsage out;
    out.t = linspace(0.0, T, N + 1);
    const Eigen::Index K = out.t.size();
    Eigen::MatrixXd qd, qdd;
    for (Eigen::Index k = 0; k < K; ++k) {
        const Eigen::VectorXd q = traj(out.t(k), 0), v = traj(out.t(k), 1), a = traj(out.t(k), 2);
        if (k == 0) {
            qd.resize(K, q.size());
            qdd.resize(K, q.size());
            out.tau.resize(K, q.size());
        }
        qd.row(k) = v.transpose();
        qdd.row(k) = a.transpose();
        out.tau.row(k) = inverse_dynamics(model, q, v, a).transpose();
    }
    out.vel_ratio = qd.cwiseAbs().colwise().maxCoeff().transpose().cwiseQuotient(qdot_max);
    out.tau_ratio = out.tau.cwiseAbs().colwise().maxCoeff().transpose().cwiseQuotient(tau_max);
    double jm = 0.0;
    for (Eigen::Index k = 0; k + 1 < K; ++k)
        jm = std::max(jm, ((qdd.row(k + 1) - qdd.row(k)) / (out.t(k + 1) - out.t(k))).cwiseAbs().maxCoeff());
    out.jerk_max = jm;
    return out;
}

// ---------------------------------------------------------------------------
// Part E: command_table, check_command
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> command_table(const TrajectoryFn& traj, double T, double dt,
                                                          double slow) {
    const double Tc = slow * T;
    const auto nstep = static_cast<Eigen::Index>(std::ceil(Tc / dt - 1e-9));
    Eigen::VectorXd t(nstep + 1);
    for (Eigen::Index k = 0; k <= nstep; ++k) t(k) = std::min(static_cast<double>(k) * dt, Tc);
    Eigen::MatrixXd q;
    for (Eigen::Index k = 0; k <= nstep; ++k) {
        const Eigen::VectorXd qk = traj(t(k) / slow, 0);
        if (k == 0) q.resize(nstep + 1, qk.size());
        q.row(k) = qk.transpose();
    }
    return {t, q};
}

CommandCheck check_command(const Eigen::VectorXd& t, const Eigen::MatrixXd& q, const Robot& model,
                           double speed_fraction, double margin_deg, double tol) {
    if (q.rows() != t.size() || q.rows() < 2) throw std::invalid_argument("check_command: one row of q per time, >= 2");
    const double m = margin_deg * (PI / 180.0);
    bool within = true;
    for (Eigen::Index k = 0; k < q.rows(); ++k)
        for (Eigen::Index j = 0; j < q.cols(); ++j)
            if (!(q(k, j) >= model.limits(j, 0) + m) || !(q(k, j) <= model.limits(j, 1) - m)) within = false;
    double ratio = 0.0;
    Eigen::VectorXd steps(q.rows() - 1);
    for (Eigen::Index k = 0; k + 1 < q.rows(); ++k) {
        const Eigen::RowVectorXd dq = q.row(k + 1) - q.row(k);
        steps(k) = dq.cwiseAbs().maxCoeff();
        for (Eigen::Index j = 0; j < q.cols(); ++j)
            ratio = std::max(ratio, std::abs(dq(j) / (t(k + 1) - t(k))) / (speed_fraction * model.qdot_max(j)));
    }
    CommandCheck out;
    out.within_limits = within;
    out.peak_speed_ratio = ratio;
    out.starts_at_rest = steps(0) < 1e-3;
    out.ends_at_rest = steps(steps.size() - 1) < 1e-3;
    out.speed_ok = ratio <= 1.0 + tol;
    out.ok = within && out.speed_ok && out.starts_at_rest && out.ends_at_rest;
    return out;
}

}  // namespace lab
