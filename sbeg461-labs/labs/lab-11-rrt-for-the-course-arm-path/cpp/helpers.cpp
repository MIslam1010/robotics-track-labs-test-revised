// SBEG 461 Lab 11 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_course_arm, load_scene                          data/course_arm.json, data/scenes.json
//   exp6, poe_frames                                     PoE of the course arm (one configuration)
//   segment_point_distance, within_limits(_batch), motion_bound
//   trapezoid_params, trapezoid_time_scaling, trapezoid_from_limits, check_limits   (Lab 9)
//   rnea                                                 mr::inverse_dynamics (or paste your Lab 2 RNEA)
//   plan_and_scale, export_results                       pipeline for the experiments, CSV output for plotting
//
// These are the "Provided" functions of starter.py, translated to C++.
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

// ---------------------------------------------------------------------------
// Robot and scenes
// ---------------------------------------------------------------------------

Arm load_course_arm(bool payload, bool derated) {
    const sbeg::json d = sbeg::load_json(sbeg::lab_path("../data/course_arm.json"));
    const double kf = derated ? d["derated_speed_fraction"].get<double>() : 1.0;
    const double tf = derated ? d["derated_current_fraction"].get<double>() : 1.0;
    Arm arm;
    arm.Slist = sbeg::as_mat(d["Slist"]);
    arm.M = sbeg::as<Eigen::Matrix4d>(d["M"]);
    arm.Mlist = sbeg::as_list<Eigen::Matrix4d>(d["Mlist"]);
    arm.Glist = sbeg::as_list<sbeg::Matrix6d>(payload ? d["Glist_payload"] : d["Glist"]);
    arm.g = sbeg::as<Eigen::Vector3d>(d["g"]);
    arm.limits = sbeg::as_mat(d["limits"]);
    arm.qdot_max = kf * sbeg::as_vec(d["qdot_max"]);
    arm.qddot_max = sbeg::as_vec(d["qddot_max"]);
    arm.tau_max = tf * sbeg::as_vec(d["tau_max"]);
    arm.q_home = sbeg::as_vec(d["q_home"]);
    arm.skeleton = sbeg::as_mat(d["skeleton"]);
    arm.skeleton_link.clear();
    for (const auto& e : d["skeleton_link"]) arm.skeleton_link.push_back(e.get<int>());
    arm.link_radius = sbeg::as_vec(d["link_radius"]);
    arm.rho = sbeg::as_vec(d["rho"]);
    return arm;
}

namespace {
Eigen::MatrixXd rows_of_4(const sbeg::json& j) {
    // Python: np.asarray(s[...], float).reshape(-1, 4)  (an empty list gives 0 x 4)
    if (j.empty()) return Eigen::MatrixXd(0, 4);
    return sbeg::as_mat(j);
}
}  // namespace

Scene load_scene(const std::string& name) {
    const sbeg::json all = sbeg::load_json(sbeg::lab_path("../data/scenes.json"));
    if (!all.contains(name)) throw std::invalid_argument("unknown scene: " + name);
    const sbeg::json& s = all[name];
    Scene sc;
    sc.name = name;
    sc.description = s.value("description", std::string());
    sc.spheres = rows_of_4(s["spheres"]);
    sc.planes = rows_of_4(s["planes"]);
    sc.q_start = sbeg::as_vec(s["q_start"]);
    sc.q_goal = sbeg::as_vec(s["q_goal"]);
    sc.sample_limits = sbeg::as_mat(s["sample_limits"]);
    return sc;
}

// ---------------------------------------------------------------------------
// Product of exponentials
// ---------------------------------------------------------------------------

Eigen::Matrix4d exp6(const sbeg::Vector6d& S, double theta) {
    const Eigen::Vector3d w = S.head<3>(), v = S.tail<3>();
    Eigen::Matrix3d W;
    W << 0.0, -w(2), w(1),
         w(2), 0.0, -w(0),
         -w(1), w(0), 0.0;
    const Eigen::Matrix3d W2 = W * W;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const double s = std::sin(theta), c = std::cos(theta);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = I + s * W + (1.0 - c) * W2;
    const Eigen::Matrix3d G = theta * I + (1.0 - c) * W + (theta - s) * W2;
    T.block<3, 1>(0, 3) = G * v;
    return T;
}

std::vector<Eigen::Matrix4d> poe_frames(const Arm& arm, const Eigen::VectorXd& q) {
    const Eigen::Index n = arm.Slist.cols();
    if (q.size() != n) throw std::invalid_argument("poe_frames: q must have " + std::to_string(n) + " entries");
    std::vector<Eigen::Matrix4d> out(static_cast<std::size_t>(n + 1));
    out[0] = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 0; i < n; ++i) {
        const sbeg::Vector6d S = arm.Slist.col(i);
        const std::size_t k = static_cast<std::size_t>(i);
        out[k + 1] = out[k] * exp6(S, q(i));
    }
    return out;
}

double segment_point_distance(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
    const Eigen::Vector3d ab = b - a;
    const double den = std::max(ab.squaredNorm(), 1e-15);
    const double t = std::clamp((c - a).dot(ab) / den, 0.0, 1.0);
    return (a + t * ab - c).norm();
}

bool within_limits(const Arm& arm, const Eigen::VectorXd& q) {
    if (q.size() != arm.limits.rows()) throw std::invalid_argument("within_limits: wrong number of joints");
    for (Eigen::Index i = 0; i < q.size(); ++i)
        if (q(i) < arm.limits(i, 0) - 1e-12 || q(i) > arm.limits(i, 1) + 1e-12) return false;
    return true;
}

std::vector<bool> within_limits_batch(const Arm& arm, const Eigen::MatrixXd& Q) {
    std::vector<bool> ok;
    ok.reserve(static_cast<std::size_t>(Q.rows()));
    for (Eigen::Index k = 0; k < Q.rows(); ++k) ok.push_back(within_limits(arm, Q.row(k).transpose()));
    return ok;
}

double motion_bound(const Arm& arm, const Eigen::VectorXd& dq) { return arm.rho.dot(dq.cwiseAbs()); }

// ---------------------------------------------------------------------------
// Lab 9 time-scaling helpers
// ---------------------------------------------------------------------------

TrapezoidParams trapezoid_params(double v, double a) {
    if (v * v / a <= 1.0) return {(a + v * v) / (v * a), v / a, v};
    const double ta = 1.0 / std::sqrt(a);
    return {2.0 * ta, ta, std::sqrt(a)};
}

TimeScaling trapezoid_time_scaling(double v, double a, const Eigen::VectorXd& t_in) {
    const TrapezoidParams P = trapezoid_params(v, a);
    const double T = P.T, ta = P.t_a, vp = P.v_peak;
    TimeScaling out;
    out.s.resize(t_in.size());
    out.sd.resize(t_in.size());
    out.sdd.resize(t_in.size());
    for (Eigen::Index k = 0; k < t_in.size(); ++k) {
        const double t = std::clamp(t_in(k), 0.0, T);
        if (t <= ta) {
            out.s(k) = 0.5 * a * t * t;
            out.sd(k) = a * t;
        } else if (t <= T - ta) {
            out.s(k) = vp * t - vp * vp / (2.0 * a);
            out.sd(k) = vp;
        } else {
            out.s(k) = (2.0 * a * vp * T - 2.0 * vp * vp - a * a * (t - T) * (t - T)) / (2.0 * a);
            out.sd(k) = a * (T - t);
        }
        out.sdd(k) = t < ta ? a : (t <= T - ta ? 0.0 : -a);
    }
    return out;
}

TrapezoidLimits trapezoid_from_limits(const Eigen::VectorXd& theta_start, const Eigen::VectorXd& theta_end,
                                      const Eigen::VectorXd& qdot_lim, const Eigen::VectorXd& qddot_lim) {
    const Eigen::VectorXd D = (theta_end - theta_start).cwiseAbs();
    double v = std::numeric_limits<double>::infinity(), a = v;
    for (Eigen::Index i = 0; i < D.size(); ++i) {
        if (D(i) <= 1e-12) continue;
        v = std::min(v, qdot_lim(i) / D(i));
        a = std::min(a, qddot_lim(i) / D(i));
    }
    if (!std::isfinite(v)) throw std::invalid_argument("trapezoid_from_limits: no joint moves");
    return {v, a, trapezoid_params(v, a).T};
}

Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const sbeg::Vector6d& Ftip,
                     const mr::TransformList& Mlist, const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist) {
    return mr::inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist);
}

LimitCheck check_limits(const Arm& arm, const Eigen::MatrixXd& Qd, const Eigen::MatrixXd& Tau, double tol) {
    LimitCheck r;
    r.vel_ratio = Qd.cwiseAbs().colwise().maxCoeff().transpose().cwiseQuotient(arm.qdot_max);
    r.tau_ratio = Tau.cwiseAbs().colwise().maxCoeff().transpose().cwiseQuotient(arm.tau_max);
    r.ok = (r.vel_ratio.array() <= 1.0 + tol).all() && (r.tau_ratio.array() <= 1.0 + tol).all();
    return r;
}

// ---------------------------------------------------------------------------
// Pipeline and CSV output
// ---------------------------------------------------------------------------

PipelineResult plan_and_scale(const Arm& arm, const Scene& scene, unsigned seed, const std::string& planner,
                              const std::string& smooth) {
    if (planner != "rrt" && planner != "rrt_star") throw std::invalid_argument("planner must be rrt or rrt_star");
    Rng rng(seed);
    PipelineResult out;
    out.planner = planner;
    Eigen::MatrixXd path;
    if (planner == "rrt") {
        out.rrt = rrt(arm, scene, scene.q_start, scene.q_goal, rng, 0.3, 0.1, 3000, 0.02, scene.sample_limits);
        if (!out.rrt.success) return out;
        path = out.rrt.path;
    } else {
        out.rrt_star = rrt_star(arm, scene, scene.q_start, scene.q_goal, rng, 0.3, 0.1, 1000, 0.6, 0.02,
                                scene.sample_limits);
        if (!out.rrt_star.success) return out;
        path = out.rrt_star.path;
    }
    out.success = true;
    if (smooth == "greedy" || smooth == "greedy+random") path = shortcut_greedy(arm, scene, path);
    if (smooth == "random" || smooth == "greedy+random") path = shortcut_random(arm, scene, path, rng);
    out.path = path;
    out.traj = time_scale_path(path, arm.qdot_max, arm.qddot_max);
    out.tau = torque_profile(arm, out.traj.Q, out.traj.Qd, out.traj.Qdd);
    out.limits = check_limits(arm, out.traj.Qd, out.tau);
    return out;
}

namespace {
std::vector<std::string> joint_names(const std::string& base, Eigen::Index n) {
    std::vector<std::string> h;
    for (Eigen::Index i = 1; i <= n; ++i) h.push_back(base + std::to_string(i));
    return h;
}
}  // namespace

void export_results(const std::string& prefix, const Arm& arm, const PipelineResult& out) {
    const bool star = out.planner == "rrt_star";
    const Eigen::MatrixXd& nodes = star ? out.rrt_star.nodes : out.rrt.nodes;
    const std::vector<int>& parent = star ? out.rrt_star.parent : out.rrt.parent;
    const Eigen::MatrixXd& raw = star ? out.rrt_star.path : out.rrt.path;
    const Eigen::Index n = nodes.cols();
    std::vector<std::string> h = {"node", "parent"};
    for (const auto& s : joint_names("q", n)) h.push_back(s);
    Eigen::MatrixXd tree(nodes.rows(), n + 2);
    for (Eigen::Index i = 0; i < nodes.rows(); ++i) {
        tree(i, 0) = static_cast<double>(i);
        tree(i, 1) = static_cast<double>(parent.at(static_cast<std::size_t>(i)));
        tree.row(i).tail(n) = nodes.row(i);
    }
    sbeg::write_csv(prefix + "_tree.csv", h, tree);
    sbeg::write_csv(prefix + "_path.csv", joint_names("q", n), raw);
    if (!out.success) return;
    sbeg::write_csv(prefix + "_smooth.csv", joint_names("q", n), out.path);
    const ScaledPath& tr = out.traj;
    std::vector<std::string> ht = {"t"};
    for (const auto& s : joint_names("q", n)) ht.push_back(s);
    for (const auto& s : joint_names("qd", n)) ht.push_back(s);
    for (const auto& s : joint_names("tau", n)) ht.push_back(s);
    Eigen::MatrixXd T(tr.t.size(), 1 + 3 * n);
    T << tr.t, tr.Q, tr.Qd, out.tau;
    sbeg::write_csv(prefix + "_traj.csv", ht, T);
    Eigen::MatrixXd tool(tr.t.size(), 4);
    for (Eigen::Index k = 0; k < tr.t.size(); ++k) {
        const Eigen::MatrixXd P = link_points(arm, tr.Q.row(k).transpose());
        tool(k, 0) = tr.t(k);
        tool.row(k).tail(3) = P.row(P.rows() - 1);
    }
    sbeg::write_csv(prefix + "_tool.csv", {"t", "x", "y", "z"}, tool);
}

}  // namespace lab
