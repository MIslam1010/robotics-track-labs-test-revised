// SBEG 46X Lab 1 -- provided helpers (C++ track).  GIVEN code: do not edit; use these functions freely.
//
// The "(Provided.)" part of starter.py, translated to C++:
//   load_task, environment_report, skew, min_jerk, port_frame, make_trajectory
// plus save_log_csv (write a simulation log as CSV for plotting).
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {
namespace {
constexpr double PI = 3.14159265358979323846;

/// One prepared segment of make_trajectory.
struct Segment {
    bool line = true;
    double ts = 0.0, T = 0.0;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();  // line: start; circle: centre
    Eigen::Vector3d b = Eigen::Vector3d::Zero();  // line: end
    double r = 0.0, k = 0.0;                      // circle: radius, turns
};
}  // namespace

Task load_task(const std::string& path) {
    const sbeg::json J = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/lab1_task.json") : path);
    Task task;
    task.p_port = sbeg::as<Eigen::Vector3d>(J.at("p_port"));
    task.theta0 = sbeg::as_vec(J.at("theta0"));
    for (const auto& s : J.at("path")) {
        PathSegment seg;
        seg.type = s.at("type").get<std::string>();
        seg.T = s.at("T").get<double>();
        if (s.contains("to")) seg.to = sbeg::as<Eigen::Vector3d>(s.at("to"));
        if (s.contains("radius")) seg.radius = s.at("radius").get<double>();
        if (s.contains("turns")) seg.turns = s.at("turns").get<double>();
        task.path.push_back(seg);
    }
    task.dt = J.at("dt").get<double>();
    task.gains.kp = J.at("gains").at("kp").get<double>();
    task.gains.k_rcm = J.at("gains").at("k_rcm").get<double>();
    task.gains.lam = J.at("gains").at("lam").get<double>();
    task.acceptance.max_tip_error_mm = J.at("acceptance").at("max_tip_error_mm").get<double>();
    task.acceptance.max_rcm_error_mm = J.at("acceptance").at("max_rcm_error_mm").get<double>();
    return task;
}

std::map<std::string, std::string> environment_report() {
    std::map<std::string, std::string> rep;
#if defined(_MSC_VER)
    rep["compiler"] = "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    rep["compiler"] = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    rep["compiler"] = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    rep["compiler"] = "unknown";
#endif
    rep["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
                   std::to_string(EIGEN_MINOR_VERSION);
    rep["kinematics"] = kinematics_backend();
    return rep;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& p) {
    Eigen::Matrix3d S;
    S << 0.0, -p(2), p(1),
         p(2), 0.0, -p(0),
         -p(1), p(0), 0.0;
    return S;
}

std::pair<double, double> min_jerk(double tau) {
    tau = std::min(std::max(tau, 0.0), 1.0);
    const double s = 10 * std::pow(tau, 3) - 15 * std::pow(tau, 4) + 6 * std::pow(tau, 5);
    const double ds = 30 * std::pow(tau, 2) - 60 * std::pow(tau, 3) + 30 * std::pow(tau, 4);
    return {s, ds};
}

Eigen::Matrix3d port_frame(const Eigen::Vector3d& z_axis) {
    const Eigen::Vector3d z = z_axis / z_axis.norm();
    Eigen::Vector3d x = Eigen::Vector3d::UnitZ().cross(z);
    if (x.norm() < 1e-9) x = Eigen::Vector3d::UnitX();
    x /= x.norm();
    const Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d R;
    R << x, y, z;  // columns
    return R;
}

std::pair<TrajectoryFn, double> make_trajectory(const Task& task, const Eigen::Vector3d& p_tip0,
                                                const Eigen::Vector3d& z0) {
    const Eigen::Matrix3d R = port_frame(z0);
    const Eigen::Vector3d o = task.p_port;
    std::vector<Segment> segs;
    double t0 = 0.0;
    Eigen::Vector3d cur = R.transpose() * (p_tip0 - o);
    for (const PathSegment& ps : task.path) {
        Segment sg;
        sg.ts = t0;
        sg.T = ps.T;
        if (ps.type == "line") {
            sg.line = true;
            sg.a = cur;
            sg.b = ps.to;
            cur = ps.to;
        } else if (ps.type == "circle") {
            sg.line = false;
            sg.r = ps.radius;
            sg.k = ps.turns;
            sg.a = cur - Eigen::Vector3d(sg.r, 0.0, 0.0);
        } else {
            throw std::invalid_argument("unknown path segment type: " + ps.type);
        }
        segs.push_back(sg);
        t0 += ps.T;
    }
    if (segs.empty()) throw std::invalid_argument("the task has no path segments");
    TrajectoryFn traj = [segs, R, o](double t) -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
        for (std::size_t i = 0; i < segs.size(); ++i) {
            const Segment& sg = segs[i];
            if (t <= sg.ts + sg.T || i + 1 == segs.size()) {
                const auto [s, ds] = min_jerk((t - sg.ts) / sg.T);
                Eigen::Vector3d pp, vp;
                if (sg.line) {
                    pp = sg.a + s * (sg.b - sg.a);
                    vp = ds / sg.T * (sg.b - sg.a);
                } else {
                    const double ang = 2 * PI * sg.k * s, dang = 2 * PI * sg.k * ds / sg.T;
                    pp = sg.a + sg.r * Eigen::Vector3d(std::cos(ang), std::sin(ang), 0.0);
                    vp = sg.r * dang * Eigen::Vector3d(-std::sin(ang), std::cos(ang), 0.0);
                }
                return {o + R * pp, R * vp};
            }
        }
        throw std::runtime_error("unreachable");
    };
    return {traj, t0};
}

void save_log_csv(const std::string& path, const SimLog& log) {
    const Eigen::Index n = log.t.size();
    if (log.theta.rows() != n || log.p_tip.rows() != n || log.p_d.rows() != n || log.tip_err.size() != n ||
        log.rcm_err.size() != n || log.depth.size() != n)
        throw std::invalid_argument("save_log_csv: the log fields have different lengths");
    std::vector<std::string> header = {"t"};
    for (int i = 1; i <= log.theta.cols(); ++i) header.push_back("theta_" + std::to_string(i));
    for (const char* c : {"x", "y", "z"}) header.push_back(std::string("p_tip_") + c);
    for (const char* c : {"x", "y", "z"}) header.push_back(std::string("p_d_") + c);
    for (const char* c : {"tip_err", "rcm_err", "depth"}) header.push_back(c);
    Eigen::MatrixXd D(n, 1 + log.theta.cols() + 6 + 3);
    D << log.t, log.theta, log.p_tip, log.p_d, log.tip_err, log.rcm_err, log.depth;
    sbeg::write_csv(path, header, D);
}

}  // namespace lab
