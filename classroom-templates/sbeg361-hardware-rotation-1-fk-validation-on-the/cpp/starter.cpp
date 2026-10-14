// SBEG 361 Hardware rotation 1 (Week 8) -- FK validation on the course arm (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py hw1 --cpp
//
// Rules (as in starter.py): do not call mrcpp (mr::...) or the course-arm model (course_arm_reference) in this file;
// the helpers above the TODO functions are complete and may be used (you may replace fk_space / screw_exp by your own
// Lab 5 versions).  gradient and median are given because NumPy provides them in Python.
//
// Eigen reminders:  A.rowwise() - v.transpose() (subtract a row vector from every row), A.colwise().minCoeff(),
//   A.colwise().mean(), A.rowwise().norm(), A.cwiseAbs(), A.topRows(k), A.row(k).transpose(), (v.array() >= x).all().
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double DEG = PI / 180.0;
}  // namespace

// =========================================================== complete helpers (same in starter.cpp)
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W << 0.0, -w(2), w(1),
         w(2), 0.0, -w(0),
         -w(1), w(0), 0.0;
    return W;
}

Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta) {
    const Eigen::Vector3d w = S.head<3>(), v = S.tail<3>();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (w.norm() < 1e-12) {
        T.topRightCorner<3, 1>() = v * theta;
        return T;
    }
    const Eigen::Matrix3d W = vec_to_so3(w);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R = I + std::sin(theta) * W + (1 - std::cos(theta)) * W * W;
    const Eigen::Matrix3d G = I * theta + (1 - std::cos(theta)) * W + (theta - std::sin(theta)) * W * W;
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = G * v;
    return T;
}

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    if (Slist.rows() != 6 || Slist.cols() != theta.size())
        throw std::invalid_argument("fk_space: Slist must be 6 x n and theta must have n entries");
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 0; i < theta.size(); ++i) T = (T * screw_exp(Slist.col(i), theta(i))).eval();
    return T * M;
}

Eigen::MatrixXd tool_points(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& Q,
                            const Eigen::Vector3d& p_tool) {
    const Eigen::Vector4d pb(p_tool(0), p_tool(1), p_tool(2), 1.0);
    Eigen::MatrixXd P(Q.rows(), 3);
    for (Eigen::Index k = 0; k < Q.rows(); ++k) {
        const Eigen::VectorXd q = Q.row(k).transpose();
        P.row(k) = (fk_space(M, Slist, q) * pb).head<3>().transpose();
    }
    return P;
}

Eigen::MatrixXd servo_to_model(const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg) {
    return (q_servo_deg.rowwise() - offsets_deg.transpose()) * DEG;
}

Eigen::MatrixXd model_to_servo(const Eigen::MatrixXd& q_model_rad, const Eigen::VectorXd& offsets_deg) {
    return (q_model_rad / DEG).rowwise() + offsets_deg.transpose();
}

Eigen::VectorXd load_offsets(const std::string& path) {
    return sbeg::as_vec(sbeg::load_json(path).at("servo_zero_offset_deg"));
}

sbeg::json load_plan(const std::string& path) { return sbeg::load_json(path); }

Eigen::MatrixXd gradient(const Eigen::MatrixXd& F, const Eigen::VectorXd& t) {
    const Eigen::Index N = F.rows();
    if (t.size() != N || N < 2) throw std::invalid_argument("gradient: need at least 2 samples and one time per row");
    Eigen::MatrixXd D(N, F.cols());
    D.row(0) = (F.row(1) - F.row(0)) / (t(1) - t(0));
    D.row(N - 1) = (F.row(N - 1) - F.row(N - 2)) / (t(N - 1) - t(N - 2));
    for (Eigen::Index i = 1; i + 1 < N; ++i) {
        const double hs = t(i) - t(i - 1), hd = t(i + 1) - t(i);
        D.row(i) = (hs * hs * F.row(i + 1) + (hd * hd - hs * hs) * F.row(i) - hd * hd * F.row(i - 1)) /
                   (hs * hd * (hd + hs));
    }
    return D;
}

double median(const Eigen::VectorXd& x) {
    if (x.size() == 0) throw std::invalid_argument("median of an empty vector");
    std::vector<double> v(x.data(), x.data() + x.size());
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

Trajectory smooth_segments(const Eigen::MatrixXd& q_via_deg, double T_seg, double rate_hz, double t_hold) {
    const Eigen::Index nv = q_via_deg.rows(), nj = q_via_deg.cols();
    if (nv < 1) throw std::invalid_argument("smooth_segments: at least one via point");
    const double dt = 1.0 / rate_hz;
    const auto n = static_cast<Eigen::Index>(std::nearbyint(T_seg * rate_hz));  // = Python round (half to even)
    const auto n_hold = static_cast<Eigen::Index>(std::nearbyint(t_hold * rate_hz)) + 1;
    const Eigen::Index N = (nv - 1) * n + n_hold;
    Trajectory tr{Eigen::VectorXd(N), Eigen::MatrixXd(N, nj)};
    Eigen::Index k = 0;
    double t0 = 0.0;
    for (Eigen::Index s = 0; s + 1 < nv; ++s) {
        for (Eigen::Index i = 0; i < n; ++i, ++k) {
            const double tau = static_cast<double>(i) / static_cast<double>(n);
            const double sc = 3 * tau * tau - 2 * tau * tau * tau;
            tr.t(k) = t0 + tau * T_seg;
            tr.q.row(k) = q_via_deg.row(s) + sc * (q_via_deg.row(s + 1) - q_via_deg.row(s));
        }
        t0 += T_seg;
    }
    for (Eigen::Index i = 0; i < n_hold; ++i, ++k) {
        tr.t(k) = t0 + dt * static_cast<double>(i);
        tr.q.row(k) = q_via_deg.row(nv - 1);
    }
    return tr;
}

void write_log(const std::string& path, const Log& log) {
    std::ofstream out(sbeg::fs_path(path), std::ios::binary);
    if (!out) throw std::runtime_error("cannot write file: " + path);
    const Eigen::Index n = log.cmd.cols();
    out << "t";
    for (Eigen::Index i = 1; i <= n; ++i) out << ",cmd_" << i;
    for (Eigen::Index i = 1; i <= n; ++i) out << ",meas_" << i;
    out << ",stop\r\n";  // csv.writer line ending, as Python
    out << std::fixed << std::setprecision(4);
    for (Eigen::Index k = 0; k < log.t.size(); ++k) {
        out << log.t(k);
        for (Eigen::Index i = 0; i < n; ++i) out << "," << log.cmd(k, i);
        for (Eigen::Index i = 0; i < n; ++i) out << "," << log.meas(k, i);
        out << "," << static_cast<int>(log.stop(k)) << "\r\n";
    }
}

Log read_log(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path);
    Eigen::Index n = 0;
    for (const auto& h : tab.header) n += h.rfind("cmd_", 0) == 0 ? 1 : 0;
    Log log;
    log.t = tab.data.col(tab.col("t"));
    log.cmd.resize(tab.data.rows(), n);
    log.meas.resize(tab.data.rows(), n);
    for (Eigen::Index i = 0; i < n; ++i) {
        log.cmd.col(i) = tab.data.col(tab.col("cmd_" + std::to_string(i + 1)));
        log.meas.col(i) = tab.data.col(tab.col("meas_" + std::to_string(i + 1)));
    }
    const bool has_stop = std::find(tab.header.begin(), tab.header.end(), "stop") != tab.header.end();
    log.stop = has_stop ? Eigen::VectorXd(tab.data.col(tab.col("stop"))) : Eigen::VectorXd::Zero(tab.data.rows());
    return log;
}

Tips read_tips(const std::string& path) {
    std::ifstream in(sbeg::fs_path(path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    std::string line;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cells.push_back(c);
        if (header.empty()) header = cells;
        else rows.push_back(cells);
    }
    auto col = [&](const std::string& name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::runtime_error(path + ": no column '" + name + "'");
        return static_cast<std::size_t>(it - header.begin());
    };
    const auto m = static_cast<Eigen::Index>(rows.size());
    Tips tips{{}, Eigen::VectorXi(m), Eigen::MatrixXd(m, 6), Eigen::MatrixXd(m, 3)};
    const char* xyz[3] = {"x_mm", "y_mm", "z_mm"};
    for (Eigen::Index k = 0; k < m; ++k) {
        const auto& r = rows[static_cast<std::size_t>(k)];
        if (r.size() != header.size()) throw std::runtime_error(path + ": wrong number of columns");
        tips.pose.push_back(r[col("pose")]);
        tips.approach(k) = std::stoi(r[col("approach")]);
        for (int i = 0; i < 6; ++i) tips.Q(k, i) = std::stod(r[col("servo_" + std::to_string(i + 1))]);
        for (int i = 0; i < 3; ++i) tips.P(k, i) = std::stod(r[col(xyz[i])]);
    }
    return tips;
}

Eigen::Index before_stop(const Log& log) {
    for (Eigen::Index k = 0; k < log.stop.size(); ++k)
        if (log.stop(k) > 0.5) return k;
    return log.t.size();
}

// =========================================================== Step 1 -- before the arm moves (safety gate)
PlanCheck check_plan(const Eigen::VectorXd& t, const Eigen::MatrixXd& q_servo_deg, const Eigen::VectorXd& offsets_deg,
                     const Eigen::MatrixXd& soft_limits_deg, double rated_deg_s, double speed_fraction,
                     double margin_deg) {
    // TODO: convert to model degrees (q - offsets: q_servo_deg.rowwise() - offsets_deg.transpose());
    //       margin = min(min(q) - lower, upper - max(q)) per joint (.colwise().minCoeff(), soft_limits_deg.col(0));
    //       peak = max |gradient(q, t)| per joint (the given gradient = np.gradient);
    //       compare with margin_deg and speed_fraction * rated_deg_s; fill every field of PlanCheck
    throw std::logic_error("TODO: check_plan");
}

// =========================================================== Step 2 -- logged run (safety gate)
Log run_logged(Arm& arm, const Eigen::VectorXd& t, const Eigen::MatrixXd& q_cmd_servo_deg,
               std::optional<double> stop_at) {
    // TODO: loop over k: arm.wait_until(t(k)); stop once when stop_at && t(k) >= *stop_at; while not stopped
    //       try { arm.send(row k) } catch (const SafetyViolation& e) { arm.stop(); remember e.what() in aborted };
    //       store the last command sent, arm.read(), the stop flag (0/1).  Start "last" at row 0 of the commands.
    throw std::logic_error("TODO: run_logged");
}

// =========================================================== Step 3 -- joint space, task space, stop
JointTracking joint_tracking(const Log& log) {
    // TODO: use only samples before the first stop = 1 (helper before_stop, then .topRows(k0)); e = meas - cmd
    throw std::logic_error("TODO: joint_tracking");
}

TaskTracking task_tracking(const Log& log, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                           const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool) {
    // TODO: FK of cmd and of meas (servo_to_model, tool_points), distance in mm (.rowwise().norm()), before the stop
    throw std::logic_error("TODO: task_tracking");
}

StopCheck stop_check(const Log& log, double band_deg, double tail_s) {
    // TODO: throw std::invalid_argument("no stop command in the log") if there is no stop;
    //       rest pose = median (given helper) of meas over the last tail_s, per joint;
    //       find the last post-stop sample outside the band (t_halt = infinity if it is the last sample:
    //       std::numeric_limits<double>::infinity()); motion = max |meas - meas(at the stop)| after the stop
    throw std::logic_error("TODO: stop_check");
}

// =========================================================== Step 4 -- is the arm where the model says?
Eigen::VectorXd hw_tip_errors_mm(const Eigen::MatrixXd& Q_servo_deg, const Eigen::MatrixXd& P_meas_mm,
                                 const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist,
                                 const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool) {
    // TODO: FK prediction (mm) of the tool point for each pose minus the measured point; norm per row
    throw std::logic_error("TODO: hw_tip_errors_mm");
}

AccuracyRepeatability accuracy_repeatability(const Eigen::MatrixXd& P_mm, const Eigen::Vector3d& p_ref_mm) {
    // TODO: barycentre (.colwise().mean()), AP, distances l_i to the barycentre, RP = mean + 3 * std (divide by m - 1)
    throw std::logic_error("TODO: accuracy_repeatability");
}

double backlash_split(const Eigen::MatrixXd& P_plus_mm, const Eigen::MatrixXd& P_minus_mm) {
    // TODO: difference of the two barycentres
    throw std::logic_error("TODO: backlash_split");
}

// =========================================================== Step 5 -- joint-offset calibration
OffsetFit fit_offsets(const Eigen::MatrixXd& Q_servo_deg, const Eigen::MatrixXd& P_meas_mm, const Eigen::Matrix4d& M,
                      const Eigen::MatrixXd& Slist, const Eigen::VectorXd& offsets_deg, const Eigen::Vector3d& p_tool,
                      const std::vector<int>& joints, int iters, double h_deg) {
    // TODO: residual(delta) = FK points with offsets[joints] -= delta, minus P, flattened row by row (x1 y1 z1 x2 ...),
    //       in mm (a lambda is convenient); Gauss-Newton: J by central differences (step h_deg),
    //       step = J.completeOrthogonalDecomposition().solve(-r)  (= np.linalg.lstsq); stop when max |step| < 1e-6
    throw std::logic_error("TODO: fit_offsets");
}

// =========================================================== Step 6 -- error budget
ErrorBudget error_budget(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta,
                         const Eigen::Vector3d& p_tool, const Eigen::VectorXd& dtheta_deg) {
    // TODO: for each joint: home axis point q_j = omega_j x v_j; move omega_j and q_j with T_{j-1};
    //       lever = |omega_j' x (p - q_j')|; then update T = T * screw_exp(S_j, theta_j)
    throw std::logic_error("TODO: error_budget");
}

ErrorBudget error_budget(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta,
                         const Eigen::Vector3d& p_tool, double dtheta_deg) {
    // Given: the same error for every joint.
    return error_budget(M, Slist, theta, p_tool, Eigen::VectorXd::Constant(Slist.cols(), dtheta_deg));
}

}  // namespace lab
