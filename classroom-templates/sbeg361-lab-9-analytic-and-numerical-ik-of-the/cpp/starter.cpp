// SBEG 361 Lab 9 -- Analytic and numerical IK of the course arm; all-solutions search (C++ STARTER).
//
// Fill in every function marked  // TODO  (12 functions: replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 9 --cpp
//
// You may NOT call mrcpp (mr::ikin_body, mr::ikin_space, mr::fkin_space, ...) inside this file -- it is what the
// tests compare against. The helpers marked "complete" (from your Labs 2-7 library) may be used freely; replace
// them by your own versions if you prefer. Notation: MR Chapter 6 and MLS Ch. 3 Sec. 3 (see lab.hpp).
//
// Eigen reminders:  a.dot(b), a.cross(b), a.norm() (3-vectors); T.topLeftCorner<3, 3>() (R), T.topRightCorner<3, 1>()
//   (p), R.col(2) (z_b), A * B (matrix product), A.transpose(), pinv(J) (given, = np.linalg.pinv),
//   A.partialPivLu().solve(b) (= np.linalg.solve(A, b)), v.head<3>() / v.tail<3>() (omega / v of a twist),
//   (a - b).cwiseAbs().maxCoeff() (= np.max(np.abs(a - b))), Eigen::Index for loop counters.
//   A list of solutions: std::vector<Eigen::VectorXd> sols; sols.push_back(th);  an empty list: return {};
//   "None": return std::nullopt;  a pair: std::make_pair(a, b) or {a, b}.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;

/// Split one CSV line; a field in double quotes may contain commas.
std::vector<std::string> split_quoted_csv(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (const char ch : line) {
        if (ch == '"') {
            quoted = !quoted;
        } else if (ch == ',' && !quoted) {
            out.push_back(cell);
            cell.clear();
        } else {
            cell += ch;
        }
    }
    out.push_back(cell);
    return out;
}
}  // namespace

// ===================================================================== helpers (complete; from Labs 2-7)

Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d m;
    m << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return m;
}

Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat) {
    return Eigen::Vector3d(so3mat(2, 1), so3mat(0, 2), so3mat(1, 0));
}

Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();
    out.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    out.topRightCorner<3, 1>() = V.tail<3>();
    return out;
}

sbeg::Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat) {
    sbeg::Vector6d V;
    V << se3mat(2, 1), se3mat(0, 2), se3mat(1, 0), se3mat(0, 3), se3mat(1, 3), se3mat(2, 3);
    return V;
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.topLeftCorner<3, 3>() = R.transpose();
    out.topRightCorner<3, 1>() = -R.transpose() * p;
    return out;
}

sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    sbeg::Matrix6d out = sbeg::Matrix6d::Zero();
    out.topLeftCorner<3, 3>() = R;
    out.bottomRightCorner<3, 3>() = R;
    out.bottomLeftCorner<3, 3>() = vec_to_so3(p) * R;
    return out;
}

Eigen::Matrix3d rot(const Eigen::Vector3d& axis, double angle) {
    const Eigen::Vector3d w = axis / axis.norm();
    const Eigen::Matrix3d K = vec_to_so3(w);
    return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K * K;
}

Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat) {
    const double th = so3_to_vec(so3mat).norm();
    if (th < TOL) return Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d K = so3mat / th;
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R) {
    const double c = (R.trace() - 1.0) / 2.0;
    if (c >= 1.0) return Eigen::Matrix3d::Zero();
    if (c <= -1.0) {
        Eigen::Vector3d w;
        if (std::abs(1.0 + R(2, 2)) > TOL)
            w = Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2)) / std::sqrt(2.0 * (1.0 + R(2, 2)));
        else if (std::abs(1.0 + R(1, 1)) > TOL)
            w = Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1)) / std::sqrt(2.0 * (1.0 + R(1, 1)));
        else
            w = Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0)) / std::sqrt(2.0 * (1.0 + R(0, 0)));
        return vec_to_so3(PI * w);
    }
    const double th = std::acos(c);
    return th / (2.0 * std::sin(th)) * (R - R.transpose());
}

Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat) {
    const Eigen::Matrix3d W = se3mat.topLeftCorner<3, 3>();
    const Eigen::Vector3d v = se3mat.topRightCorner<3, 1>();
    const double th = so3_to_vec(W).norm();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    if (th < TOL) {
        out.topRightCorner<3, 1>() = v;
        return out;
    }
    const Eigen::Matrix3d K = W / th;
    out.topLeftCorner<3, 3>() = matrix_exp3(W);
    const Eigen::Matrix3d G = Eigen::Matrix3d::Identity() * th + (1.0 - std::cos(th)) * K + (th - std::sin(th)) * K * K;
    out.topRightCorner<3, 1>() = G * (v / th);
    return out;
}

Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    const Eigen::Matrix3d W = matrix_log3(R);
    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();
    if (W.cwiseAbs().maxCoeff() <= 1e-8) {  // np.allclose(W, 0)
        out.topRightCorner<3, 1>() = p;
        return out;
    }
    const double th = std::acos(std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0)));
    const Eigen::Matrix3d Ginv =
        Eigen::Matrix3d::Identity() - W / 2.0 + (1.0 / th - 1.0 / std::tan(th / 2.0) / 2.0) * (W * W) / th;
    out.topLeftCorner<3, 3>() = W;
    out.topRightCorner<3, 1>() = Ginv * p;
    return out;
}

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 0; i < theta.size(); ++i) {
        const sbeg::Vector6d S = Slist.col(i) * theta(i);
        T = (T * matrix_exp6(vec_to_se3(S))).eval();
    }
    return T * M;
}

Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    Eigen::Matrix4d T = M;
    for (Eigen::Index i = 0; i < theta.size(); ++i) {
        const sbeg::Vector6d B = Blist.col(i) * theta(i);
        T = (T * matrix_exp6(vec_to_se3(B))).eval();
    }
    return T;
}

Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M) {
    return adjoint(trans_inv(M)) * Slist;
}

Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    Eigen::MatrixXd Js = Slist;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 1; i < Slist.cols(); ++i) {
        const sbeg::Vector6d S = Slist.col(i - 1) * theta(i - 1);
        T = (T * matrix_exp6(vec_to_se3(S))).eval();
        Js.col(i) = adjoint(T) * Slist.col(i);
    }
    return Js;
}

Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    const Eigen::Index n = Blist.cols();
    Eigen::MatrixXd Jb = Blist;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = n - 2; i >= 0; --i) {
        const sbeg::Vector6d B = -Blist.col(i + 1) * theta(i + 1);
        T = (T * matrix_exp6(vec_to_se3(B))).eval();
        Jb.col(i) = adjoint(T) * Blist.col(i);
    }
    return Jb;
}

Eigen::MatrixXd pinv(const Eigen::MatrixXd& A) {
    // np.linalg.pinv(A, rcond=1e-15): singular values <= rcond * s_max are treated as zero.
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd s = svd.singularValues();
    const double cutoff = s.size() > 0 ? 1e-15 * s.maxCoeff() : 0.0;
    Eigen::VectorXd sinv = Eigen::VectorXd::Zero(s.size());
    for (Eigen::Index i = 0; i < s.size(); ++i)
        if (s(i) > cutoff) sinv(i) = 1.0 / s(i);
    return svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose();
}

double wrap_to_pi(double theta) {
    double r = std::fmod(theta + PI, 2.0 * PI);  // Python's % takes the sign of the divisor
    if (r < 0.0) r += 2.0 * PI;
    return r - PI;
}

Eigen::VectorXd wrap_to_pi(const Eigen::VectorXd& theta) {
    Eigen::VectorXd out(theta.size());
    for (Eigen::Index i = 0; i < theta.size(); ++i) out(i) = wrap_to_pi(theta(i));
    return out;
}

std::pair<double, double> pose_error(const Eigen::Matrix4d& T, const Eigen::Matrix4d& T_d) {
    const Eigen::Matrix3d R_err = T.topLeftCorner<3, 3>().transpose() * T_d.topLeftCorner<3, 3>();
    const double ang = std::acos(std::min(1.0, std::max(-1.0, (R_err.trace() - 1.0) / 2.0)));
    return {ang, (T.topRightCorner<3, 1>() - T_d.topRightCorner<3, 1>()).norm()};
}

bool within_limits(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits, double margin) {
    if (limits.rows() != theta.size() || limits.cols() != 2)
        throw std::invalid_argument("within_limits: limits must be n x 2");
    for (Eigen::Index j = 0; j < theta.size(); ++j)
        if (theta(j) < limits(j, 0) + margin || theta(j) > limits(j, 1) - margin) return false;
    return true;
}

Robot load_robot(const std::string& name, const std::string& path) {
    const sbeg::json all = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/robots.json") : path);
    if (!all.contains(name)) throw std::invalid_argument("load_robot: no robot named '" + name + "'");
    const sbeg::json& d = all.at(name);
    Robot r;
    r.Slist = sbeg::as_mat(d.at("Slist"));
    r.M = sbeg::as<Eigen::Matrix4d>(d.at("M"));
    r.name = d.at("name").get<std::string>();
    r.joint_types = d.at("joint_types").get<std::string>();
    r.limits = sbeg::as_mat(d.at("limits"));
    r.q_home = sbeg::as_vec(d.at("q_home"));
    if (d.contains("geometry") && !d.at("geometry").is_null()) {
        const sbeg::json& g = d.at("geometry");
        r.has_geometry = true;
        r.geometry.W1 = g.at("W1").get<double>();
        r.geometry.W2 = g.at("W2").get<double>();
        r.geometry.L1 = g.at("L1").get<double>();
        r.geometry.L2 = g.at("L2").get<double>();
        r.geometry.H1 = g.at("H1").get<double>();
        r.geometry.H2 = g.at("H2").get<double>();
        r.geometry.tool = g.value("tool", 0.0);
    }
    return r;
}

std::vector<BiopsyTarget> load_biopsy_targets(const std::string& path) {
    const std::string file = path.empty() ? sbeg::lab_path("../data/biopsy_targets.csv") : path;
    std::ifstream in = sbeg::open_input(file);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty file: " + file);
    const std::vector<std::string> header = split_quoted_csv(line);
    auto col = [&](const std::string& key) {
        const auto it = std::find(header.begin(), header.end(), key);
        if (it == header.end()) throw std::runtime_error(file + " has no column '" + key + "'");
        return static_cast<std::size_t>(it - header.begin());
    };
    const std::size_t c_id = col("id"), c_note = col("note");
    const std::size_t c_e[3] = {col("ex"), col("ey"), col("ez")}, c_a[3] = {col("ax"), col("ay"), col("az")};
    std::vector<BiopsyTarget> out;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        const std::vector<std::string> cells = split_quoted_csv(line);
        if (cells.size() != header.size()) throw std::runtime_error(file + ": wrong number of columns: " + line);
        BiopsyTarget t;
        t.id = cells[c_id];
        t.note = cells[c_note];
        for (int k = 0; k < 3; ++k) {
            t.entry(k) = std::stod(cells[c_e[k]]);
            t.axis(k) = std::stod(cells[c_a[k]]);
        }
        out.push_back(t);
    }
    return out;
}

Eigen::MatrixXd random_starts(int n_starts, int n, unsigned seed) {
    // std::uniform_real_distribution is implementation-defined (GCC, Clang and MSVC give different numbers), so the
    // 32-bit outputs of std::mt19937 (identical everywhere) are scaled by hand: u in [0, 1), theta = -pi + 2 pi u.
    if (n_starts < 0 || n < 0) throw std::invalid_argument("random_starts: negative size");
    std::mt19937 gen(seed);
    Eigen::MatrixXd starts(n_starts, n);
    for (int i = 0; i < n_starts; ++i)
        for (int k = 0; k < n; ++k) starts(i, k) = -PI + 2.0 * PI * (static_cast<double>(gen()) / 4294967296.0);
    return starts;
}

// ===================================================================== Checkpoint 1 -- Paden-Kahan subproblems

double pk1(const Eigen::Vector3d& omega, const Eigen::Vector3d& r, const Eigen::Vector3d& p,
           const Eigen::Vector3d& q) {
    // TODO: u = p - r, v = q - r; project on the plane normal to omega (u' = u - w w^T u); check MLS Eq. 3.19
    //       (throw std::invalid_argument if it fails); theta = atan2(w . (u' x v'), u' . v').
    //       Eigen: w.dot(u), up.cross(vp), up.norm().
    throw std::logic_error("TODO: pk1");
}

std::vector<std::pair<double, double>> pk2(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2,
                                           const Eigen::Vector3d& r, const Eigen::Vector3d& p,
                                           const Eigen::Vector3d& q) {
    // TODO: alpha, beta, gamma^2 from MLS p. 102; z = alpha w1 + beta w2 + gamma (w1 x w2); c = z + r;
    //       then theta2 = pk1(w2, r, p, c) and theta1 = pk1(w1, r, c, q) for each gamma.
    //       Skip a gamma for which pk1 throws (catch const std::invalid_argument&); sols.emplace_back(th1, th2).
    throw std::logic_error("TODO: pk2");
}

std::vector<double> pk3(const Eigen::Vector3d& omega, const Eigen::Vector3d& r, const Eigen::Vector3d& p,
                        const Eigen::Vector3d& q, double delta) {
    // TODO: delta'^2 = delta^2 - (w . (p - q))^2; theta0 = atan2(w . (u' x v'), u' . v');
    //       theta = theta0 -+ acos((|u'|^2 + |v'|^2 - delta'^2) / (2 |u'| |v'|))  (MLS Eq. 3.29); wrap to [-pi, pi).
    throw std::logic_error("TODO: pk3");
}

// ===================================================================== Checkpoint 2 -- analytic IK

std::vector<std::pair<double, double>> ik_2r(double x, double y, double L1, double L2) {
    // TODO: law of cosines (MR p. 218): beta, alpha, gamma = atan2(y, x); return {righty, lefty}.
    //       std::clamp(c, -1.0, 1.0) before std::acos (= np.clip).
    throw std::logic_error("TODO: ik_2r");
}

std::vector<Eigen::VectorXd> ik_course_arm(const Eigen::Matrix4d& T, const Geometry& geom) {
    // TODO: follow the six steps of lab.hpp; loop over 2 x theta1, 2 x theta5 and the ik_2r branches.
    //       Use rot(axis, angle) and wrap_to_pi; R_M = [[-1,0,0],[0,0,1],[0,1,0]] is the rotation part of M.
    //       Eigen: R = T.topLeftCorner<3, 3>(), p = T.topRightCorner<3, 1>(), z_b = R.col(2);
    //       a joint vector: Eigen::VectorXd th(6); th << th1, th2, th3, th4, th5, th6;
    throw std::logic_error("TODO: ik_course_arm");
}

// ===================================================================== Checkpoint 3 -- Newton-Raphson

NewtonResult newton_raphson(const ResidualFunction& g, const JacobianFunction& dg, const Eigen::VectorXd& theta0,
                            double tol, int max_iter) {
    // TODO: loop k = 0..max_iter: store {k, theta, g} in result.history; stop if |g| <= tol (or k == max_iter);
    //       step = dg.partialPivLu().solve(g) (square) or pinv(dg) * g; theta -= step.
    throw std::logic_error("TODO: newton_raphson");
}

IKTableResult ik_newton_body(const Eigen::MatrixXd& Blist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_sd,
                             const Eigen::VectorXd& theta0, double eomg, double ev, int max_iter) {
    // TODO: V_b = se3_to_vec(matrix_log6(trans_inv(fk_body(M, Blist, th)) * T_sd)); record the row {i, th, V_b,
    //       |w_b|, |v_b|}; stop if |w_b| <= eomg and |v_b| <= ev (or i == max_iter); else th += pinv(J_b) V_b.
    throw std::logic_error("TODO: ik_newton_body");
}

IKTableResult ik_newton_space(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_sd,
                              const Eigen::VectorXd& theta0, double eomg, double ev, int max_iter) {
    // TODO: as ik_newton_body, but V_s = [Ad_Tsb] V_b and J_s (fk_space, jacobian_space).
    throw std::logic_error("TODO: ik_newton_space");
}

// ===================================================================== Checkpoint 4 -- all-solutions search

std::vector<Eigen::VectorXd> unique_solutions(const std::vector<Eigen::VectorXd>& sols, double tol) {
    // TODO: compare wrap_to_pi(s - o) for every kept o  (all |entries| < tol: .cwiseAbs().maxCoeff() < tol).
    throw std::logic_error("TODO: unique_solutions");
}

SearchResult all_solutions_search(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M,
                                  const Eigen::Matrix4d& T_sd, const Eigen::MatrixXd& starts, double eomg,
                                  double ev, int max_iter) {
    // TODO: B = body_screws(Slist, M); run ik_newton_body from every row of starts (starts.row(k).transpose());
    //       group the converged (wrapped) results with tolerance 1e-4 and count the members of each group.
    throw std::logic_error("TODO: all_solutions_search");
}

std::optional<Eigen::VectorXd> select_solution(const std::vector<Eigen::VectorXd>& sols,
                                               const Eigen::VectorXd& theta_now, const Eigen::MatrixXd& limits) {
    // TODO: for each joint try s_j - 2 pi, s_j, s_j + 2 pi inside the limits; keep the closest to theta_now[j].
    //       Return std::nullopt if no solution is feasible.
    throw std::logic_error("TODO: select_solution");
}

// ===================================================================== Checkpoint 5 -- biopsy needle guide

Eigen::Matrix4d needle_guide_pose(const Eigen::Vector3d& entry, const Eigen::Vector3d& axis, double standoff,
                                  const Eigen::Vector3d& x_hint) {
    // TODO: z_b = axis / |axis|; x_b = normalised (x_hint - (x_hint . z_b) z_b); y_b = z_b x x_b; p = entry - standoff z_b.
    throw std::logic_error("TODO: needle_guide_pose");
}

}  // namespace lab
