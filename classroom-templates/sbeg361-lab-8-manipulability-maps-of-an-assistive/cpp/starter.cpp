// SBEG 361 Lab 8 -- Statics, manipulability maps of an assistive arm, workspace design (C++ STARTER).
//
// Fill in every function marked  // TODO  (14 functions: replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 8 --cpp
//
// You may NOT call mrcpp (mr::jacobian_space, mr::adjoint, ...) inside this file -- it is what the tests compare
// against. The helpers marked "complete" (your Lab 2-7 library: adjoint, FK, jacobian_space/body, plus the 2R
// helpers jacobian_2r, ik_2r_given, within_limits, select_rows, measure_of and the loaders) may be used freely.
// Notation: Lynch & Park, *Modern Robotics* (MR), Sections 5.2-5.4 (see lab.hpp).
//
// Eigen reminders:  J.transpose(), A * B (matrix product), J.rows(), J.cols(), M.rightCols(k) (last k columns),
//   J.topRows(3) / J.bottomRows(3), v.size(), v.maxCoeff(), v.minCoeff(), v.prod(), v.cwiseMax(0.0),
//   (s.array() > tol).count() (= np.sum(s > tol)), a.cross(b), Eigen::Index for loop counters.
//   Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV):
//     svd.singularValues() (decreasing), svd.matrixU() = U, svd.matrixV() = V (NOT V^T as numpy returns).
//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A): es.eigenvalues() increasing, es.eigenvectors() columns.
//   NaN / inf: std::numeric_limits<double>::quiet_NaN() / infinity(); std::isnan(x).
//   std::optional (ik_2r_given):  if (!th) ... (None);  *th is the Eigen::Vector2d.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

// ===================================================================== helpers (complete; from Labs 2-7)

Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d m;
    m << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return m;
}

Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();
    out.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    out.topRightCorner<3, 1>() = V.tail<3>();
    return out;
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

Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat) {
    const Eigen::Matrix3d W = se3mat.topLeftCorner<3, 3>();
    const Eigen::Vector3d v = se3mat.topRightCorner<3, 1>();
    const double th = Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0)).norm();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    if (th < TOL) {
        out.topRightCorner<3, 1>() = v;
        return out;
    }
    const Eigen::Matrix3d K = W / th;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    out.topLeftCorner<3, 3>() = I + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
    const Eigen::Matrix3d Gm = I * th + (1.0 - std::cos(th)) * K + (th - std::sin(th)) * K * K;
    out.topRightCorner<3, 1>() = Gm * (v / th);
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

Eigen::Matrix2d jacobian_2r(const Eigen::Vector2d& theta, double L1, double L2) {
    const double t1 = theta(0), t2 = theta(1);
    const double s1 = std::sin(t1), c1 = std::cos(t1), s12 = std::sin(t1 + t2), c12 = std::cos(t1 + t2);
    Eigen::Matrix2d J;
    J << -L1 * s1 - L2 * s12, -L2 * s12,
         L1 * c1 + L2 * c12, L2 * c12;
    return J;
}

std::optional<Eigen::Vector2d> ik_2r_given(double x, double y, double L1, double L2, int elbow) {
    const double c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2);
    if (std::abs(c2) > 1.0 + 1e-12) return std::nullopt;
    const double t2 = elbow * std::acos(std::min(1.0, std::max(-1.0, c2)));
    double t1 = std::atan2(y, x) - std::atan2(L2 * std::sin(t2), L1 + L2 * std::cos(t2));
    double r = std::fmod(t1 + PI, 2.0 * PI);  // Python's % is never negative for a positive divisor
    if (r < 0.0) r += 2.0 * PI;
    t1 = r - PI;
    return Eigen::Vector2d(t1, t2);
}

bool within_limits(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits) {
    if (limits.size() == 0) return true;
    if (limits.rows() != theta.size() || limits.cols() != 2)
        throw std::invalid_argument("within_limits: limits must be n x 2");
    for (Eigen::Index k = 0; k < theta.size(); ++k)
        if (theta(k) < limits(k, 0) - 1e-12 || theta(k) > limits(k, 1) + 1e-12) return false;
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
    // limits, q_home, tau_max may be null in the file and L exists only for the planar arms: left empty then.
    if (d.contains("limits") && !d.at("limits").is_null()) r.limits = sbeg::as_mat(d.at("limits"));
    if (d.contains("q_home") && !d.at("q_home").is_null()) r.q_home = sbeg::as_vec(d.at("q_home"));
    if (d.contains("tau_max") && !d.at("tau_max").is_null()) r.tau_max = sbeg::as_vec(d.at("tau_max"));
    if (d.contains("L") && !d.at("L").is_null()) r.L = sbeg::as_vec(d.at("L"));
    return r;
}

Eigen::MatrixXd load_targets(const std::string& path) {
    return sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/therapy_targets.csv") : path).data;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> load_feeding_csv(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/feeding_path.csv") : path);
    const Eigen::Index n = tab.data.cols();
    return {tab.data.col(0), tab.data.rightCols(n - 1)};
}

Eigen::MatrixXd select_rows(const Eigen::MatrixXd& J, const std::string& rows) {
    if (rows == "all") return J;
    if (rows == "angular") return J.topRows(3);
    if (rows == "linear") return J.bottomRows(3);
    throw std::invalid_argument("rows must be 'all', 'angular' or 'linear', not '" + rows + "'");
}

double measure_of(const Eigen::MatrixXd& J, const std::string& measure) {
    if (measure == "sigma_min") {
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);  // singular values only, in decreasing order
        const Eigen::VectorXd s = svd.singularValues();
        return s(s.size() - 1);
    }
    if (measure == "mu1") return manipulability_measures(J).mu1;
    if (measure == "mu2") return manipulability_measures(J).mu2;
    if (measure == "mu3") return manipulability_measures(J).mu3;
    throw std::invalid_argument("unknown measure '" + measure + "'");
}

// ===================================================================== Checkpoint 1 -- statics

Eigen::VectorXd joint_torques(const Eigen::MatrixXd& J, const Eigen::VectorXd& F) {
    // TODO: tau = J^T F (one line).
    throw std::logic_error("TODO: joint_torques");
}

Eigen::VectorXd wrench_from_torques(const Eigen::MatrixXd& J, const Eigen::VectorXd& tau, double tol) {
    // TODO: check that J is square and non-singular (SVD), throw std::invalid_argument otherwise; then solve
    // J^T F = tau.  Singular values only: Eigen::JacobiSVD<Eigen::MatrixXd> svd(J); svd.singularValues() (decreasing);
    // solve: J.transpose().partialPivLu().solve(tau).
    throw std::logic_error("TODO: wrench_from_torques");
}

sbeg::Vector6d body_to_space_wrench(const sbeg::Vector6d& F_b, const Eigen::Matrix4d& T_sb) {
    // TODO: use adjoint() and trans_inv(): F_s = [Ad_{T_bs}]^T F_b.
    throw std::logic_error("TODO: body_to_space_wrench");
}

sbeg::Vector6d holding_wrench_space(double m, const Eigen::Vector3d& p, const Eigen::Vector3d& g) {
    // TODO: f = -m g; return the 6-vector (p x f, f):  sbeg::Vector6d F; F << p.cross(f), f;
    throw std::logic_error("TODO: holding_wrench_space");
}

Eigen::MatrixXd wrench_null_space(const Eigen::MatrixXd& J, double tol) {
    // TODO: SVD J = U S V^T (Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU));
    // rank = number of singular values > tol; return the last 6 - rank columns of U (svd.matrixU().rightCols(...)).
    throw std::logic_error("TODO: wrench_null_space");
}

// ===================================================================== Checkpoint 2 -- ellipsoids and measures

Ellipsoid manipulability_ellipsoid(const Eigen::MatrixXd& J) {
    // TODO: A = J J^T; Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A) (= np.linalg.eigh: eigenvalues() in
    // INCREASING order, eigenvectors() as columns); sort in DESCENDING order (reorder the eigenvectors too);
    // semi-axes = sqrt (clip tiny negative round-off at 0).
    throw std::logic_error("TODO: manipulability_ellipsoid");
}

Ellipsoid force_ellipsoid(const Eigen::MatrixXd& J, double tol) {
    // TODO: reuse manipulability_ellipsoid; semi-axes 1/sqrt(lambda), std::numeric_limits<double>::infinity()
    // where lambda <= tol.
    throw std::logic_error("TODO: force_ellipsoid");
}

Measures manipulability_measures(const Eigen::MatrixXd& J, double tol) {
    // TODO: eigenvalues of A = J J^T (SelfAdjointEigenSolver, Eigen::EigenvaluesOnly); clip at 0; treat
    // lambda <= tol*max(1, lmax) as 0; mu2 = infinity at a singularity.  Return Measures{mu1, mu2, mu3}.
    throw std::logic_error("TODO: manipulability_measures");
}

// ===================================================================== Checkpoint 3 -- singularities

SingularityReport singularity_report(const Eigen::MatrixXd& J, double tol) {
    // TODO: Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // rank = number of singular values > tol; lost_twists = last (m - rank) columns of svd.matrixU(),
    // null_motions = last (n - rank) columns of svd.matrixV() (this is V, not V^T: no transpose needed).
    throw std::logic_error("TODO: singularity_report");
}

// ===================================================================== Checkpoint 4 -- maps and design (rehab_2r)

Eigen::MatrixXd manipulability_map_2r(double L1, double L2, const Eigen::VectorXd& xs, const Eigen::VectorXd& ys,
                                      const std::string& measure, int elbow, const Eigen::MatrixXd& limits,
                                      const Eigen::Vector2d& base) {
    // TODO: start from Eigen::MatrixXd::Constant(ys.size(), xs.size(), std::numeric_limits<double>::quiet_NaN());
    // loop over ys (rows) and xs (columns); th = ik_2r_given(x - bx, y - by, ...); skip an empty th or
    // !within_limits(*th, limits); measure_of(jacobian_2r(*th, L1, L2), measure).
    throw std::logic_error("TODO: manipulability_map_2r");
}

std::pair<double, double> region_score(double L1, double L2, const Eigen::MatrixXd& points,
                                       const std::string& measure, int elbow, const Eigen::MatrixXd& limits,
                                       const Eigen::Vector2d& base) {
    // TODO: same loop as the map but over the rows of points; worst = min (mu3, sigma_min) or max (mu1, mu2)
    // over the reachable points; fraction = reachable / total (as a double); worst = NaN if none is reachable.
    throw std::logic_error("TODO: region_score");
}

DesignSearchResult design_search(double total, const Eigen::VectorXd& ratios, const Eigen::VectorXd& base_offsets,
                                 const Eigen::MatrixXd& points, const Eigen::MatrixXd& limits, int elbow,
                                 const std::string& measure) {
    // TODO: two nested loops (ratios outer, base_offsets inner), base = Eigen::Vector2d(0, -b); push one DesignRow
    // per design into result.rows; best = among rows with fraction == 1.0 (first one wins a tie), or left empty.
    throw std::logic_error("TODO: design_search");
}

// ===================================================================== Checkpoint 5 -- spatial arms

PathMeasures measures_along_path(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Q,
                                 const std::string& rows) {
    // TODO: B = body_screws(Slist, M); for each row q of Q (Eigen::VectorXd q = Q.row(k).transpose()):
    // J = select_rows(jacobian_body(B, q), rows); manipulability_measures(J); smallest singular value
    // (measure_of(J, "sigma_min")).
    throw std::logic_error("TODO: measures_along_path");
}

Eigen::MatrixXd joint_space_map(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::VectorXd& q0,
                                int i, int j, const Eigen::VectorXd& gi, const Eigen::VectorXd& gj,
                                const std::string& measure, const std::string& rows) {
    // TODO: copy q0, set q(i), q(j); J = select_rows(jacobian_body(B, q), rows); measure_of(J, measure).
    throw std::logic_error("TODO: joint_space_map");
}

}  // namespace lab
