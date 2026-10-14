// mrcpp.cpp -- C++17 port of modern_robotics 1.1.1 (core.py). See include/sbeg/mrcpp.hpp.
// Each function follows the Python code line by line (same branches, tolerances and order of products).
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mr {

namespace {
constexpr double PI = 3.14159265358979323846;

void require(bool ok, const std::string& msg) {
    if (!ok) throw std::invalid_argument("mrcpp: " + msg);
}

void check_screw_list(const MatrixXd& list, const VectorXd& thetalist, const char* fn) {
    require(list.rows() == 6, std::string(fn) + ": the screw-axis list must be 6 x n (axes are columns)");
    require(list.cols() >= thetalist.size(),
            std::string(fn) + ": the screw-axis list has fewer columns than thetalist has entries");
}

void check_chain(const VectorXd& thetalist, const TransformList& Mlist, const InertiaList& Glist,
                 const MatrixXd& Slist, const char* fn) {
    const auto n = static_cast<std::size_t>(thetalist.size());
    require(Mlist.size() >= n + 1, std::string(fn) + ": Mlist needs n+1 frames");
    require(Glist.size() >= n, std::string(fn) + ": Glist needs n inertia matrices");
    check_screw_list(Slist, thetalist, fn);
}
}  // namespace

// ============================================================ basic helpers
bool near_zero(double z) { return std::abs(z) < 1e-6; }

VectorXd normalize(const VectorXd& V) { return V / V.norm(); }

MatrixXd pinv(const MatrixXd& A) {
    // numpy.linalg.pinv(A, rcond=1e-15): singular values <= rcond * s_max are set to zero.
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const VectorXd& s = svd.singularValues();
    const double cutoff = s.size() > 0 ? 1e-15 * s.maxCoeff() : 0.0;
    VectorXd sinv = VectorXd::Zero(s.size());
    for (Eigen::Index i = 0; i < s.size(); ++i)
        if (s(i) > cutoff) sinv(i) = 1.0 / s(i);
    return svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose();
}

// ============================================================ Chapter 3
Matrix3d rot_inv(const Matrix3d& R) { return R.transpose(); }

Matrix3d vec_to_so3(const Vector3d& omg) {
    Matrix3d m;
    m << 0, -omg(2), omg(1),
         omg(2), 0, -omg(0),
         -omg(1), omg(0), 0;
    return m;
}

Vector3d so3_to_vec(const Matrix3d& so3mat) { return Vector3d(so3mat(2, 1), so3mat(0, 2), so3mat(1, 0)); }

std::pair<Vector3d, double> axis_ang3(const Vector3d& expc3) {
    return {expc3 / expc3.norm(), expc3.norm()};
}

Matrix3d matrix_exp3(const Matrix3d& so3mat) {
    const Vector3d omgtheta = so3_to_vec(so3mat);
    if (near_zero(omgtheta.norm())) return Matrix3d::Identity();
    const double theta = axis_ang3(omgtheta).second;
    const Matrix3d omgmat = so3mat / theta;
    return Matrix3d::Identity() + std::sin(theta) * omgmat + (1 - std::cos(theta)) * (omgmat * omgmat);
}

Matrix3d matrix_log3(const Matrix3d& R) {
    const double acosinput = (R.trace() - 1) / 2.0;
    if (acosinput >= 1) return Matrix3d::Zero();
    if (acosinput <= -1) {
        Vector3d omg;
        if (!near_zero(1 + R(2, 2)))
            omg = (1.0 / std::sqrt(2 * (1 + R(2, 2)))) * Vector3d(R(0, 2), R(1, 2), 1 + R(2, 2));
        else if (!near_zero(1 + R(1, 1)))
            omg = (1.0 / std::sqrt(2 * (1 + R(1, 1)))) * Vector3d(R(0, 1), 1 + R(1, 1), R(2, 1));
        else
            omg = (1.0 / std::sqrt(2 * (1 + R(0, 0)))) * Vector3d(1 + R(0, 0), R(1, 0), R(2, 0));
        return vec_to_so3(PI * omg);
    }
    const double theta = std::acos(acosinput);
    return theta / 2.0 / std::sin(theta) * (R - R.transpose());
}

Matrix4d rp_to_trans(const Matrix3d& R, const Vector3d& p) {
    Matrix4d T = Matrix4d::Zero();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = p;
    T(3, 3) = 1;
    return T;
}

std::pair<Matrix3d, Vector3d> trans_to_rp(const Matrix4d& T) {
    return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()};
}

Matrix4d trans_inv(const Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    const Matrix3d Rt = R.transpose();
    return rp_to_trans(Rt, -(Rt * p));
}

Matrix4d vec_to_se3(const Vector6d& V) {
    Matrix4d m = Matrix4d::Zero();
    m.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    m.topRightCorner<3, 1>() = V.tail<3>();
    return m;
}

Vector6d se3_to_vec(const Matrix4d& se3mat) {
    Vector6d V;
    V << se3mat(2, 1), se3mat(0, 2), se3mat(1, 0), se3mat(0, 3), se3mat(1, 3), se3mat(2, 3);
    return V;
}

Matrix6d adjoint(const Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    Matrix6d A = Matrix6d::Zero();
    A.topLeftCorner<3, 3>() = R;
    A.bottomLeftCorner<3, 3>() = vec_to_so3(p) * R;
    A.bottomRightCorner<3, 3>() = R;
    return A;
}

Vector6d screw_to_axis(const Vector3d& q, const Vector3d& s, double h) {
    Vector6d S;
    S << s, q.cross(s) + h * s;
    return S;
}

std::pair<Vector6d, double> axis_ang6(const Vector6d& expc6) {
    double theta = expc6.head<3>().norm();
    if (near_zero(theta)) theta = expc6.tail<3>().norm();
    return {expc6 / theta, theta};
}

Matrix4d matrix_exp6(const Matrix4d& se3mat) {
    const Matrix3d so3 = se3mat.topLeftCorner<3, 3>();
    const Vector3d v = se3mat.topRightCorner<3, 1>();
    const Vector3d omgtheta = so3_to_vec(so3);
    if (near_zero(omgtheta.norm())) return rp_to_trans(Matrix3d::Identity(), v);
    const double theta = axis_ang3(omgtheta).second;
    const Matrix3d omgmat = so3 / theta;
    const Matrix3d G = Matrix3d::Identity() * theta + (1 - std::cos(theta)) * omgmat +
                       (theta - std::sin(theta)) * (omgmat * omgmat);
    return rp_to_trans(matrix_exp3(so3), G * v / theta);
}

Matrix4d matrix_log6(const Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    const Matrix3d omgmat = matrix_log3(R);
    Matrix4d out = Matrix4d::Zero();
    if ((omgmat.array() == 0.0).all()) {  // np.array_equal(omgmat, zeros): exact comparison, as in Python
        out.topRightCorner<3, 1>() = p;
        return out;
    }
    const double theta = std::acos((R.trace() - 1) / 2.0);
    const Matrix3d Ginv = Matrix3d::Identity() - omgmat / 2.0 +
                          (1.0 / theta - 1.0 / std::tan(theta / 2.0) / 2) * (omgmat * omgmat) / theta;
    out.topLeftCorner<3, 3>() = omgmat;
    out.topRightCorner<3, 1>() = Ginv * p;
    return out;
}

Matrix3d project_to_so3(const Matrix3d& mat) {
    Eigen::JacobiSVD<Matrix3d> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) R.col(2) = -R.col(2);
    return R;
}

Matrix4d project_to_se3(const Matrix4d& mat) {
    return rp_to_trans(project_to_so3(mat.topLeftCorner<3, 3>()), mat.topRightCorner<3, 1>());
}

double distance_to_so3(const Matrix3d& mat) {
    if (mat.determinant() > 0) return (mat.transpose() * mat - Matrix3d::Identity()).norm();
    return 1e+9;
}

double distance_to_se3(const Matrix4d& mat) {
    const Matrix3d matR = mat.topLeftCorner<3, 3>();
    if (matR.determinant() > 0) {
        Matrix4d m = Matrix4d::Zero();
        m.topLeftCorner<3, 3>() = matR.transpose() * matR;
        m.row(3) = mat.row(3);
        return (m - Matrix4d::Identity()).norm();
    }
    return 1e+9;
}

bool test_if_so3(const Matrix3d& mat) { return std::abs(distance_to_so3(mat)) < 1e-3; }

bool test_if_se3(const Matrix4d& mat) { return std::abs(distance_to_se3(mat)) < 1e-3; }

// ============================================================ Chapter 4
Matrix4d fkin_body(const Matrix4d& M, const MatrixXd& Blist, const VectorXd& thetalist) {
    check_screw_list(Blist, thetalist, "fkin_body");
    Matrix4d T = M;
    for (Eigen::Index i = 0; i < thetalist.size(); ++i)
        T = T * matrix_exp6(vec_to_se3(Vector6d(Blist.col(i) * thetalist(i))));
    return T;
}

Matrix4d fkin_space(const Matrix4d& M, const MatrixXd& Slist, const VectorXd& thetalist) {
    check_screw_list(Slist, thetalist, "fkin_space");
    Matrix4d T = M;
    for (Eigen::Index i = thetalist.size() - 1; i >= 0; --i)
        T = matrix_exp6(vec_to_se3(Vector6d(Slist.col(i) * thetalist(i)))) * T;
    return T;
}

// ============================================================ Chapter 5
MatrixXd jacobian_body(const MatrixXd& Blist, const VectorXd& thetalist) {
    check_screw_list(Blist, thetalist, "jacobian_body");
    MatrixXd Jb = Blist;
    Matrix4d T = Matrix4d::Identity();
    for (Eigen::Index i = thetalist.size() - 2; i >= 0; --i) {
        T = T * matrix_exp6(vec_to_se3(Vector6d(Blist.col(i + 1) * -thetalist(i + 1))));
        Jb.col(i) = adjoint(T) * Vector6d(Blist.col(i));
    }
    return Jb;
}

MatrixXd jacobian_space(const MatrixXd& Slist, const VectorXd& thetalist) {
    check_screw_list(Slist, thetalist, "jacobian_space");
    MatrixXd Js = Slist;
    Matrix4d T = Matrix4d::Identity();
    for (Eigen::Index i = 1; i < thetalist.size(); ++i) {
        T = T * matrix_exp6(vec_to_se3(Vector6d(Slist.col(i - 1) * thetalist(i - 1))));
        Js.col(i) = adjoint(T) * Vector6d(Slist.col(i));
    }
    return Js;
}

// ============================================================ Chapter 6
IKResult ikin_body(const MatrixXd& Blist, const Matrix4d& M, const Matrix4d& T, const VectorXd& thetalist0,
                   double eomg, double ev) {
    VectorXd thetalist = thetalist0;
    int i = 0;
    const int maxiterations = 20;
    Vector6d Vb = se3_to_vec(matrix_log6(trans_inv(fkin_body(M, Blist, thetalist)) * T));
    bool err = Vb.head<3>().norm() > eomg || Vb.tail<3>().norm() > ev;
    while (err && i < maxiterations) {
        thetalist = thetalist + pinv(jacobian_body(Blist, thetalist)) * Vb;
        i = i + 1;
        Vb = se3_to_vec(matrix_log6(trans_inv(fkin_body(M, Blist, thetalist)) * T));
        err = Vb.head<3>().norm() > eomg || Vb.tail<3>().norm() > ev;
    }
    return {thetalist, !err};
}

IKResult ikin_space(const MatrixXd& Slist, const Matrix4d& M, const Matrix4d& T, const VectorXd& thetalist0,
                    double eomg, double ev) {
    VectorXd thetalist = thetalist0;
    int i = 0;
    const int maxiterations = 20;
    Matrix4d Tsb = fkin_space(M, Slist, thetalist);
    Vector6d Vs = adjoint(Tsb) * se3_to_vec(matrix_log6(trans_inv(Tsb) * T));
    bool err = Vs.head<3>().norm() > eomg || Vs.tail<3>().norm() > ev;
    while (err && i < maxiterations) {
        thetalist = thetalist + pinv(jacobian_space(Slist, thetalist)) * Vs;
        i = i + 1;
        Tsb = fkin_space(M, Slist, thetalist);
        Vs = adjoint(Tsb) * se3_to_vec(matrix_log6(trans_inv(Tsb) * T));
        err = Vs.head<3>().norm() > eomg || Vs.tail<3>().norm() > ev;
    }
    return {thetalist, !err};
}

// ============================================================ Chapter 8
Matrix6d ad(const Vector6d& V) {
    const Matrix3d omgmat = vec_to_so3(V.head<3>());
    Matrix6d A = Matrix6d::Zero();
    A.topLeftCorner<3, 3>() = omgmat;
    A.bottomLeftCorner<3, 3>() = vec_to_so3(V.tail<3>());
    A.bottomRightCorner<3, 3>() = omgmat;
    return A;
}

VectorXd inverse_dynamics(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& ddthetalist,
                          const Vector3d& g, const Vector6d& Ftip, const TransformList& Mlist,
                          const InertiaList& Glist, const MatrixXd& Slist) {
    check_chain(thetalist, Mlist, Glist, Slist, "inverse_dynamics");
    const Eigen::Index n = thetalist.size();
    require(dthetalist.size() == n && ddthetalist.size() == n,
            "inverse_dynamics: thetalist, dthetalist and ddthetalist must have the same length");
    const auto un = static_cast<std::size_t>(n);
    Matrix4d Mi = Matrix4d::Identity();
    MatrixXd Ai = MatrixXd::Zero(6, n);
    std::vector<Matrix6d> AdTi(un + 1);
    MatrixXd Vi = MatrixXd::Zero(6, n + 1);
    MatrixXd Vdi = MatrixXd::Zero(6, n + 1);
    Vdi.col(0).tail<3>() = -g;
    AdTi[un] = adjoint(trans_inv(Mlist[un]));
    Vector6d Fi = Ftip;
    VectorXd taulist = VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        Mi = Mi * Mlist[ui];
        Ai.col(i) = adjoint(trans_inv(Mi)) * Vector6d(Slist.col(i));
        AdTi[ui] = adjoint(matrix_exp6(vec_to_se3(Vector6d(Ai.col(i) * -thetalist(i)))) * trans_inv(Mlist[ui]));
        Vi.col(i + 1) = AdTi[ui] * Vector6d(Vi.col(i)) + Ai.col(i) * dthetalist(i);
        Vdi.col(i + 1) = AdTi[ui] * Vector6d(Vdi.col(i)) + Ai.col(i) * ddthetalist(i) +
                         ad(Vi.col(i + 1)) * Vector6d(Ai.col(i)) * dthetalist(i);
    }
    for (Eigen::Index i = n - 1; i >= 0; --i) {
        const auto ui = static_cast<std::size_t>(i);
        const Vector6d Vnext = Vi.col(i + 1);
        Fi = AdTi[ui + 1].transpose() * Fi + Glist[ui] * Vector6d(Vdi.col(i + 1)) -
             ad(Vnext).transpose() * (Glist[ui] * Vnext);
        taulist(i) = Fi.dot(Ai.col(i));
    }
    return taulist;
}

MatrixXd mass_matrix(const VectorXd& thetalist, const TransformList& Mlist, const InertiaList& Glist,
                     const MatrixXd& Slist) {
    const Eigen::Index n = thetalist.size();
    MatrixXd M = MatrixXd::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        VectorXd ddthetalist = VectorXd::Zero(n);
        ddthetalist(i) = 1;
        M.col(i) = inverse_dynamics(thetalist, VectorXd::Zero(n), ddthetalist, Vector3d::Zero(), Vector6d::Zero(),
                                    Mlist, Glist, Slist);
    }
    return M;
}

VectorXd vel_quadratic_forces(const VectorXd& thetalist, const VectorXd& dthetalist, const TransformList& Mlist,
                              const InertiaList& Glist, const MatrixXd& Slist) {
    return inverse_dynamics(thetalist, dthetalist, VectorXd::Zero(thetalist.size()), Vector3d::Zero(),
                            Vector6d::Zero(), Mlist, Glist, Slist);
}

VectorXd gravity_forces(const VectorXd& thetalist, const Vector3d& g, const TransformList& Mlist,
                        const InertiaList& Glist, const MatrixXd& Slist) {
    const Eigen::Index n = thetalist.size();
    return inverse_dynamics(thetalist, VectorXd::Zero(n), VectorXd::Zero(n), g, Vector6d::Zero(), Mlist, Glist,
                            Slist);
}

VectorXd end_effector_forces(const VectorXd& thetalist, const Vector6d& Ftip, const TransformList& Mlist,
                             const InertiaList& Glist, const MatrixXd& Slist) {
    const Eigen::Index n = thetalist.size();
    return inverse_dynamics(thetalist, VectorXd::Zero(n), VectorXd::Zero(n), Vector3d::Zero(), Ftip, Mlist, Glist,
                            Slist);
}

VectorXd forward_dynamics(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& taulist,
                          const Vector3d& g, const Vector6d& Ftip, const TransformList& Mlist,
                          const InertiaList& Glist, const MatrixXd& Slist) {
    return mass_matrix(thetalist, Mlist, Glist, Slist).inverse() *
           (taulist - vel_quadratic_forces(thetalist, dthetalist, Mlist, Glist, Slist) -
            gravity_forces(thetalist, g, Mlist, Glist, Slist) -
            end_effector_forces(thetalist, Ftip, Mlist, Glist, Slist));
}

std::pair<VectorXd, VectorXd> euler_step(const VectorXd& thetalist, const VectorXd& dthetalist,
                                         const VectorXd& ddthetalist, double dt) {
    return {thetalist + dt * dthetalist, dthetalist + dt * ddthetalist};
}

MatrixXd inverse_dynamics_trajectory(const MatrixXd& thetamat, const MatrixXd& dthetamat, const MatrixXd& ddthetamat,
                                     const Vector3d& g, const MatrixXd& Ftipmat, const TransformList& Mlist,
                                     const InertiaList& Glist, const MatrixXd& Slist) {
    require(Ftipmat.cols() == 6 && Ftipmat.rows() == thetamat.rows(),
            "inverse_dynamics_trajectory: Ftipmat must be N x 6");
    MatrixXd taumat = thetamat;
    for (Eigen::Index i = 0; i < thetamat.rows(); ++i)
        taumat.row(i) = inverse_dynamics(thetamat.row(i).transpose(), dthetamat.row(i).transpose(),
                                         ddthetamat.row(i).transpose(), g, Ftipmat.row(i).transpose(), Mlist,
                                         Glist, Slist)
                            .transpose();
    return taumat;
}

std::pair<MatrixXd, MatrixXd> forward_dynamics_trajectory(const VectorXd& thetalist0, const VectorXd& dthetalist0,
                                                          const MatrixXd& taumat, const Vector3d& g,
                                                          const MatrixXd& Ftipmat, const TransformList& Mlist,
                                                          const InertiaList& Glist, const MatrixXd& Slist, double dt,
                                                          int intRes) {
    require(Ftipmat.cols() == 6 && Ftipmat.rows() == taumat.rows(),
            "forward_dynamics_trajectory: Ftipmat must be N x 6");
    VectorXd thetalist = thetalist0, dthetalist = dthetalist0;
    MatrixXd thetamat = taumat, dthetamat = taumat;
    thetamat.row(0) = thetalist.transpose();
    dthetamat.row(0) = dthetalist.transpose();
    for (Eigen::Index i = 0; i < taumat.rows() - 1; ++i) {
        for (int j = 0; j < intRes; ++j) {
            const VectorXd ddthetalist = forward_dynamics(thetalist, dthetalist, taumat.row(i).transpose(), g,
                                                          Ftipmat.row(i).transpose(), Mlist, Glist, Slist);
            std::tie(thetalist, dthetalist) = euler_step(thetalist, dthetalist, ddthetalist, 1.0 * dt / intRes);
        }
        thetamat.row(i + 1) = thetalist.transpose();
        dthetamat.row(i + 1) = dthetalist.transpose();
    }
    return {thetamat, dthetamat};
}

// ============================================================ Chapter 9
double cubic_time_scaling(double Tf, double t) {
    return 3 * std::pow(1.0 * t / Tf, 2) - 2 * std::pow(1.0 * t / Tf, 3);
}

double quintic_time_scaling(double Tf, double t) {
    return 10 * std::pow(1.0 * t / Tf, 3) - 15 * std::pow(1.0 * t / Tf, 4) + 6 * std::pow(1.0 * t / Tf, 5);
}

namespace {
double time_scaling(int method, double Tf, double t) {
    return method == 3 ? cubic_time_scaling(Tf, t) : quintic_time_scaling(Tf, t);
}
}  // namespace

MatrixXd joint_trajectory(const VectorXd& thetastart, const VectorXd& thetaend, double Tf, int N, int method) {
    require(N >= 2, "joint_trajectory: N must be at least 2");
    const double timegap = Tf / (N - 1.0);
    MatrixXd traj(N, thetastart.size());
    for (int i = 0; i < N; ++i) {
        const double s = time_scaling(method, Tf, timegap * i);
        traj.row(i) = (s * thetaend + (1 - s) * thetastart).transpose();
    }
    return traj;
}

TransformList screw_trajectory(const Matrix4d& Xstart, const Matrix4d& Xend, double Tf, int N, int method) {
    require(N >= 2, "screw_trajectory: N must be at least 2");
    const double timegap = Tf / (N - 1.0);
    TransformList traj(static_cast<std::size_t>(N));
    const Matrix4d logm = matrix_log6(trans_inv(Xstart) * Xend);
    for (int i = 0; i < N; ++i) {
        const double s = time_scaling(method, Tf, timegap * i);
        traj[static_cast<std::size_t>(i)] = Xstart * matrix_exp6(logm * s);
    }
    return traj;
}

TransformList cartesian_trajectory(const Matrix4d& Xstart, const Matrix4d& Xend, double Tf, int N, int method) {
    require(N >= 2, "cartesian_trajectory: N must be at least 2");
    const double timegap = Tf / (N - 1.0);
    TransformList traj(static_cast<std::size_t>(N));
    const auto [Rstart, pstart] = trans_to_rp(Xstart);
    const auto [Rend, pend] = trans_to_rp(Xend);
    const Matrix3d logm = matrix_log3(Rstart.transpose() * Rend);
    for (int i = 0; i < N; ++i) {
        const double s = time_scaling(method, Tf, timegap * i);
        traj[static_cast<std::size_t>(i)] =
            rp_to_trans(Rstart * matrix_exp3(logm * s), s * pend + (1 - s) * pstart);
    }
    return traj;
}

// ============================================================ Chapter 11
VectorXd computed_torque(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& eint,
                         const Vector3d& g, const TransformList& Mlist, const InertiaList& Glist,
                         const MatrixXd& Slist, const VectorXd& thetalistd, const VectorXd& dthetalistd,
                         const VectorXd& ddthetalistd, double Kp, double Ki, double Kd) {
    const VectorXd e = thetalistd - thetalist;
    return mass_matrix(thetalist, Mlist, Glist, Slist) *
               (Kp * e + Ki * (eint + e) + Kd * (dthetalistd - dthetalist)) +
           inverse_dynamics(thetalist, dthetalist, ddthetalistd, g, Vector6d::Zero(), Mlist, Glist, Slist);
}

std::pair<MatrixXd, MatrixXd> simulate_control(const VectorXd& thetalist, const VectorXd& dthetalist,
                                               const Vector3d& g, const MatrixXd& Ftipmat,
                                               const TransformList& Mlist, const InertiaList& Glist,
                                               const MatrixXd& Slist, const MatrixXd& thetamatd,
                                               const MatrixXd& dthetamatd, const MatrixXd& ddthetamatd,
                                               const Vector3d& gtilde, const TransformList& Mtildelist,
                                               const InertiaList& Gtildelist, double Kp, double Ki, double Kd,
                                               double dt, int intRes) {
    const Eigen::Index N = thetamatd.rows(), n = thetamatd.cols();
    require(Ftipmat.cols() == 6 && Ftipmat.rows() == N, "simulate_control: Ftipmat must be N x 6");
    VectorXd thetacurrent = thetalist, dthetacurrent = dthetalist;
    VectorXd eint = VectorXd::Zero(n);
    MatrixXd taumat = MatrixXd::Zero(N, n), thetamat = MatrixXd::Zero(N, n);
    for (Eigen::Index i = 0; i < N; ++i) {
        const VectorXd taulist =
            computed_torque(thetacurrent, dthetacurrent, eint, gtilde, Mtildelist, Gtildelist, Slist,
                            thetamatd.row(i).transpose(), dthetamatd.row(i).transpose(),
                            ddthetamatd.row(i).transpose(), Kp, Ki, Kd);
        for (int j = 0; j < intRes; ++j) {
            const VectorXd ddthetalist = forward_dynamics(thetacurrent, dthetacurrent, taulist, g,
                                                          Ftipmat.row(i).transpose(), Mlist, Glist, Slist);
            std::tie(thetacurrent, dthetacurrent) =
                euler_step(thetacurrent, dthetacurrent, ddthetalist, 1.0 * dt / intRes);
        }
        taumat.row(i) = taulist.transpose();
        thetamat.row(i) = thetacurrent.transpose();
        eint = eint + dt * (thetamatd.row(i).transpose() - thetacurrent);
    }
    return {taumat, thetamat};
}

}  // namespace mr
