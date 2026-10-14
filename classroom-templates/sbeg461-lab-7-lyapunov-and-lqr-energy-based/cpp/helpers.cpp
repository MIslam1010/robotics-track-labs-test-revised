// SBEG 461 Lab 7 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_config, plant_2r, plant_course_arm, scaled_model         data/lab07_config.json and the two robots
//   mass_matrix, coriolis_vector, gravity_torque, potential_energy, forward_dynamics   the dynamics
//   pd_gains_from_inertia, is_hurwitz, time_grid, rk4_step, simulate                   gains and the simulator
//   care_solve                                                     the Riccati solver (Python: scipy.linalg)
//
// These are the "Provided" functions of starter.py, translated to C++, plus care_solve, which replaces
// scipy.linalg.solve_continuous_are (C++ has no SciPy).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

namespace {
constexpr double G0 = 9.81;

Eigen::Matrix4d trans(double x, double y, double z) {
    return mr::rp_to_trans(Eigen::Matrix3d::Identity(), Eigen::Vector3d(x, y, z));
}

/// Spatial inertia of a uniform rod along x (radius r) in its CoM frame: MR Studio rod_inertia.
mr::Matrix6d rod_inertia(double m, double L, double r) {
    mr::Vector6d d;
    const double Iyy = m * (3 * r * r + L * L) / 12.0;
    d << 0.5 * m * r * r, Iyy, Iyy, m, m, m;
    return d.asDiagonal();
}

/// The planar 2R of robots.py (shared model planar_2r), friction b.
Robot planar_2r(const Eigen::VectorXd& b) {
    const double L1 = 0.6, L2 = 0.4, m1 = 2.0, m2 = 1.0, r = 0.025;
    Robot rb;
    rb.name = "Planar 2R arm";
    rb.key = "planar_2r";
    rb.n = 2;
    rb.Slist.resize(6, 2);
    rb.Slist.col(0) << 0, 0, 1, 0, 0, 0;
    rb.Slist.col(1) << 0, 0, 1, 0, -L1, 0;
    rb.M = trans(L1 + L2, 0, 0);
    rb.Mlist = {trans(L1 / 2, 0, 0), trans((L1 + L2) / 2, 0, 0), trans(L2 / 2, 0, 0)};
    rb.Glist = {rod_inertia(m1, L1, r), rod_inertia(m2, L2, r)};
    rb.g = Eigen::Vector3d(0.0, -G0, 0.0);
    rb.tau_max = Eigen::Vector2d(40.0, 15.0);
    rb.q_home = Eigen::Vector2d(0.6, 0.9);
    rb.b = b;
    rb.has_closed_form = true;
    rb.cf = ClosedForm2R{L1, L2, m1, m2, L1 / 2, L2 / 2, rb.Glist[0](2, 2), rb.Glist[1](2, 2), G0};
    return rb;
}

/// log |det(Z)| from an LU factorisation (avoids overflow of det for 24 x 24 matrices).
double log_abs_det(const Eigen::PartialPivLU<Eigen::MatrixXd>& lu) {
    const Eigen::MatrixXd& U = lu.matrixLU();
    double s = 0.0;
    for (Eigen::Index i = 0; i < U.rows(); ++i) s += std::log(std::abs(U(i, i)));
    return s;
}
}  // namespace

sbeg::json load_config() { return sbeg::load_json(sbeg::lab_path("../data/lab07_config.json")); }

Robot plant_2r(const sbeg::json& cfg, bool friction) {
    return planar_2r(friction ? sbeg::as_vec(cfg.at("planar_2r").at("b")) : Eigen::VectorXd::Zero(2));
}

Robot plant_course_arm(const sbeg::json& cfg) {
    Robot r;
    static_cast<course_arm::Robot&>(r) = course_arm::course_arm();
    r.key = "course_arm";
    r.b = sbeg::as_vec(cfg.at("course_arm").at("b"));
    r.has_closed_form = false;
    return r;
}

Robot scaled_model(const Robot& robot, double mass_scale) {
    Robot r = robot;
    for (auto& G : r.Glist) G = (mass_scale * G).eval();
    if (r.has_closed_form) {
        r.cf.m1 *= mass_scale;
        r.cf.m2 *= mass_scale;
        r.cf.I1 *= mass_scale;
        r.cf.I2 *= mass_scale;
    }
    return r;
}

Eigen::MatrixXd mass_matrix(const Robot& robot, const Eigen::VectorXd& q) {
    if (robot.has_closed_form) {
        const ClosedForm2R& c = robot.cf;
        const double c2 = std::cos(q(1));
        const double m22 = c.I2 + c.m2 * c.lc2 * c.lc2;
        const double m12 = m22 + c.m2 * c.L1 * c.lc2 * c2;
        const double m11 = c.I1 + c.m1 * c.lc1 * c.lc1 + c.I2 +
                           c.m2 * (c.L1 * c.L1 + c.lc2 * c.lc2 + 2 * c.L1 * c.lc2 * c2);
        Eigen::MatrixXd M(2, 2);
        M << m11, m12, m12, m22;
        return M;
    }
    return mr::mass_matrix(q, robot.Mlist, robot.Glist, robot.Slist);
}

Eigen::VectorXd coriolis_vector(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
    if (robot.has_closed_form) {
        const ClosedForm2R& c = robot.cf;
        const double hh = -c.m2 * c.L1 * c.lc2 * std::sin(q(1));
        Eigen::VectorXd v(2);
        v << hh * (2 * dq(0) * dq(1) + dq(1) * dq(1)), -hh * dq(0) * dq(0);
        return v;
    }
    return mr::vel_quadratic_forces(q, dq, robot.Mlist, robot.Glist, robot.Slist);
}

Eigen::VectorXd gravity_torque(const Robot& robot, const Eigen::VectorXd& q) {
    if (robot.has_closed_form) {
        const ClosedForm2R& c = robot.cf;
        const double g2 = c.m2 * c.lc2 * c.g0 * std::cos(q(0) + q(1));
        Eigen::VectorXd v(2);
        v << (c.m1 * c.lc1 + c.m2 * c.L1) * c.g0 * std::cos(q(0)) + g2, g2;
        return v;
    }
    return mr::gravity_forces(q, robot.g, robot.Mlist, robot.Glist, robot.Slist);
}

double potential_energy(const Robot& robot, const Eigen::VectorXd& q) {
    if (robot.has_closed_form) {
        const ClosedForm2R& c = robot.cf;
        const double y1 = std::sin(q(0)), y12 = std::sin(q(0) + q(1));
        return c.g0 * (c.m1 * c.lc1 * y1 + c.m2 * (c.L1 * y1 + c.lc2 * y12));
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity(), Mhome = Eigen::Matrix4d::Identity();
    double P = 0.0;
    for (int i = 0; i < robot.n; ++i) {
        const mr::Vector6d S = robot.Slist.col(i) * q(i);
        T = (T * mr::matrix_exp6(mr::vec_to_se3(S))).eval();
        Mhome = (Mhome * robot.Mlist[static_cast<std::size_t>(i)]).eval();
        const Eigen::Matrix4d Ti = T * Mhome;
        P -= robot.Glist[static_cast<std::size_t>(i)](3, 3) * robot.g.dot(Ti.block<3, 1>(0, 3));
    }
    return P;
}

Eigen::VectorXd forward_dynamics(const Robot& robot, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& tau) {
    const Eigen::VectorXd rhs = tau - coriolis_vector(robot, q, dq) - gravity_torque(robot, q) - robot.b.cwiseProduct(dq);
    return mass_matrix(robot, q).partialPivLu().solve(rhs);
}

PDGains pd_gains_from_inertia(const Robot& robot, const Eigen::VectorXd& q, double zeta, double wn) {
    const Eigen::VectorXd m = mass_matrix(robot, q).diagonal();
    PDGains g;
    g.Kp = (m * wn * wn).asDiagonal();
    g.Kd = (2 * zeta * wn * m - robot.b).asDiagonal();
    return g;
}

bool is_hurwitz(const Eigen::MatrixXd& A) {
    Eigen::EigenSolver<Eigen::MatrixXd> es(A, false);
    return es.eigenvalues().real().maxCoeff() < 0.0;
}

Eigen::VectorXd time_grid(double T, double dt) {
    const auto n = static_cast<Eigen::Index>(std::llround(T / dt));
    Eigen::VectorXd t(n + 1);
    for (Eigen::Index k = 0; k <= n; ++k) t(k) = static_cast<double>(k) * dt;
    return t;
}

Eigen::VectorXd rk4_step(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x,
                         double h) {
    const Eigen::VectorXd k1 = f(x);
    const Eigen::VectorXd k2 = f(x + 0.5 * h * k1);
    const Eigen::VectorXd k3 = f(x + 0.5 * h * k2);
    const Eigen::VectorXd k4 = f(x + h * k3);
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
}

SimResult simulate(const Robot& plant, const Policy& policy, const Eigen::VectorXd& q0, const Eigen::VectorXd& dq0,
                   double T, double dt) {
    const int n = plant.n;
    SimResult s;
    s.t = time_grid(T, dt);
    const Eigen::Index N = s.t.size();
    s.q.resize(N, n);
    s.dq.resize(N, n);
    s.tau.resize(N, n);
    Eigen::VectorXd x(2 * n);
    x << q0, dq0;
    for (Eigen::Index k = 0; k < N; ++k) {
        s.q.row(k) = x.head(n).transpose();
        s.dq.row(k) = x.tail(n).transpose();
        const Eigen::VectorXd u = policy(s.t(k), x.head(n), x.tail(n));
        if (u.size() != n) throw std::invalid_argument("simulate: the policy must return n torques");
        s.tau.row(k) = u.transpose();
        if (k + 1 < N) {
            x = rk4_step(
                [&](const Eigen::VectorXd& z) {
                    Eigen::VectorXd dz(2 * n);
                    dz << z.tail(n), forward_dynamics(plant, z.head(n), z.tail(n), u);
                    return dz;
                },
                x, dt);
        }
    }
    return s;
}

Eigen::MatrixXd care_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R) {
    const Eigen::Index n = A.rows();
    if (A.cols() != n || B.rows() != n || Q.rows() != n || Q.cols() != n || R.rows() != B.cols() ||
        R.cols() != B.cols())
        throw std::invalid_argument("care_solve: wrong matrix sizes");
    // Hamiltonian H = [[A, -B R^-1 B^T], [-Q, -A^T]]; its stable invariant subspace is range [I; P].
    Eigen::MatrixXd H(2 * n, 2 * n);
    H << A, -B * R.llt().solve(B.transpose()), -Q, -A.transpose();
    // Matrix sign function by the scaled Newton iteration Z <- (Z / c + c Z^-1) / 2, c = |det Z|^(1 / 2n).
    Eigen::MatrixXd Z = H;
    bool converged = false;
    for (int it = 0; it < 100; ++it) {
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(Z);
        const double c = std::exp(log_abs_det(lu) / static_cast<double>(2 * n));
        const Eigen::MatrixXd Zn = 0.5 * (Z / c + c * lu.inverse());
        const double change = (Zn - Z).lpNorm<1>();
        Z = Zn;
        if (change <= 1e-13 * Z.lpNorm<1>()) {
            converged = true;
            break;
        }
    }
    if (!converged) throw std::invalid_argument("care_solve: sign iteration did not converge");
    // W = sign(H): (W + I) [I; P] = 0  ->  [W12; W22 + I] P = -[W11 + I; W21]  (least squares)
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd lhs(2 * n, n), rhs(2 * n, n);
    lhs << Z.topRightCorner(n, n), Z.bottomRightCorner(n, n) + I;
    rhs << -(Z.topLeftCorner(n, n) + I), -Z.bottomLeftCorner(n, n);
    Eigen::MatrixXd P = lhs.colPivHouseholderQr().solve(rhs);
    P = (0.5 * (P + P.transpose())).eval();
    return P;
}

}  // namespace lab
