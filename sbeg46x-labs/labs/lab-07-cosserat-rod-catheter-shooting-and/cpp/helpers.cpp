// SBEG 46X Lab 7 -- C++ track: GIVEN code (do not edit).  The "Provided" functions of starter.py, translated:
//   load_catheter, load_csv               data/catheter.json, data/*.csv
//   tip_pose, contact_direction           small helpers
//   integrate_rod  (Python _integrate)    RK4 integration of your rod_rhs (Python: solve_ivp, rtol 1e-10)
//   rigid_guess    (Python _rigid_guess)  straight-rod base wrench
//   weights        (Python _weights), free_tip_angle (calls your tendon_curvature)
// and the small numerical tools that replace SciPy in C++:
//   fsolve         Newton with a finite-difference Jacobian             (scipy.optimize.fsolve)
//   least_squares  Levenberg-Marquardt, central-difference Jacobian     (scipy.optimize.least_squares)
//   solve_bvp      Lobatto IIIA (Simpson) collocation + Newton, mesh halving  (scipy.integrate.solve_bvp)
//   linspace       np.linspace
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
std::string data_path(const std::string& name) { return sbeg::lab_path("../data/" + name); }
}  // namespace

// ---------------------------------------------------------------------------
// Data files and small helpers
// ---------------------------------------------------------------------------
Catheter load_catheter() {
    const sbeg::json j = sbeg::load_json(data_path("catheter.json"));
    Catheter c;
    c.L = j.at("L").get<double>();
    c.EI = j.at("EI").get<double>();
    c.r = j.at("r").get<double>();
    c.tau_nominal = j.at("tau_nominal").get<double>();
    c.tau_max = j.at("tau_max").get<double>();
    c.sigma_p = j.at("sigma_p").get<double>();
    c.sigma_theta = j.at("sigma_theta").get<double>();
    c.description = j.value("description", std::string());
    return c;
}

Log load_csv(const std::string& name) {
    const sbeg::CsvTable t = sbeg::read_csv(data_path(name));
    Log log;
    for (std::size_t k = 0; k < t.header.size(); ++k) log[t.header[k]] = t.data.col(static_cast<Eigen::Index>(k));
    return log;
}

Eigen::Vector3d tip_pose(const RodSolution& sol) {
    if (sol.x.size() == 0 || sol.y.size() == 0 || sol.theta.size() == 0)
        throw std::invalid_argument("tip_pose: empty rod solution");
    return Eigen::Vector3d(sol.x(sol.x.size() - 1), sol.y(sol.y.size() - 1), sol.theta(sol.theta.size() - 1));
}

Eigen::Vector2d contact_direction(double theta_tip) { return Eigen::Vector2d(std::sin(theta_tip), -std::cos(theta_tip)); }

Eigen::VectorXd linspace(double a, double b, int n) {
    if (n < 2) throw std::invalid_argument("linspace: n must be >= 2");
    Eigen::VectorXd v(n);
    const double step = (b - a) / static_cast<double>(n - 1);
    for (int k = 0; k < n; ++k) v(k) = a + static_cast<double>(k) * step;
    v(n - 1) = b;
    return v;
}

// ---------------------------------------------------------------------------
// Rod integration (RK4) and the straight-rod guess
// ---------------------------------------------------------------------------
Eigen::MatrixXd integrate_rod(const Vector6d& z0, double L, double EI, const Eigen::Vector2d& f, double kappa0,
                             int n_eval) {
    if (n_eval < 2) throw std::invalid_argument("integrate_rod: n_eval must be >= 2");
    constexpr int TOTAL_STEPS = 4000;
    const int sub = std::max(1, (TOTAL_STEPS + n_eval - 2) / (n_eval - 1));  // RK4 steps between output points
    const Eigen::VectorXd s = linspace(0.0, L, n_eval);
    Eigen::MatrixXd Z(6, n_eval);
    Vector6d z = z0;
    Z.col(0) = z;
    for (int k = 1; k < n_eval; ++k) {
        const double h = (s(k) - s(k - 1)) / sub;
        for (int i = 0; i < sub; ++i) {
            const double si = s(k - 1) + i * h;
            const Vector6d k1 = rod_rhs(si, z, EI, f, kappa0);
            const Vector6d k2 = rod_rhs(si + 0.5 * h, z + 0.5 * h * k1, EI, f, kappa0);
            const Vector6d k3 = rod_rhs(si + 0.5 * h, z + 0.5 * h * k2, EI, f, kappa0);
            const Vector6d k4 = rod_rhs(si + h, z + h * k3, EI, f, kappa0);
            z += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        Z.col(k) = z;
    }
    return Z;
}

Eigen::Vector3d rigid_guess(double L, const Eigen::Vector2d& F, double M, const Eigen::Vector2d& f) {
    const Eigen::Vector2d n0 = F + f * L;
    const double m0 = M + L * F(1) + 0.5 * L * L * f(1);
    return Eigen::Vector3d(n0(0), n0(1), m0);
}

Eigen::Matrix3d weights(const Catheter& cat) {
    return Eigen::Vector3d(1.0 / cat.sigma_p, 1.0 / cat.sigma_p, 1.0 / cat.sigma_theta).asDiagonal();
}

double free_tip_angle(double tau, const Catheter& cat) { return tendon_curvature(tau, cat.r, cat.EI) * cat.L; }

// ---------------------------------------------------------------------------
// fsolve: Newton with a forward-difference Jacobian and step halving
// ---------------------------------------------------------------------------
Eigen::VectorXd fsolve(const VecFun& fun, const Eigen::VectorXd& x0, double xtol, int max_iter) {
    Eigen::VectorXd x = x0;
    Eigen::VectorXd r = fun(x);
    if (r.size() != x.size()) throw std::invalid_argument("fsolve: fun must return as many values as unknowns");
    const Eigen::Index n = x.size();
    for (int it = 0; it < max_iter; ++it) {
        if (r.cwiseAbs().maxCoeff() == 0.0) break;
        Eigen::MatrixXd J(n, n);
        for (Eigen::Index k = 0; k < n; ++k) {
            const double dx = 1.4901161193847656e-08 * std::max(std::abs(x(k)), 1e-3);
            Eigen::VectorXd xp = x;
            xp(k) += dx;
            J.col(k) = (fun(xp) - r) / dx;
        }
        const Eigen::VectorXd step = J.colPivHouseholderQr().solve(r);
        double alpha = 1.0;
        Eigen::VectorXd x_new = x - step, r_new = fun(x_new);
        for (int h = 0; h < 30 && !(r_new.norm() < r.norm()) && r.norm() > 0.0; ++h) {
            alpha *= 0.5;
            x_new = x - alpha * step;
            r_new = fun(x_new);
        }
        const bool small = (alpha * step).norm() <= xtol * (xtol + x.norm());
        if (!(r_new.norm() <= r.norm())) break;  // no progress possible (round-off level)
        x = x_new;
        r = r_new;
        if (small) break;
    }
    return x;
}

// ---------------------------------------------------------------------------
// least_squares: Levenberg-Marquardt with a central-difference Jacobian (scaled unknowns u = x / x_scale)
// ---------------------------------------------------------------------------
Eigen::VectorXd least_squares(const VecFun& fun, const Eigen::VectorXd& x0, double x_scale, double xtol,
                              int max_iter) {
    if (!(x_scale > 0.0)) throw std::invalid_argument("least_squares: x_scale must be > 0");
    const Eigen::Index n = x0.size();
    Eigen::VectorXd x = x0;
    Eigen::VectorXd r = fun(x);
    double cost = 0.5 * r.squaredNorm();
    double lambda = 1e-3;
    for (int it = 0; it < max_iter; ++it) {
        Eigen::MatrixXd J(r.size(), n);  // Jacobian w.r.t. the scaled unknowns
        const double du = 1e-6;
        for (Eigen::Index k = 0; k < n; ++k) {
            Eigen::VectorXd xp = x, xm = x;
            xp(k) += du * x_scale;
            xm(k) -= du * x_scale;
            J.col(k) = (fun(xp) - fun(xm)) / (2.0 * du);
        }
        const Eigen::MatrixXd A = J.transpose() * J;
        const Eigen::VectorXd g = J.transpose() * r;
        if (g.cwiseAbs().maxCoeff() <= 1e-14 * std::max(1.0, cost)) break;
        bool accepted = false, converged = false;
        for (int tries = 0; tries < 40; ++tries) {
            Eigen::MatrixXd Ad = A;
            for (Eigen::Index k = 0; k < n; ++k) Ad(k, k) += lambda * std::max(A(k, k), 1e-300);
            const Eigen::VectorXd step_u = -Ad.ldlt().solve(g);
            const Eigen::VectorXd x_new = x + x_scale * step_u;
            const Eigen::VectorXd r_new = fun(x_new);
            const double cost_new = 0.5 * r_new.squaredNorm();
            const double step_rel = step_u.norm() / (xtol + x.norm() / x_scale);
            if (cost_new <= cost) {
                converged = step_rel <= xtol || cost - cost_new <= 1e-15 * cost;
                x = x_new;
                r = r_new;
                cost = cost_new;
                lambda = std::max(lambda / 10.0, 1e-12);
                accepted = true;
                break;
            }
            if (step_rel <= xtol) {  // even tiny steps do not reduce the cost: at the minimum
                converged = true;
                break;
            }
            lambda *= 10.0;
        }
        if (converged || !accepted) break;
    }
    return x;
}

// ---------------------------------------------------------------------------
// solve_bvp: Lobatto IIIA (Simpson) collocation, Newton with a finite-difference block Jacobian, mesh halving
// ---------------------------------------------------------------------------
namespace {
using OdeFun = std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>;
using BcFun = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/// Collocation defect of interval [x_i, x_i + h] (Simpson's rule with the Hermite midpoint).
Eigen::VectorXd defect(const OdeFun& fun, double xi, double h, const Eigen::VectorXd& yi, const Eigen::VectorXd& yj) {
    const Eigen::VectorXd fi = fun(xi, yi), fj = fun(xi + h, yj);
    const Eigen::VectorXd ym = 0.5 * (yi + yj) - h / 8.0 * (fj - fi);
    const Eigen::VectorXd fm = fun(xi + 0.5 * h, ym);
    return yj - yi - h / 6.0 * (fi + 4.0 * fm + fj);
}

Eigen::VectorXd residual(const OdeFun& fun, const BcFun& bc, const Eigen::VectorXd& x, const Eigen::MatrixXd& Y) {
    const Eigen::Index m = Y.rows(), n = Y.cols();
    Eigen::VectorXd R(m * n);
    const Eigen::VectorXd b = bc(Y.col(0), Y.col(n - 1));
    if (b.size() != m) throw std::invalid_argument("solve_bvp: bc must return as many residuals as states");
    R.head(m) = b;
    for (Eigen::Index i = 0; i + 1 < n; ++i) R.segment(m * (i + 1), m) = defect(fun, x(i), x(i + 1) - x(i), Y.col(i), Y.col(i + 1));
    return R;
}

/// Newton's method on the collocation equations; Y is updated in place.  Returns true on convergence.
bool newton_collocation(const OdeFun& fun, const BcFun& bc, const Eigen::VectorXd& x, Eigen::MatrixXd& Y) {
    const Eigen::Index m = Y.rows(), n = Y.cols(), N = m * n;
    Eigen::VectorXd R = residual(fun, bc, x, Y);
    for (int it = 0; it < 30; ++it) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(N, N);
        for (Eigen::Index jn = 0; jn < n; ++jn) {
            for (Eigen::Index c = 0; c < m; ++c) {
                const double d = 1.4901161193847656e-08 * (1.0 + std::abs(Y(c, jn)));
                Eigen::MatrixXd Yp = Y;
                Yp(c, jn) += d;
                const Eigen::Index col = m * jn + c;
                if (jn == 0 || jn == n - 1) J.block(0, col, m, 1) = (bc(Yp.col(0), Yp.col(n - 1)) - R.head(m)) / d;
                if (jn > 0) {
                    const Eigen::Index i = jn - 1;
                    J.block(m * (i + 1), col, m, 1) =
                        (defect(fun, x(i), x(i + 1) - x(i), Yp.col(i), Yp.col(i + 1)) - R.segment(m * (i + 1), m)) / d;
                }
                if (jn + 1 < n) {
                    const Eigen::Index i = jn;
                    J.block(m * (i + 1), col, m, 1) =
                        (defect(fun, x(i), x(i + 1) - x(i), Yp.col(i), Yp.col(i + 1)) - R.segment(m * (i + 1), m)) / d;
                }
            }
        }
        const Eigen::VectorXd du = J.partialPivLu().solve(R);
        if (!du.allFinite()) return false;
        double alpha = 1.0;
        Eigen::MatrixXd Y_new;
        Eigen::VectorXd R_new;
        for (int h = 0; h < 20; ++h) {
            Y_new = Y - alpha * Eigen::Map<const Eigen::MatrixXd>(du.data(), m, n);
            R_new = residual(fun, bc, x, Y_new);
            if (R_new.allFinite() && R_new.norm() <= R.norm()) break;
            alpha *= 0.5;
        }
        if (!R_new.allFinite()) return false;
        const double step = (alpha * du).cwiseAbs().maxCoeff();
        const double scale = 1.0 + Y.cwiseAbs().maxCoeff();
        Y = Y_new;
        R = R_new;
        if (step <= 1e-13 * scale || R.cwiseAbs().maxCoeff() <= 1e-15 * scale) return true;
    }
    return R.cwiseAbs().maxCoeff() <= 1e-10 * (1.0 + Y.cwiseAbs().maxCoeff());
}

Eigen::MatrixXd derivatives(const OdeFun& fun, const Eigen::VectorXd& x, const Eigen::MatrixXd& Y) {
    Eigen::MatrixXd D(Y.rows(), Y.cols());
    for (Eigen::Index i = 0; i < Y.cols(); ++i) D.col(i) = fun(x(i), Y.col(i));
    return D;
}
}  // namespace

Eigen::MatrixXd BvpResult::sol(const Eigen::VectorXd& s) const {
    const Eigen::Index n = x.size();
    if (n < 2 || y.cols() != n || yp.cols() != n) throw std::logic_error("BvpResult::sol: no solution stored");
    Eigen::MatrixXd out(y.rows(), s.size());
    for (Eigen::Index k = 0; k < s.size(); ++k) {
        Eigen::Index i = static_cast<Eigen::Index>(std::upper_bound(x.data(), x.data() + n, s(k)) - x.data()) - 1;
        i = std::clamp<Eigen::Index>(i, 0, n - 2);
        const double h = x(i + 1) - x(i), t = (s(k) - x(i)) / h;
        const double h00 = (1 + 2 * t) * (1 - t) * (1 - t), h10 = t * (1 - t) * (1 - t);
        const double h01 = t * t * (3 - 2 * t), h11 = t * t * (t - 1);
        out.col(k) = h00 * y.col(i) + h10 * h * yp.col(i) + h01 * y.col(i + 1) + h11 * h * yp.col(i + 1);
    }
    return out;
}

BvpResult solve_bvp(const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& fun,
                    const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& bc,
                    const Eigen::VectorXd& s, const Eigen::MatrixXd& Z0, double tol, int max_nodes) {
    const Eigen::Index n0 = s.size();
    if (n0 < 2 || Z0.cols() != n0 || Z0.rows() < 1)
        throw std::invalid_argument("solve_bvp: Z0 must have one column per mesh node (>= 2 nodes)");
    for (Eigen::Index i = 0; i + 1 < n0; ++i)
        if (!(s(i + 1) > s(i))) throw std::invalid_argument("solve_bvp: the mesh must be strictly increasing");
    BvpResult res;
    res.x = s;
    res.y = Z0;
    Eigen::MatrixXd Y_prev;
    for (int level = 0;; ++level) {
        if (!newton_collocation(fun, bc, res.x, res.y)) {
            res.yp = derivatives(fun, res.x, res.y);
            res.message = "Newton iteration on the collocation equations did not converge";
            return res;
        }
        res.yp = derivatives(fun, res.x, res.y);
        if (level > 0) {  // compare with the previous (coarser) solution at its nodes = every other node
            double err = 0.0;
            for (Eigen::Index i = 0; i < Y_prev.cols(); ++i)
                for (Eigen::Index c = 0; c < Y_prev.rows(); ++c)
                    err = std::max(err, std::abs(res.y(c, 2 * i) - Y_prev(c, i)) / (1.0 + std::abs(res.y(c, 2 * i))));
            if (err <= tol) {
                res.success = true;
                res.message = "The algorithm converged to the desired accuracy.";
                return res;
            }
        }
        const Eigen::Index n = res.x.size(), n2 = 2 * n - 1;
        if (n2 > max_nodes) {
            res.message = "The maximum number of mesh nodes is exceeded.";
            return res;
        }
        Eigen::VectorXd x2(n2);
        for (Eigen::Index i = 0; i < n; ++i) x2(2 * i) = res.x(i);
        for (Eigen::Index i = 0; i + 1 < n; ++i) x2(2 * i + 1) = 0.5 * (res.x(i) + res.x(i + 1));
        Y_prev = res.y;
        res.y = res.sol(x2);
        res.x = x2;
    }
}

}  // namespace lab
