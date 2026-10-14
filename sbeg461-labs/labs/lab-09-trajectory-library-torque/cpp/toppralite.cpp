// SBEG 461 Lab 9 -- C++ track: GIVEN code (do not edit).  toppralite: TOPP-RA as in toppra 0.6 (see toppralite.hpp).
#include "toppralite.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace toppralite {

namespace {
constexpr double TINY = 1e-8;       // toppra.constants.TINY
constexpr double SMALL = 1e-5;      // toppra.constants.SMALL
constexpr double ZERO_COEF = 1e-10; // |coefficient of sdd| below this: the row bounds x = sdot^2 only
constexpr double JVEL_MAXSD = 1e8;  // toppra.constants.JVEL_MAXSD (largest sdot of a velocity constraint)
constexpr int MAX_TRIES = 10;       // toppra.constants.MAX_TRIES
const double INF = std::numeric_limits<double>::infinity();

/// One half-plane  alpha * u + beta * x <= gamma  in the variables u = sdd, x = sdot^2.
struct Row {
    double alpha, beta, gamma;
};

void check_lim(const Eigen::MatrixXd& lim, const char* what) {
    if (lim.cols() != 2) throw std::invalid_argument(std::string(what) + ": limits must be an n x 2 matrix");
}

/// Append the two half-planes of  lo <= a u + b x + c <= hi.
void add_two_sided(std::vector<Row>& rows, double a, double b, double c, double lo, double hi) {
    rows.push_back({a, b, hi - c});
    rows.push_back({-a, -b, c - lo});
}
}  // namespace

Constraint joint_torque_constraint(InverseDynamics inv_dyn, const Eigen::MatrixXd& tau_lim,
                                   const Eigen::VectorXd& fs_coef, Discretization discretization) {
    check_lim(tau_lim, "joint_torque_constraint");
    if (fs_coef.size() != tau_lim.rows())
        throw std::invalid_argument("joint_torque_constraint: fs_coef must have one entry per joint");
    Constraint c;
    c.kind = Constraint::Kind::Torque;
    c.inv_dyn = std::move(inv_dyn);
    c.lim = tau_lim;
    c.fs_coef = fs_coef;
    c.discretization = discretization;
    return c;
}

Constraint joint_velocity_constraint(const Eigen::MatrixXd& vlim) {
    check_lim(vlim, "joint_velocity_constraint");
    for (Eigen::Index j = 0; j < vlim.rows(); ++j)
        if (!(vlim(j, 0) < vlim(j, 1))) throw std::invalid_argument("joint_velocity_constraint: bad velocity limits");
    Constraint c;
    c.kind = Constraint::Kind::Velocity;
    c.lim = vlim;
    return c;
}

Constraint joint_acceleration_constraint(const Eigen::MatrixXd& alim, Discretization discretization) {
    check_lim(alim, "joint_acceleration_constraint");
    Constraint c;
    c.kind = Constraint::Kind::Acceleration;
    c.lim = alim;
    c.discretization = discretization;
    return c;
}

// ------------------------------------------------------------------ ParametrizeConstAccel
ParametrizeConstAccel::ParametrizeConstAccel(Path path, const Eigen::VectorXd& gridpoints,
                                             const Eigen::VectorXd& velocities)
    : path_(std::move(path)), ss_(gridpoints), vs_(velocities) {
    if (ss_.size() != vs_.size() || ss_.size() < 2)
        throw std::invalid_argument("ParametrizeConstAccel: gridpoints and velocities must have the same size >= 2");
    const Eigen::Index N = ss_.size() - 1;
    ts_ = Eigen::VectorXd::Zero(N + 1);
    us_ = Eigen::VectorXd::Zero(N);
    for (Eigen::Index i = 0; i < N; ++i) {
        const double ds = ss_(i + 1) - ss_(i);
        us_(i) = 0.5 * (vs_(i + 1) * vs_(i + 1) - vs_(i) * vs_(i)) / ds;
        ts_(i + 1) = ts_(i) + 2.0 * ds / (vs_(i) + vs_(i + 1));
    }
}

Eigen::VectorXd ParametrizeConstAccel::operator()(double t, int order) const {
    // segment: np.searchsorted(ts, t, side="right") - 1, the last one is used beyond the end
    const Eigen::Index N = us_.size();
    Eigen::Index idx = static_cast<Eigen::Index>(std::upper_bound(ts_.data(), ts_.data() + ts_.size(), t) -
                                                 ts_.data()) - 1;
    if (idx >= N) idx = N - 1;
    if (idx < 0) idx = 0;
    const double dt = t - ts_(idx);
    const double u = us_(idx);
    const double v = vs_(idx) + dt * u;
    const double s = ss_(idx) + dt * vs_(idx) + 0.5 * dt * dt * u;
    if (order == 0) return path_(s, 0);
    if (order == 1) return path_(s, 1) * v;
    if (order == 2) return path_(s, 2) * (v * v) + path_(s, 1) * u;
    throw std::invalid_argument("ParametrizeConstAccel: order must be 0, 1 or 2");
}

// ------------------------------------------------------------------ TOPPRA
TOPPRA::TOPPRA(std::vector<Constraint> constraint_list, Path path, const Eigen::VectorXd& gridpoints)
    : constraints_(std::move(constraint_list)), path_(std::move(path)) {
    if (gridpoints.size() < 2) throw std::invalid_argument("TOPPRA: at least two gridpoints are needed");
    for (Eigen::Index i = 0; i + 1 < gridpoints.size(); ++i)
        if (!(gridpoints(i + 1) > gridpoints(i))) throw std::invalid_argument("TOPPRA: bad input gridpoints");
    data_.gridpoints = gridpoints;
}

std::shared_ptr<ParametrizeConstAccel> TOPPRA::compute_trajectory(double sd_start, double sd_end) {
    if (sd_start < 0 || sd_end < 0) throw std::invalid_argument("TOPPRA: negative path velocities");
    const Eigen::VectorXd& grid = data_.gridpoints;
    const Eigen::Index N = grid.size() - 1;
    data_.ok = false;
    data_.sd_vec.resize(0);
    data_.sdd_vec.resize(0);

    // path samples
    std::vector<Eigen::VectorXd> p(static_cast<std::size_t>(N + 1)), ps(p.size()), pss(p.size());
    for (Eigen::Index i = 0; i <= N; ++i) {
        const auto k = static_cast<std::size_t>(i);
        p[k] = path_(grid(i), 0);
        ps[k] = path_(grid(i), 1);
        pss[k] = path_(grid(i), 2);
    }
    const Eigen::Index n = p[0].size();

    // canonical rows of every stage, and the upper bound on x from velocity constraints
    std::vector<std::vector<Row>> rows(static_cast<std::size_t>(N + 1));
    Eigen::VectorXd xmax = Eigen::VectorXd::Constant(N + 1, JVEL_MAXSD * JVEL_MAXSD);
    for (const Constraint& con : constraints_) {
        if (con.lim.rows() != n) throw std::invalid_argument("TOPPRA: constraint dof is not equal to path dof");
        if (con.kind == Constraint::Kind::Velocity) {
            for (Eigen::Index i = 0; i <= N; ++i) {
                double sdmax = JVEL_MAXSD;
                const Eigen::VectorXd& d = ps[static_cast<std::size_t>(i)];
                for (Eigen::Index j = 0; j < n; ++j) {
                    if (d(j) > 0) sdmax = std::min(sdmax, con.lim(j, 1) / d(j));
                    if (d(j) < 0) sdmax = std::min(sdmax, con.lim(j, 0) / d(j));
                }
                xmax(i) = std::min(xmax(i), sdmax * sdmax);
            }
            continue;
        }
        // a, b, c of every grid point (collocation form)
        std::vector<Eigen::VectorXd> A(p.size()), B(p.size()), C(p.size());
        for (std::size_t k = 0; k < p.size(); ++k) {
            if (con.kind == Constraint::Kind::Torque) {
                const Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
                C[k] = con.inv_dyn(p[k], z, z);
                A[k] = con.inv_dyn(p[k], z, ps[k]) - C[k];
                B[k] = con.inv_dyn(p[k], ps[k], pss[k]) - C[k];
                for (Eigen::Index j = 0; j < n; ++j) {
                    const double sg = ps[k](j) > 0 ? 1.0 : (ps[k](j) < 0 ? -1.0 : 0.0);
                    C[k](j) += con.fs_coef(j) * sg;
                }
            } else {  // acceleration: qdd = theta' sdd + theta'' sdot^2
                A[k] = ps[k];
                B[k] = pss[k];
                C[k] = Eigen::VectorXd::Zero(n);
            }
        }
        for (Eigen::Index i = 0; i <= N; ++i) {
            const auto k = static_cast<std::size_t>(i);
            for (Eigen::Index j = 0; j < n; ++j) add_two_sided(rows[k], A[k](j), B[k](j), C[k](j), con.lim(j, 0), con.lim(j, 1));
            if (con.discretization == Discretization::Interpolation) {
                // the constraint at the next grid point, with x_{i+1} = x_i + 2 ds u_i (toppra: identical F, g)
                if (i < N) {
                    const double ds = grid(i + 1) - grid(i);
                    for (Eigen::Index j = 0; j < n; ++j)
                        add_two_sided(rows[k], A[k + 1](j) + 2.0 * ds * B[k + 1](j), B[k + 1](j), C[k + 1](j),
                                      con.lim(j, 0), con.lim(j, 1));
                } else {
                    for (Eigen::Index j = 0; j < n; ++j)
                        add_two_sided(rows[k], A[k](j), B[k](j), C[k](j), con.lim(j, 0), con.lim(j, 1));
                }
            }
        }
    }

    // stage rows + the link to the next stage: K_next_min <= x + 2 ds u <= K_next_max
    auto stage_rows = [&](Eigen::Index i, double kmin, double kmax) {
        std::vector<Row> r = rows[static_cast<std::size_t>(i)];
        const double ds = grid(i + 1) - grid(i);
        r.push_back({2.0 * ds, 1.0, kmax});
        r.push_back({-2.0 * ds, -1.0, -kmin});
        return r;
    };

    // backward pass: K_i = [min x, max x] over (u, x) feasible at stage i (u eliminated: Fourier-Motzkin)
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(N + 1, 2);
    K(N, 0) = sd_end * sd_end;
    K(N, 1) = sd_end * sd_end;
    bool controllable = true;
    for (Eigen::Index i = N - 1; i >= 0; --i) {
        const std::vector<Row> r = stage_rows(i, K(i + 1, 0), K(i + 1, 1));
        double lo = 0.0, hi = xmax(i);
        auto bound = [&](double beta, double gamma) {  // beta * x <= gamma
            if (beta > 0) hi = std::min(hi, gamma / beta);
            else if (beta < 0) lo = std::max(lo, gamma / beta);
            else if (gamma < 0) lo = INF;  // 0 <= negative: infeasible
        };
        for (const Row& a : r)
            if (std::abs(a.alpha) <= ZERO_COEF) bound(a.beta, a.gamma);
        for (const Row& a : r) {
            if (a.alpha <= ZERO_COEF) continue;
            for (const Row& b : r) {
                if (b.alpha >= -ZERO_COEF) continue;
                // (-b.alpha) * row a + a.alpha * row b eliminates u
                bound(-b.alpha * a.beta + a.alpha * b.beta, -b.alpha * a.gamma + a.alpha * b.gamma);
            }
        }
        if (!(lo <= hi)) {
            controllable = false;
            break;
        }
        K(i, 0) = lo;
        K(i, 1) = hi;
    }
    data_.K = K;
    const double x_start = sd_start * sd_start;
    if (!controllable || x_start + SMALL < K(0, 0) || K(0, 1) + SMALL < x_start) return nullptr;

    // forward pass: the largest u (hence x_{i+1}) at the current x_i
    Eigen::VectorXd xs = Eigen::VectorXd::Zero(N + 1), us = Eigen::VectorXd::Zero(N);
    xs(0) = x_start;
    int tries = 0;
    for (Eigen::Index i = 0; i < N;) {
        const std::vector<Row> r = stage_rows(i, K(i + 1, 0), K(i + 1, 1));
        const double x = xs(i);
        double ulo = -INF, uhi = INF;
        bool feasible = true;
        for (const Row& a : r) {
            const double rest = a.gamma - a.beta * x;
            if (a.alpha > ZERO_COEF) uhi = std::min(uhi, rest / a.alpha);
            else if (a.alpha < -ZERO_COEF) ulo = std::max(ulo, rest / a.alpha);
            else if (rest < -1e-9 * (1.0 + std::abs(a.gamma))) feasible = false;
        }
        if (ulo > uhi + 1e-12 * (1.0 + std::abs(uhi)) || !std::isfinite(uhi)) feasible = false;
        if (!feasible) {
            if (tries < MAX_TRIES) {
                xs(i) = std::max(xs(i) - TINY, 0.999 * xs(i));  // retry with x_i slightly reduced (as toppra)
                ++tries;
                continue;
            }
            return nullptr;
        }
        tries = 0;
        us(i) = uhi;
        double x_next = x + 2.0 * (grid(i + 1) - grid(i)) * uhi;
        x_next = std::max(x_next - TINY, 0.9999 * x_next);
        xs(i + 1) = std::min(K(i + 1, 1), std::max(K(i + 1, 0), x_next));
        ++i;
    }
    data_.sd_vec = xs.cwiseMax(0.0).cwiseSqrt();
    data_.sdd_vec = us;
    data_.ok = true;
    return std::make_shared<ParametrizeConstAccel>(path_, grid, data_.sd_vec);
}

}  // namespace toppralite
