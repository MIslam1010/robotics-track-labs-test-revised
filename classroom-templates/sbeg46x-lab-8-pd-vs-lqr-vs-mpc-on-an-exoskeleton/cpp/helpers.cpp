// SBEG 46X Lab 8 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_params, setpoint, gravity            data/exo_joint.json, the set-point profile, the gravity torque
//   solve_qp                                  the course QP solver (null space + least-distance programming + NNLS)
//   error_limits, make_controller, simulate   the error-state limits, the controller data, the closed-loop simulator
//   limit_rows                                Python _limit_rows: the bound rows of the MPC QPs
//   expm, solve_continuous_are                C++ only: scipy.linalg.expm and scipy.linalg.solve_continuous_are
//   write_sim_csv                             C++ only: save a run as CSV (plot it with cpp/plot_results.py)
//
// These are the "Provided" functions of starter.py, translated to C++ line by line (the same RK4 sub-steps and the
// same QP method, with a C++ port of scipy.optimize.nnls), so a correct starter.cpp reproduces the reference runs in
// data/ref_elbow_*.csv.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace lab {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

JointParams joint_from_json(const sbeg::json& c) {
    JointParams j;
    j.model = c.value("model", std::string());
    j.M = sbeg::as_double(c["M"]);
    j.b = sbeg::as_double(c["b"]);
    j.g_s = sbeg::as_double(c["g_s"]);
    j.g_c = sbeg::as_double(c["g_c"]);
    j.tau_rated = sbeg::as_double(c["tau_rated"]);
    j.mech_limits = sbeg::as<Eigen::Vector2d>(c["mech_limits"]);
    j.tau_max = sbeg::as_double(c["tau_max"]);
    j.rom = sbeg::as<Eigen::Vector2d>(c["rom"]);
    j.v_max = sbeg::as_double(c["v_max"]);
    j.h = sbeg::as_double(c["h"]);
    j.pd.wn = sbeg::as_double(c["pd"]["wn"]);
    j.pd.zeta = sbeg::as_double(c["pd"]["zeta"]);
    j.lqr.x_max = sbeg::as_vec(c["lqr"]["x_max"]);
    j.lqr.u_max = sbeg::as_vec(c["lqr"]["u_max"]);
    j.mpc.N = c["mpc"]["N"].get<int>();
    j.mpc.g_reserve = sbeg::as_double(c["mpc"]["g_reserve"]);
    j.task.theta0 = sbeg::as_double(c["task"]["theta0"]);
    j.task.T = sbeg::as_double(c["task"]["T"]);
    for (const auto& s : c["task"]["steps"]) j.task.steps.emplace_back(sbeg::as_double(s[0]), sbeg::as_double(s[1]));
    return j;
}

/// Non-negative least squares  min |E x - e|  s.t.  x >= 0  (Lawson & Hanson, Solving Least Squares Problems,
/// Ch. 23, algorithm NNLS; the method of scipy.optimize.nnls). Throws std::runtime_error after max_iter inner steps.
VectorXd nnls(const MatrixXd& E, const VectorXd& e, int max_iter) {
    const Index m = E.rows(), n = E.cols();
    const double tol = 10.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(m, n)) *
                       std::max(1.0, E.cwiseAbs().colwise().sum().maxCoeff());
    VectorXd x = VectorXd::Zero(n);
    std::vector<bool> passive(static_cast<std::size_t>(n), false);
    VectorXd wgrad = E.transpose() * (e - E * x);
    int iter = 0;
    // least squares on the passive columns; the other entries of s are 0
    auto solve_passive = [&](VectorXd& s) {
        std::vector<Index> idx;
        for (Index i = 0; i < n; ++i)
            if (passive[static_cast<std::size_t>(i)]) idx.push_back(i);
        MatrixXd Ep(m, static_cast<Index>(idx.size()));
        for (std::size_t k = 0; k < idx.size(); ++k) Ep.col(static_cast<Index>(k)) = E.col(idx[k]);
        const VectorXd sp = Ep.colPivHouseholderQr().solve(e);
        s = VectorXd::Zero(n);
        for (std::size_t k = 0; k < idx.size(); ++k) s(idx[k]) = sp(static_cast<Index>(k));
    };
    for (;;) {
        Index t = -1;
        double best = tol;
        for (Index i = 0; i < n; ++i) {
            if (!passive[static_cast<std::size_t>(i)] && wgrad(i) > best) {
                best = wgrad(i);
                t = i;
            }
        }
        if (t < 0) break;  // Kuhn-Tucker conditions hold
        passive[static_cast<std::size_t>(t)] = true;
        VectorXd s;
        solve_passive(s);
        for (;;) {
            bool all_pos = true;
            for (Index i = 0; i < n; ++i)
                if (passive[static_cast<std::size_t>(i)] && s(i) <= 0.0) all_pos = false;
            if (all_pos) break;
            if (++iter > max_iter) throw std::runtime_error("nnls: iteration limit reached");
            double alpha = std::numeric_limits<double>::infinity();
            for (Index i = 0; i < n; ++i)
                if (passive[static_cast<std::size_t>(i)] && s(i) <= 0.0) alpha = std::min(alpha, x(i) / (x(i) - s(i)));
            x += alpha * (s - x);
            for (Index i = 0; i < n; ++i) {
                if (passive[static_cast<std::size_t>(i)] && x(i) <= tol) {
                    passive[static_cast<std::size_t>(i)] = false;
                    x(i) = 0.0;
                }
            }
            solve_passive(s);
        }
        x = s;
        wgrad = E.transpose() * (e - E * x);
        if (++iter > max_iter) throw std::runtime_error("nnls: iteration limit reached");
    }
    return x;
}

}  // namespace

// ---------------------------------------------------------------------------
// Parameters, set-points, gravity
// ---------------------------------------------------------------------------

Params load_params(const std::string& path) {
    const sbeg::json c = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/exo_joint.json") : path);
    Params p;
    p.note = c.value("note", std::string());
    p.elbow = joint_from_json(c["elbow"]);
    p.knee = joint_from_json(c["knee"]);
    return p;
}

double setpoint(double t, const Task& task) {
    if (task.steps.empty()) throw std::invalid_argument("setpoint: task.steps is empty");
    double th = task.steps[0].second;
    for (const auto& s : task.steps)
        if (t >= s.first - 1e-12) th = s.second;
    return th;
}

double gravity(double theta, const JointParams& j, double scale) {
    return scale * (j.g_s * std::sin(theta) + j.g_c * std::cos(theta));
}

// ---------------------------------------------------------------------------
// QP solver
// ---------------------------------------------------------------------------

std::pair<VectorXd, std::string> solve_qp(const MatrixXd& H_in, const VectorXd& f, const MatrixXd& G_in,
                                          const VectorXd& w, const MatrixXd& Aeq, const VectorXd& beq, double tol) {
    const Index n = f.size();
    if (H_in.rows() != n || H_in.cols() != n) throw std::invalid_argument("solve_qp: H must be n x n");
    const MatrixXd G = G_in.size() == 0 ? MatrixXd(0, n) : G_in;
    if (G.cols() != n || G.rows() != w.size()) throw std::invalid_argument("solve_qp: G must be k x n, w length k");
    VectorXd zp = VectorXd::Zero(n);
    MatrixXd N = MatrixXd::Identity(n, n);
    if (Aeq.size() > 0) {
        if (Aeq.cols() != n || Aeq.rows() != beq.size())
            throw std::invalid_argument("solve_qp: Aeq must be p x n, beq length p");
        Eigen::JacobiSVD<MatrixXd> svd(Aeq, Eigen::ComputeFullU | Eigen::ComputeFullV);
        zp = svd.solve(beq);  // minimum-norm least squares (np.linalg.lstsq)
        if ((Aeq * zp - beq).norm() > 1e-8 * (1.0 + beq.norm())) return {zp, "infeasible"};
        const VectorXd s = svd.singularValues();
        Index rank = 0;
        for (Index i = 0; i < s.size(); ++i)
            if (s(i) > 1e-10 * std::max(1.0, s(0))) ++rank;
        N = svd.matrixV().rightCols(n - rank);
    }
    MatrixXd Hr = N.transpose() * H_in * N;
    const VectorXd fr = N.transpose() * (H_in * zp + f);
    const MatrixXd Gr = G * N;
    const VectorXd wr = w - G * zp;
    const MatrixXd Hs = 0.5 * (Hr + Hr.transpose());
    Hr = Hs;
    const VectorXd v_free = -Hr.partialPivLu().solve(fr);
    if (Gr.rows() == 0 || ((Gr * v_free).array() <= (wr.array() + tol)).all()) return {zp + N * v_free, "optimal"};
    const Eigen::LLT<MatrixXd> llt(Hr);
    if (llt.info() != Eigen::Success) throw std::invalid_argument("solve_qp: H is not positive definite");
    const MatrixXd R = llt.matrixU();  // Hr = R^T R
    const MatrixXd Rinv = R.triangularView<Eigen::Upper>().solve(MatrixXd::Identity(R.rows(), R.cols()));
    const MatrixXd Mc = Gr * Rinv;
    const VectorXd d = wr - Gr * v_free;  // constraints on y = R (v - v_free)
    const Index nv = Hr.rows();
    MatrixXd E(nv + 1, Mc.rows());  // LDP:  (-Mc) y >= -d
    E.topRows(nv) = -Mc.transpose();
    E.row(nv) = -d.transpose();
    VectorXd e = VectorXd::Zero(nv + 1);
    e(nv) = 1.0;
    const VectorXd lam = nnls(E, e, 50 * static_cast<int>(E.cols()));
    const VectorXd res = E * lam - e;
    if (res.norm() < 1e-10) return {zp + N * v_free, "infeasible"};
    const VectorXd y = -res.head(nv) / res(nv);
    const VectorXd v = v_free + Rinv * y;
    const VectorXd z = zp + N * v;
    if (((G * z).array() > (w.array() + 1e-6 * (1.0 + w.array().abs()))).any()) return {z, "infeasible"};
    return {z, "optimal"};
}

// ---------------------------------------------------------------------------
// Controllers and simulator
// ---------------------------------------------------------------------------

Bounds error_limits(double theta_d, const JointParams& j) {
    VectorXd lo(2), hi(2);
    lo << j.rom(0) - theta_d, -j.v_max;
    hi << j.rom(1) - theta_d, j.v_max;
    return {lo, hi};
}

Controller make_controller(const std::string& kind, const JointParams& j) {
    Controller c;
    c.kind = kind;
    std::tie(c.A, c.B) = joint_state_space(j.M, j.b);
    std::tie(c.Ad, c.Bd) = discretize_zoh(c.A, c.B, j.h);
    if (kind == "pd") {
        std::tie(c.Kp, c.Kd) = pd_gains(j.M, j.b, j.pd.zeta, j.pd.wn);
    } else {
        std::tie(c.Q, c.R) = bryson_weights(j.lqr.x_max, j.lqr.u_max);
        std::tie(c.K, c.S) = dlqr(c.Ad, c.Bd, c.Q, c.R);
        if (kind == "mpc") {
            c.N = j.mpc.N;
            const double u_m = j.tau_max - j.mpc.g_reserve;
            c.u_lim = Bounds{VectorXd::Constant(1, -u_m), VectorXd::Constant(1, u_m)};
        }
    }
    return c;
}

SimResult simulate(const std::string& kind, const JointParams& j, std::optional<double> T_opt, double g_scale,
                   const std::optional<Task>& task_opt, int substeps, const std::optional<Controller>& ctrl) {
    const Task task = task_opt ? *task_opt : j.task;
    const double T = T_opt ? *T_opt : task.T;
    const double h = j.h;
    const Index n_steps = static_cast<Index>(std::nearbyint(T / h));
    if (kind != "pd" && kind != "lqr" && kind != "mpc") throw std::invalid_argument("simulate: unknown kind " + kind);
    if (n_steps < 0 || substeps < 1) throw std::invalid_argument("simulate: T >= 0 and substeps >= 1 required");
    SimResult out;
    out.ctrl = ctrl ? *ctrl : make_controller(kind, j);
    const Controller& c = out.ctrl;
    double th = task.theta0, om = 0.0;
    for (VectorXd* v : {&out.t, &out.theta, &out.theta_dot, &out.theta_d, &out.u, &out.tau_cmd, &out.tau})
        *v = VectorXd::Zero(n_steps);
    for (Index n = 0; n < n_steps; ++n) {
        const double t = static_cast<double>(n) * h;
        const double th_d = setpoint(t, task);
        Eigen::VectorXd x(2);
        x << th - th_d, om;
        double u = 0.0;
        std::string st = "-";
        if (kind == "pd") {
            u = c.Kp * (th_d - th) - c.Kd * om;
        } else if (kind == "lqr") {
            const VectorXd Kx = c.K * x;
            u = -Kx(0);
        } else {
            const Bounds lim = error_limits(th_d, j);
            const std::pair<VectorXd, std::string> r = mpc_control(x, c.Ad, c.Bd, c.Q, c.R, c.S, c.N, c.u_lim, lim);
            if (r.first.size() < 1) throw std::invalid_argument("simulate: mpc_control returned an empty input");
            u = r.first(0);
            st = r.second;
            if (st != "optimal") {  // state constraints cannot be met: drop them for this sample
                const std::pair<VectorXd, std::string> r2 =
                    mpc_control(x, c.Ad, c.Bd, c.Q, c.R, c.S, c.N, c.u_lim, std::nullopt);
                if (r2.first.size() < 1) throw std::invalid_argument("simulate: mpc_control returned an empty input");
                u = r2.first(0);
                st = "state-constraints-dropped";
            }
        }
        const double tau_cmd = gravity(th, j) + u;
        const double tau = std::min(std::max(tau_cmd, -j.tau_max), j.tau_max);
        out.t(n) = t;
        out.theta(n) = th;
        out.theta_dot(n) = om;
        out.theta_d(n) = th_d;
        out.u(n) = u;
        out.tau_cmd(n) = tau_cmd;
        out.tau(n) = tau;
        out.status.push_back(st);

        auto f = [&](const Eigen::Vector2d& s) {
            return Eigen::Vector2d(s(1), (tau - j.b * s(1) - gravity(s(0), j, g_scale)) / j.M);
        };
        Eigen::Vector2d s(th, om);
        const double dt = h / substeps;
        for (int k = 0; k < substeps; ++k) {
            const Eigen::Vector2d k1 = f(s);
            const Eigen::Vector2d k2 = f(s + 0.5 * dt * k1);
            const Eigen::Vector2d k3 = f(s + 0.5 * dt * k2);
            const Eigen::Vector2d k4 = f(s + dt * k3);
            s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        th = s(0);
        om = s(1);
    }
    return out;
}

std::pair<MatrixXd, VectorXd> limit_rows(int N, int n, const VectorXd& lim_lo, const VectorXd& lim_hi) {
    if (lim_lo.size() != n || lim_hi.size() != n) throw std::invalid_argument("limit_rows: bounds must have length n");
    const Index Nn = static_cast<Index>(N) * n;
    std::vector<Index> keep_hi, keep_lo;
    for (Index i = 0; i < Nn; ++i) {
        if (std::isfinite(lim_hi(i % n))) keep_hi.push_back(i);
    }
    for (Index i = 0; i < Nn; ++i) {
        if (std::isfinite(lim_lo(i % n))) keep_lo.push_back(i);
    }
    const Index rows = static_cast<Index>(keep_hi.size() + keep_lo.size());
    MatrixXd G = MatrixXd::Zero(rows, Nn);
    VectorXd w(rows);
    Index r = 0;
    for (Index i : keep_hi) {
        G(r, i) = 1.0;
        w(r++) = lim_hi(i % n);
    }
    for (Index i : keep_lo) {
        G(r, i) = -1.0;
        w(r++) = -lim_lo(i % n);
    }
    return {G, w};
}

// ---------------------------------------------------------------------------
// C++ only: expm, solve_continuous_are, write_sim_csv
// ---------------------------------------------------------------------------

MatrixXd expm(const MatrixXd& A) {
    if (A.rows() != A.cols()) throw std::invalid_argument("expm: matrix must be square");
    const Index n = A.rows();
    // Pade(13) scaling and squaring (Higham 2005), the method of scipy.linalg.expm for larger norms.
    static const double b[] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
                               129060195264000.0,   10559470521600.0,    670442572800.0,     33522128640.0,
                               1323241920.0,        40840800.0,          960960.0,           16380.0,
                               182.0,               1.0};
    const double norm1 = n == 0 ? 0.0 : A.cwiseAbs().colwise().sum().maxCoeff();
    int s = 0;
    if (norm1 > 5.371920351148152) s = static_cast<int>(std::ceil(std::log2(norm1 / 5.371920351148152)));
    const MatrixXd As = A / std::ldexp(1.0, s);
    const MatrixXd I = MatrixXd::Identity(n, n);
    const MatrixXd A2 = As * As, A4 = A2 * A2, A6 = A4 * A2;
    const MatrixXd U =
        As * (A6 * (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I);
    const MatrixXd V = A6 * (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I;
    MatrixXd R = (V - U).partialPivLu().solve(V + U);
    for (int i = 0; i < s; ++i) R = (R * R).eval();
    return R;
}

MatrixXd solve_continuous_are(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R) {
    const Index n = A.rows();
    if (A.cols() != n || B.rows() != n || Q.rows() != n || Q.cols() != n || R.rows() != B.cols() ||
        R.cols() != B.cols())
        throw std::invalid_argument("solve_continuous_are: incompatible sizes");
    // Hamiltonian matrix Z = [[A, -B R^-1 B^T], [-Q, -A^T]]; its matrix sign W = sign(Z) gives the stabilising
    // solution from [W12; W22 + I] S = -[W11 + I; W21] (Roberts 1971; Byers 1987, determinant scaling).
    const MatrixXd Gm = B * R.llt().solve(B.transpose());
    MatrixXd Z(2 * n, 2 * n);
    Z << A, -Gm, -Q, -A.transpose();
    for (int it = 0; it < 100; ++it) {
        const Eigen::PartialPivLU<MatrixXd> lu(Z);
        const MatrixXd Zi = lu.inverse();
        const double c = std::pow(std::abs(lu.determinant()), 1.0 / static_cast<double>(2 * n));
        const MatrixXd Zn = 0.5 * (Z / c + c * Zi);
        const double change = (Zn - Z).cwiseAbs().colwise().sum().maxCoeff();
        const double size = Z.cwiseAbs().colwise().sum().maxCoeff();
        Z = Zn;
        if (change <= 1e-14 * size) break;
    }
    const MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd Lhs(2 * n, n), Rhs(2 * n, n);
    Lhs << Z.topRightCorner(n, n), Z.bottomRightCorner(n, n) + I;
    Rhs << Z.topLeftCorner(n, n) + I, Z.bottomLeftCorner(n, n);
    const MatrixXd S = Lhs.colPivHouseholderQr().solve(-Rhs);
    const MatrixXd Ssym = 0.5 * (S + S.transpose());
    return Ssym;
}

void write_sim_csv(const std::string& path, const SimResult& sim) {
    const Index n = sim.t.size();
    MatrixXd M(n, 7);
    M << sim.t, sim.theta, sim.theta_dot, sim.theta_d, sim.u, sim.tau_cmd, sim.tau;
    sbeg::write_csv(path, {"t", "theta", "theta_dot", "theta_d", "u", "tau_cmd", "tau"}, M);
}

}  // namespace lab
