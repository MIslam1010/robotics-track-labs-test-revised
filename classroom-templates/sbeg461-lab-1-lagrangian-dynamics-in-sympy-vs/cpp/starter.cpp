// SBEG 461 Lab 1 -- Lagrangian dynamics, checked against modern_robotics (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 1 --cpp
//
// C++ has no SymPy: here the Lagrangian machinery works on NUMBERS at the requested state (see lab.hpp).
// Functions of the state are std::function objects; write them as lambdas, e.g.
//     const lab::StateFn K = [&params](const Eigen::VectorXd& q, const Eigen::VectorXd& dq) {
//         return planar_chain_energy(params, q, dq).K; };
// Inside numeric_model capture BY VALUE ([p], [K, P], ...), not by reference: the returned model must keep working
// after numeric_model has returned.
// Derivatives: use the given helpers partial / partial_q / partial_dq / partial_matrix (5-point finite differences).
// mrcpp (the C++ port of modern_robotics) is allowed where the Python lab uses modern_robotics (Checkpoint 4).
//
// Eigen reminders:  Eigen::VectorXd::Zero(n), Eigen::VectorXd::Unit(n, i) (i-th unit vector),
//   Eigen::MatrixXd::Zero(n, n), M(i, j), v(i), v.dot(w), a.cross(b) (Vector3d), A.transpose(), A * B,
//   (A - B).cwiseAbs().maxCoeff()  (= np.max(np.abs(A - B))),  S.leftCols(k), v.head(k), T.block<3, 1>(0, 3).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lab {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---------------------------------------------------------------------------
// Given: robot parameters and finite-difference helpers
// ---------------------------------------------------------------------------

ChainParams load_params(const std::string& name) {
    // Given.
    const sbeg::json arms = sbeg::load_json(sbeg::lab_path("../data/arms.json"));
    if (!arms.contains(name)) throw std::invalid_argument("unknown robot in arms.json: " + name);
    const sbeg::json& a = arms[name];
    ChainParams p;
    p.L = sbeg::as_vec(a["L"]);
    p.m = sbeg::as_vec(a["m"]);
    p.r = sbeg::as_vec(a["r"]);
    p.I = sbeg::as_vec(a["I"]);
    p.g = a["g"].get<double>();
    return p;
}

double partial(const ConfigFn& f, const VectorXd& x, int i) {
    // Given: (f(x - 2h e_i) - 8 f(x - h e_i) + 8 f(x + h e_i) - f(x + 2h e_i)) / (12 h),  h = FD_STEP.
    VectorXd xs = x;
    auto at = [&](double s) {
        xs(i) = x(i) + s * FD_STEP;
        return f(xs);
    };
    const double fm2 = at(-2.0), fm1 = at(-1.0), fp1 = at(1.0), fp2 = at(2.0);
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * FD_STEP);
}

double partial_q(const StateFn& f, const VectorXd& q, const VectorXd& dq, int i) {
    // Given.
    return partial([&](const VectorXd& qs) { return f(qs, dq); }, q, i);
}

double partial_dq(const StateFn& f, const VectorXd& q, const VectorXd& dq, int i) {
    // Given.
    return partial([&](const VectorXd& dqs) { return f(q, dqs); }, dq, i);
}

MatrixXd partial_matrix(const MatrixFn& F, const VectorXd& x, int k) {
    // Given: the same 5-point formula, entry by entry.
    VectorXd xs = x;
    auto at = [&](double s) {
        xs(k) = x(k) + s * FD_STEP;
        return MatrixXd(F(xs));
    };
    const MatrixXd fm2 = at(-2.0), fm1 = at(-1.0), fp1 = at(1.0), fp2 = at(2.0);
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * FD_STEP);
}

// ---------------------------------------------------------------------------
// Checkpoint 1 -- Euler-Lagrange machinery
// ---------------------------------------------------------------------------

double time_derivative(const StateFn& f, const VectorXd& q, const VectorXd& dq, const VectorXd& ddq) {
    // TODO: chain rule, sum over i of partial_q(f, q, dq, i) * dq(i) + partial_dq(f, q, dq, i) * ddq(i).
    throw std::logic_error("TODO: time_derivative");
}

VectorXd euler_lagrange(const StateFn& Lag, const VectorXd& q, const VectorXd& dq, const VectorXd& ddq) {
    // TODO: use time_derivative.  For each i, dL/d(dq_i) is itself a function of the state:
    //     const StateFn dL_ddq_i = [&Lag, i](const VectorXd& a, const VectorXd& b) { return partial_dq(Lag, a, b, i); };
    // f(i) = time_derivative(dL_ddq_i, q, dq, ddq) - partial_q(Lag, q, dq, i).
    throw std::logic_error("TODO: euler_lagrange");
}

// ---------------------------------------------------------------------------
// Checkpoint 2 -- energies, mass matrix, gravity, Christoffel symbols, Coriolis matrix
// ---------------------------------------------------------------------------

Energy planar_chain_energy(const ChainParams& params, const VectorXd& theta, const VectorXd& dtheta) {
    // TODO: keep a running joint position (x0, y0) and absolute link angle phi = th_1 + ... + th_i, and
    // likewise the joint velocity (vx0, vy0) and the link angular velocity w = dth_1 + ... + dth_i.
    // CoM of link i: (x0 + r_i cos phi, y0 + r_i sin phi);  its velocity: (vx0 - w r_i sin phi, vy0 + w r_i cos phi)
    // (the time derivative of the position).  Move to the next joint with L_i instead of r_i.
    // return {K, P};   (an Energy with fields K and P)
    throw std::logic_error("TODO: planar_chain_energy");
}

MatrixXd mass_matrix(const StateFn& K, const VectorXd& theta) {
    // TODO: K = 1/2 dtheta^T M dtheta is EXACTLY quadratic in dtheta, so read M off K with unit vectors
    // (no finite differences needed):  e_i = VectorXd::Unit(n, i),
    //     M_ii = 2 K(theta, e_i),   M_ij = K(theta, e_i + e_j) - K(theta, e_i) - K(theta, e_j)   (i != j).
    throw std::logic_error("TODO: mass_matrix");
}

VectorXd gravity_vector(const ConfigFn& P, const VectorXd& theta) {
    // TODO: g_i = dP/dtheta_i  (use partial)
    throw std::logic_error("TODO: gravity_vector");
}

std::vector<MatrixXd> christoffel(const MatrixFn& M, const VectorXd& theta) {
    // TODO: dM[k] = partial_matrix(M, theta, k) holds dm_ij/dth_k in entry (i, j).
    // G[i](j, k) = 1/2 (dM[k](i, j) + dM[j](i, k) - dM[i](j, k))   (MR Eq. 8.17); G has n matrices of size n x n.
    throw std::logic_error("TODO: christoffel");
}

MatrixXd coriolis_matrix(const MatrixFn& M, const VectorXd& theta, const VectorXd& dtheta) {
    // TODO: C_ij = sum_k Gamma_ijk dtheta_k  (MR Eq. 8.19)  -- row i of C is (G[i] * dtheta)^T
    throw std::logic_error("TODO: coriolis_matrix");
}

// ---------------------------------------------------------------------------
// Checkpoint 3 -- numeric model
// ---------------------------------------------------------------------------

NumericModel numeric_model(const ChainParams& params) {
    // TODO: fill every member of NumericModel (n, K, P, M, C, c, g, Mdot, tau) with lambdas that call the functions
    // above. Capture by value. Hints:
    //   K(q, dq) = planar_chain_energy(p, q, dq).K;   P(q) = planar_chain_energy(p, q, VectorXd::Zero(q.size())).P;
    //   M(q) = mass_matrix(K, q);   c = C dtheta;   g(q) = gravity_vector(P, q);
    //   Mdot = sum_k partial_matrix(M, q, k) * dq(k)    (Python: time_derivative of each m_ij with ddq = 0);
    //   tau(q, dq, ddq) = euler_lagrange(L, q, dq, ddq) with the Lagrangian L(a, b) = K(a, b) - P(a).
    throw std::logic_error("TODO: numeric_model");
}

double skew_residual(const NumericModel& model, const VectorXd& theta, const VectorXd& dtheta) {
    // TODO: N = Mdot - 2C;  return max |N + N^T|
    throw std::logic_error("TODO: skew_residual");
}

double power_residual(const NumericModel& model, const VectorXd& theta, const VectorXd& dtheta,
                      const VectorXd& ddtheta) {
    // TODO: |d/dt(K + P) - dtheta^T tau|  with  d/dt(K + P) = dtheta^T M ddtheta + 1/2 dtheta^T Mdot dtheta + dtheta^T g
    throw std::logic_error("TODO: power_residual");
}

// ---------------------------------------------------------------------------
// Checkpoint 4 -- the same robots in modern_robotics (mrcpp)
// ---------------------------------------------------------------------------

MRChain mr_chain(const ChainParams& params) {
    // TODO: joint i sits at x_i = L_1 + ... + L_(i-1) (x_1 = 0) at theta = 0; CoM i at x_i + r_i.
    //   Mlist: n + 1 transforms mr::rp_to_trans(Eigen::Matrix3d::Identity(), Eigen::Vector3d(dx, 0, 0)) between
    //          consecutive frames {0} (origin), CoM 1, ..., CoM n, tip (x = sum L);
    //   Glist: n matrices diag(I_i, I_i, I_i, m_i, m_i, m_i)  (sbeg::Vector6d d; d << ...; sbeg::Matrix6d(d.asDiagonal()));
    //   Slist: 6 x n, column i = (omega, -omega x q_i) with omega = (0, 0, 1), q_i = (x_i, 0, 0);
    //   gvec = (0, -g, 0)  (the gravity VECTOR).
    throw std::logic_error("TODO: mr_chain");
}

MRTerms mr_terms(const ChainParams& params, const VectorXd& theta, const VectorXd& dtheta) {
    // TODO: mr::mass_matrix, mr::vel_quadratic_forces, mr::gravity_forces with the description from mr_chain.
    throw std::logic_error("TODO: mr_terms");
}

// Kinematics backend (given). You may replace the two bodies by your own SBEG 361 C++ functions (forward kinematics
// from 361 Lab 5, space Jacobian from 361 Lab 7); test_361_library_matches_modern_robotics checks them.
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const MatrixXd& Slist, const VectorXd& theta) {
    return mr::fkin_space(M, Slist, theta);  // fallback: mrcpp
}

MatrixXd jacobian_space(const MatrixXd& Slist, const VectorXd& theta) {
    return mr::jacobian_space(Slist, theta);  // fallback: mrcpp
}

MatrixXd mass_matrix_from_jacobians(const ChainParams& params, const VectorXd& theta) {
    // TODO: Js = jacobian_space(Slist, theta) (Slist as in mr_chain).  For CoM i:
    //   p_ci = fk_space(M_ci, Slist.leftCols(i + 1), theta.head(i + 1)).block<3, 1>(0, 3), with M_ci = Trans(x_i + r_i, 0, 0);
    //   for j <= i: column j of Jw_i = omega_j (first 3 rows of column j of Js),
    //               column j of Jv_i = v_j + omega_j x p_ci (last 3 rows + cross product); columns j > i are zero;
    //   M += m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i.
    // Check: in the plane, column j of Jv_i equals z x (p_ci - p_joint_j).
    throw std::logic_error("TODO: mass_matrix_from_jacobians");
}

// ---------------------------------------------------------------------------
// Checkpoint 5 -- numeric Lagrangian model vs modern_robotics
// ---------------------------------------------------------------------------

std::map<std::string, double> compare(const ChainParams& params, const StateList& states) {
    // TODO: model = numeric_model(params); start with {{"M", 0.0}, {"c", 0.0}, {"g", 0.0}, {"M_jac", 0.0}} and, for
    // every state  for (const auto& [th, dth] : states), keep the maximum of (A - B).cwiseAbs().maxCoeff() for
    // model M vs mr_terms M, model c vs mr_terms c, model g vs mr_terms g, model M vs mass_matrix_from_jacobians.
    throw std::logic_error("TODO: compare");
}

}  // namespace lab
