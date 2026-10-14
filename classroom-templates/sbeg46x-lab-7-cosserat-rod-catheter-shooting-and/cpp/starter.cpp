// SBEG 46X Lab 7 -- Cosserat-rod statics of a tendon-driven catheter: shooting and BVP solvers, CC and EB models
// under tip loads, tip-force estimation (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line), in the order
// A1 ... D5.  Do not change function names or signatures (they are declared in lab.hpp, with the full documentation
// and the sign conventions).  Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 7 --cpp
//
// Provided (helpers.cpp, do not edit): load_catheter, load_csv, tip_pose, contact_direction, integrate_rod (RK4 of
// your rod_rhs; Python _integrate), rigid_guess (Python _rigid_guess), weights (Python _weights), free_tip_angle,
// linspace, and the SciPy stand-ins fsolve, least_squares and solve_bvp (see lab.hpp).
// Catheter fields: cat.L, cat.EI, cat.r, cat.tau_nominal, cat.sigma_p, cat.sigma_theta.
//
// Eigen reminders: Eigen::Vector3d(x, y, th); v(0), v.head<2>(), v.norm(), a.dot(b); Vector6d z; z << a, b, c, ...;
//   Eigen::MatrixXd Z(6, n); Z.col(k), Z.row(i).transpose(), Z(i, k), Z.cols(); M.row(i) = v.transpose();
//   std::optional: `if (direction) { const Eigen::Vector2d d = *direction; ... }`.
//   A lambda for the solvers:  [&](const Eigen::VectorXd& g) -> Eigen::VectorXd { return ...; }
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>  // mr::vec_to_se3, mr::matrix_exp6 (A3), mr::pinv (= np.linalg.pinv, D2)

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// =====================================================================  Part A -- CC and PCC models

Eigen::Vector3d cc_tip(double kappa, double L) {
    // TODO A1.  For |kappa L| < 1e-6 use the series sin(kL)/k = L(1 - th^2/6), (1 - cos kL)/k = L th/2 (1 - th^2/12).
    throw std::logic_error("TODO: cc_tip");
}

double tendon_curvature(double tau, double r, double EI) {
    // TODO A2
    throw std::logic_error("TODO: tendon_curvature");
}

Eigen::Matrix4d cc_transform(double kappa, double L) {
    // TODO A3.  sbeg::Vector6d V; V << 0, 0, kappa, 1, 0, 0;  then mr::matrix_exp6(mr::vec_to_se3(V * L)).
    throw std::logic_error("TODO: cc_transform");
}

Eigen::Vector3d pcc_tip(const std::vector<double>& kappas, const std::vector<double>& lengths) {
    // TODO A4.  Loop over the segments: rotate each cc_tip(kappa_i, L_i) position by the current angle and add.
    throw std::logic_error("TODO: pcc_tip");
}

double pneumatic_curvature(double p, double A, double d, double EI) {
    // TODO A5
    throw std::logic_error("TODO: pneumatic_curvature");
}

// =====================================================================  Part B -- Cosserat-rod statics

Vector6d rod_rhs(double s, const Vector6d& z, double EI, const Eigen::Vector2d& f, double kappa0) {
    // TODO B1.  z = [x, y, theta, nx, ny, m]; return [x', y', theta', nx', ny', m'] (see lab.hpp).
    throw std::logic_error("TODO: rod_rhs");
}

Eigen::Vector3d shooting_residual(const Eigen::Vector3d& g, double L, double EI, const Eigen::Vector2d& F, double M,
                                  const Eigen::Vector2d& f, double kappa0) {
    // TODO B2.  Base state z0 = [0, 0, 0, g(0), g(1), g(2)]; Z = integrate_rod(z0, L, EI, f, kappa0);
    // the tip state is Z.col(Z.cols() - 1).
    throw std::logic_error("TODO: shooting_residual");
}

RodSolution solve_rod_shooting(double L, double EI, const Eigen::Vector2d& F, double M, const Eigen::Vector2d& f,
                               double kappa0, int n_eval, const std::optional<Eigen::Vector3d>& guess) {
    // TODO B3
    // Hint: g = fsolve([&](const Eigen::VectorXd& gg) -> Eigen::VectorXd { return shooting_residual(
    //           Eigen::Vector3d(gg), L, EI, F, M, f, kappa0); }, g0) from g0 = rigid_guess(L, F, M, f) (or *guess);
    // then integrate once more with integrate_rod(..., n_eval) and fill a RodSolution (s = linspace(0, L, n_eval),
    // x = Z.row(0).transpose(), ..., g).
    throw std::logic_error("TODO: solve_rod_shooting");
}

RodSolution solve_rod_bvp(double L, double EI, const Eigen::Vector2d& F, double M, const Eigen::Vector2d& f,
                          double kappa0, int n_mesh, int n_eval) {
    // TODO B4
    // Hint: res = solve_bvp(fun, bc, s, Z0) with fun(s, z) = rod_rhs(s, Vector6d(z), EI, f, kappa0) (returned as an
    // Eigen::VectorXd) and bc(za, zb) returning the six boundary residuals; Z0 = 6 x n_mesh initial guess (one column
    // per node of s = linspace(0, L, n_mesh)); throw std::runtime_error(res.message) if !res.success; then
    // res.sol(linspace(0, L, n_eval)) is the 6 x n_eval output.
    throw std::logic_error("TODO: solve_rod_bvp");
}

Eigen::Vector3d euler_bernoulli_tip(double L, double EI, double P, double M, double w) {
    // TODO B5
    throw std::logic_error("TODO: euler_bernoulli_tip");
}

// =====================================================================  Part C -- Cosserat vs CC vs EB

Eigen::Vector3d catheter_tip(double tau, const Eigen::Vector2d& F, const Catheter& cat) {
    // TODO C1.  Only the tip is needed: n_eval = 2 makes it fast.
    throw std::logic_error("TODO: catheter_tip");
}

ModelComparison compare_models(double tau, const std::vector<double>& Fn_values, const Catheter& cat) {
    // TODO C2.  Size the outputs first (out.cosserat.resize(k, 3), ...), then fill one row per load.
    throw std::logic_error("TODO: compare_models");
}

// =====================================================================  Part D -- tip-force estimation

Matrix32d compliance_matrix(double tau, const Catheter& cat, const Eigen::Vector2d& F0, double h) {
    // TODO D1
    throw std::logic_error("TODO: compliance_matrix");
}

Eigen::Vector2d estimate_force_linear(const Eigen::Vector3d& pose_meas, double tau, const Catheter& cat,
                                      const std::optional<Eigen::Vector2d>& direction) {
    // TODO D2.  (W C)^+ : mr::pinv(W * C) (a 2 x 3 matrix).
    throw std::logic_error("TODO: estimate_force_linear");
}

Eigen::Vector2d estimate_force_nl(const Eigen::Vector3d& pose_meas, double tau, const Catheter& cat,
                                  const std::optional<Eigen::Vector2d>& direction,
                                  const std::optional<Eigen::Vector2d>& F_init) {
    // TODO D3
    // Hint: least_squares([&](const Eigen::VectorXd& F) -> Eigen::VectorXd {
    //           return W * (catheter_tip(tau, Eigen::Vector2d(F), cat) - z); }, x0, 0.05)
    // (with a direction: one unknown, an Eigen::VectorXd of size 1, F = a(0) * d).
    throw std::logic_error("TODO: estimate_force_nl");
}

Eigen::MatrixXd estimate_log(const Log& log, const Catheter& cat, const std::string& method, bool normal_only) {
    // TODO D4.  Columns: log.at("tau"), log.at("x_meas"), log.at("y_meas"), log.at("theta_meas").
    throw std::logic_error("TODO: estimate_log");
}

ForceErrors force_errors(const Eigen::MatrixXd& F_est, const Eigen::MatrixXd& F_ref,
                         const Eigen::VectorXd& theta_tip) {
    // TODO D5
    throw std::logic_error("TODO: force_errors");
}

}  // namespace lab
