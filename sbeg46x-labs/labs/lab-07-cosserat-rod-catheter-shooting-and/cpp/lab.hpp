// SBEG 46X Lab 7 -- Cosserat-rod statics of a tendon-driven catheter: shooting and BVP solvers, comparison with
// the constant-curvature (CC) and Euler-Bernoulli (EB) models under tip loads, and tip-force estimation
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided functions (load_catheter, load_csv, tip_pose, contact_direction, integrate_rod, rigid_guess,
// weights, free_tip_angle) and the numerical tools that replace SciPy (fsolve, least_squares, solve_bvp) are in
// helpers.cpp (given code, do not edit).
//
// Model and sign conventions (the same as the Week 7 lectures and workshop)
//   * Planar rod in the x-y plane, clamped at the origin with its base tangent along +x (theta0 = 0), arc length
//     s in [0, L].  State z = [x, y, theta, nx, ny, m]:
//         x' = cos(theta),  y' = sin(theta)                  (inextensible, unshearable: Kirchhoff rod)
//         theta' = m / EI + kappa0                           (linear constitutive law, precurvature kappa0)
//         n' = -f                                            (force balance, f = distributed force per length)
//         m' = -(x' ny - y' nx)                              (moment balance, m' + (p' x n)_z = 0)
//     n(s), m(s) = internal force and moment exerted by the distal part (s+) on the proximal part.
//     Boundary conditions: x(0) = y(0) = 0, theta(0) = theta0;  n(L) = F, m(L) = M  (F, M = tip wrench).
//   * A tendon routed parallel to the backbone at offset r on the +normal (+y at the base) side, tension tau and
//     terminated at the tip, is exactly equivalent to an extra tip moment M_t = r * tau for an inextensible rod (its
//     axial pull compresses the rod but does not bend it).  The tendon side becomes the inside of the bend.
//   * F is the force applied TO the catheter tip (e.g. by tissue), in the base frame.
// Units SI (m, N, N m, N m^2).  ILLUSTRATIVE VALUES (data/catheter.json): L = 0.10 m, EI = 1.0e-3 N m^2, r = 1.2 mm.
//
// Python -> C++: the catheter dict -> lab::Catheter (same field names: cat.L, cat.EI, cat.r, ...); the log dict of
// load_csv -> lab::Log (std::map from column name to Eigen::VectorXd: log.at("tau")); the rod-solution dict ->
// lab::RodSolution (sol.s, sol.x, ..., sol.m, sol.g); the dicts of compare_models / force_errors ->
// lab::ModelComparison / lab::ForceErrors; [x, y, theta] -> Eigen::Vector3d; force and distributed-load pairs ->
// Eigen::Vector2d; the (k, 3) arrays -> Eigen::MatrixXd (row = one load); optional arguments `guess=None`,
// `direction=None`, `F_init=None` -> std::optional (pass std::nullopt or a vector); SciPy solve_ivp, fsolve,
// least_squares, solve_bvp -> the given lab::integrate_rod, lab::fsolve, lab::least_squares, lab::solve_bvp;
// ValueError -> std::invalid_argument, RuntimeError -> std::runtime_error.
#ifndef SBEG46X_LAB7_HPP
#define SBEG46X_LAB7_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace lab {

using Vector6d = sbeg::Vector6d;
using Matrix32d = Eigen::Matrix<double, 3, 2>;

// ===========================================================================================
// Provided (helpers.cpp, do not edit): data files and small helpers
// ===========================================================================================

/// Catheter and sensor parameters from data/catheter.json (illustrative values): L (m), EI (N m^2),
/// r (tendon offset, m), tau_nominal and tau_max (N), sigma_p (EM-sensor position noise, m), sigma_theta (rad).
struct Catheter {
    double L = 0.0, EI = 0.0, r = 0.0, tau_nominal = 0.0, tau_max = 0.0, sigma_p = 0.0, sigma_theta = 0.0;
    std::string description;
};
Catheter load_catheter();

/// A CSV log: one Eigen::VectorXd per column, by column name (Python: dict of float arrays).
using Log = std::map<std::string, Eigen::VectorXd>;

/// Load a CSV file from data/ into a Log (one vector per column).
Log load_csv(const std::string& name);

/// A rod solution of solve_rod_shooting / solve_rod_bvp: the state on the grid s (one entry per grid point) and
/// the base values g = [nx(0), ny(0), m(0)].
struct RodSolution {
    Eigen::VectorXd s, x, y, theta, nx, ny, m;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
};

/// Tip pose [x, y, theta] of a rod solution returned by solve_rod_shooting / solve_rod_bvp.
Eigen::Vector3d tip_pose(const RodSolution& sol);

/// Unit direction of a contact force that pushes the tip back against its bend: -(tip normal) =
/// (sin theta, -cos theta).  A force F_n >= 0 along it 'unbends' the catheter (tissue resisting the tip).
Eigen::Vector2d contact_direction(double theta_tip);

// ===========================================================================================
// Part A -- constant-curvature (CC) and piecewise-constant-curvature (PCC) models
// ===========================================================================================

/// A1. Tip pose [x, y, theta] of a planar constant-curvature arc of curvature kappa and length L that leaves
/// the origin along +x:  x = sin(kappa L)/kappa, y = (1 - cos(kappa L))/kappa, theta = kappa L.
/// Must also work for kappa -> 0 (straight: [L, 0, 0]) without dividing by zero.
Eigen::Vector3d cc_tip(double kappa, double L);

/// A2. Curvature of an unloaded tendon-driven segment: the tendon gives the constant moment r tau, so
/// kappa = r tau / EI (exact for the Kirchhoff rod; this is why the CC model holds without external load).
double tendon_curvature(double tau, double r, double EI);

/// A3. The same arc as an element of SE(3): T(L) = exp([V] L) with the constant body twist per unit arc
/// length V = (omega, v) = (0, 0, kappa, 1, 0, 0) (MR Eq. 3.71 with s in place of t; MR Prop. 3.25, Eq. 3.88).
/// Use mrcpp (mr::vec_to_se3, mr::matrix_exp6).
Eigen::Matrix4d cc_transform(double kappa, double L);

/// A4. Tip pose [x, y, theta] of a planar piecewise-constant-curvature (PCC) robot: segments i = 1..n with
/// curvature kappas[i] and length lengths[i], each starting where (and tangent to where) the previous ended:
///     p_i = p_{i-1} + R(theta_{i-1}) cc_xy(kappa_i, L_i),  theta_i = theta_{i-1} + kappa_i L_i.
Eigen::Vector3d pcc_tip(const std::vector<double>& kappas, const std::vector<double>& lengths);

/// A5. Linear bending model of a pneumatic soft segment: chamber pressure p (Pa, gauge) on area A (m^2) whose
/// centroid is d (m) from the neutral axis gives the moment p A d, so kappa = p A d / EI (EI = effective bending
/// stiffness of the elastomer body; calibrate it, real elastomers are nonlinear).
double pneumatic_curvature(double p, double A, double d, double EI);

// ===========================================================================================
// Part B -- Cosserat-rod statics: equations, shooting, BVP collocation, Euler-Bernoulli check
// ===========================================================================================

/// B1. Right-hand side dz/ds of the planar Cosserat (Kirchhoff) rod, z = [x, y, theta, nx, ny, m].
/// f = (fx, fy) is a constant distributed force per unit length (e.g. weight), kappa0 the precurvature.
/// (C++: one state at a time; the solvers call it point by point.)
Vector6d rod_rhs(double s, const Vector6d& z, double EI, const Eigen::Vector2d& f = Eigen::Vector2d::Zero(),
                 double kappa0 = 0.0);

/// Provided (do not edit; Python: _integrate with solve_ivp): integrate the rod ODE (your rod_rhs) from the base
/// state z0 with the classical 4th-order Runge-Kutta method (about 4000 fixed steps over [0, L]).  Returns a
/// 6 x n_eval matrix: column k = z(s_k) on s = linspace(0, L, n_eval) (n_eval >= 2; the last column is the tip).
Eigen::MatrixXd integrate_rod(const Vector6d& z0, double L, double EI, const Eigen::Vector2d& f, double kappa0,
                             int n_eval = 2);

/// Provided (do not edit; Python: _rigid_guess): base wrench [nx(0), ny(0), m(0)] of the straight (undeformed)
/// rod: a good first guess for shooting.
Eigen::Vector3d rigid_guess(double L, const Eigen::Vector2d& F, double M, const Eigen::Vector2d& f);

/// B2. Shooting residual.  g = guessed base values [nx(0), ny(0), m(0)] (the unknown half of the initial state;
/// x(0) = y(0) = theta(0) = 0 are known).  Integrate from s = 0 to L and return the tip-condition error
/// [nx(L) - Fx, ny(L) - Fy, m(L) - M].
Eigen::Vector3d shooting_residual(const Eigen::Vector3d& g, double L, double EI,
                                  const Eigen::Vector2d& F = Eigen::Vector2d::Zero(), double M = 0.0,
                                  const Eigen::Vector2d& f = Eigen::Vector2d::Zero(), double kappa0 = 0.0);

/// B3. Solve the rod BVP by shooting: find g = [nx(0), ny(0), m(0)] with shooting_residual(g) = 0 (lab::fsolve;
/// start from the straight-rod statics unless `guess` is given), then integrate once more on
/// s = linspace(0, L, n_eval).  Return a RodSolution (s, x, y, theta, nx, ny, m, g).
/// Shortcut (allowed): with f = 0 the internal force is constant, n(0) = F, and only m(0) is unknown
/// (scalar shooting, secant iterations).
RodSolution solve_rod_shooting(double L, double EI, const Eigen::Vector2d& F = Eigen::Vector2d::Zero(),
                               double M = 0.0, const Eigen::Vector2d& f = Eigen::Vector2d::Zero(),
                               double kappa0 = 0.0, int n_eval = 101,
                               const std::optional<Eigen::Vector3d>& guess = std::nullopt);

/// B4. Solve the same BVP by collocation with lab::solve_bvp: six boundary conditions
/// x(0) = y(0) = theta(0) = 0, nx(L) = Fx, ny(L) = Fy, m(L) = M.  Initial guess on s = linspace(0, L, n_mesh):
/// the straight rod with the rigid-body internal wrench.  Throw std::runtime_error if the solver fails.
/// Return a RodSolution evaluated on s = linspace(0, L, n_eval) (g = the base values [nx, ny, m] at s = 0).
RodSolution solve_rod_bvp(double L, double EI, const Eigen::Vector2d& F = Eigen::Vector2d::Zero(), double M = 0.0,
                          const Eigen::Vector2d& f = Eigen::Vector2d::Zero(), double kappa0 = 0.0, int n_mesh = 41,
                          int n_eval = 101);

/// B5. Linear (small-deflection) Euler-Bernoulli cantilever: transverse tip force P (+y), tip moment M and
/// uniform transverse load w per length (+y), superposed.  Return [x, y, theta] with x = L (no foreshortening):
///     y = P L^3/(3EI) + M L^2/(2EI) + w L^4/(8EI),   theta = P L^2/(2EI) + M L/EI + w L^3/(6EI).
Eigen::Vector3d euler_bernoulli_tip(double L, double EI, double P = 0.0, double M = 0.0, double w = 0.0);

// ===========================================================================================
// Part C -- catheter: Cosserat vs CC vs EB under tip loads
// ===========================================================================================

/// C1. Cosserat tip pose [x, y, theta] of the tendon-driven catheter `cat` with tendon tension tau and tip
/// force F (base frame): the rod of B3 with tip moment M = r tau.
Eigen::Vector3d catheter_tip(double tau, const Eigen::Vector2d& F, const Catheter& cat);

/// Result of compare_models (Python: the returned dict, same keys).
struct ModelComparison {
    Eigen::VectorXd Fn;        // (k)    contact-force magnitudes
    Eigen::MatrixXd cosserat;  // (k, 3) Cosserat tip poses, one row per load
    Eigen::Vector3d cc = Eigen::Vector3d::Zero();  // CC tip pose (independent of the load)
    Eigen::MatrixXd eb;        // (k, 3) Euler-Bernoulli tip poses
    Eigen::VectorXd err_cc;    // (k)    tip-position errors (m) w.r.t. Cosserat
    Eigen::VectorXd err_eb;    // (k)
};

/// C2. For contact forces of magnitude Fn along contact_direction(theta_free) of the UNLOADED tip
/// (theta_free = CC tip angle for tension tau), return Fn, cosserat (k,3), cc (3), eb (k,3), err_cc (k), err_eb (k):
/// cc = CC tip pose (independent of the load), eb = euler_bernoulli_tip with P = Fy (the transverse component)
/// and M = r tau, err_* = Euclidean tip-POSITION error (m) of that model w.r.t. Cosserat.
ModelComparison compare_models(double tau, const std::vector<double>& Fn_values, const Catheter& cat);

// ===========================================================================================
// Part D -- tip-force estimation from the measured tip pose (EM sensor)
// ===========================================================================================

/// D1. 3x2 tip compliance C = d[x, y, theta]/d[Fx, Fy] at tension tau and load F0, by central differences
/// with step h (N) on the Cosserat model (catheter_tip).
Matrix32d compliance_matrix(double tau, const Catheter& cat, const Eigen::Vector2d& F0 = Eigen::Vector2d::Zero(),
                            double h = 1e-3);

/// Provided (do not edit; Python: _weights): W = diag(1/sigma_p, 1/sigma_p, 1/sigma_theta).
Eigen::Matrix3d weights(const Catheter& cat);

/// Provided (do not edit): tip angle of the unloaded catheter (CC: kappa L with kappa = r tau / EI).
double free_tip_angle(double tau, const Catheter& cat);

/// D2. Linear (small-force) weighted least-squares estimate from dz = pose_meas - pose_free (pose_free = unloaded
/// Cosserat tip pose), C = compliance_matrix(tau) at F0 = 0 and W = weights(cat):
///   * no direction:  both components,  F = (W C)^+ W dz;
///   * direction = unit vector d (contact model: force along a known direction, e.g. the tissue normal):
///     one unknown magnitude, c = C d, F = d (W c)^+ W dz.
/// Returns F.
Eigen::Vector2d estimate_force_linear(const Eigen::Vector3d& pose_meas, double tau, const Catheter& cat,
                                      const std::optional<Eigen::Vector2d>& direction = std::nullopt);

/// D3. Nonlinear estimate: minimise || W (catheter_tip(tau, F) - pose_meas) ||^2 with lab::least_squares over F
/// (2 unknowns) or, if `direction` d is given, over the magnitude a in F = a d.
/// Start from F_init (default: zero force).  Returns F.
Eigen::Vector2d estimate_force_nl(const Eigen::Vector3d& pose_meas, double tau, const Catheter& cat,
                                  const std::optional<Eigen::Vector2d>& direction = std::nullopt,
                                  const std::optional<Eigen::Vector2d>& F_init = std::nullopt);

/// D4. Estimate the tip force for every row of a log (columns tau, x_meas, y_meas, theta_meas): returns an
/// (N, 2) matrix (row = [Fx, Fy]).  method = "linear" (D2) or "nl" (D3); normal_only = true uses the contact model
/// direction = contact_direction(free_tip_angle(tau)) (force pushing the tip back against its bend).
Eigen::MatrixXd estimate_log(const Log& log, const Catheter& cat, const std::string& method = "nl",
                             bool normal_only = true);

/// Result of force_errors (Python: the returned dict, same keys).  Units N.
struct ForceErrors {
    double rms = 0.0, rms_normal = 0.0, rms_tangential = 0.0;
};

/// D5. Error statistics of force estimates (N, 2) against reference forces (N, 2).  theta_tip (N) is the tip
/// angle used to split each error into the component along the tip normal (-sin, cos) and along the tip
/// tangent (cos, sin).  Return rms, rms_normal, rms_tangential (N).
ForceErrors force_errors(const Eigen::MatrixXd& F_est, const Eigen::MatrixXd& F_ref,
                         const Eigen::VectorXd& theta_tip);

// ===========================================================================================
// Provided numerical tools (helpers.cpp, do not edit): small C++ stand-ins for the SciPy functions of the lab
// ===========================================================================================

/// A vector function x -> r(x) (residuals).
using VecFun = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/// Like scipy.optimize.fsolve: Newton's method with a finite-difference Jacobian and step halving, from x0.
/// Stops when the step is below xtol (relative) or the residual is zero; returns the last iterate (check the
/// residual yourself, as the Python reference does).
Eigen::VectorXd fsolve(const VecFun& fun, const Eigen::VectorXd& x0, double xtol = 1e-12, int max_iter = 100);

/// Like scipy.optimize.least_squares(fun, x0, x_scale=x_scale, xtol=xtol): Levenberg-Marquardt on
/// 0.5 ||fun(x)||^2 with a central-difference Jacobian; x_scale = typical size of the unknowns.  Returns x.
Eigen::VectorXd least_squares(const VecFun& fun, const Eigen::VectorXd& x0, double x_scale = 1.0, double xtol = 1e-10,
                              int max_iter = 200);

/// Result of solve_bvp (Python: the fields of the scipy result with the same names).
struct BvpResult {
    bool success = false;
    std::string message;
    Eigen::VectorXd x;   // final mesh
    Eigen::MatrixXd y;   // solution at the mesh nodes: one column per node (rows = state components)
    Eigen::MatrixXd yp;  // derivative fun(x_i, y_i) at the mesh nodes
    /// The continuous solution (cubic Hermite interpolation, as res.sol in SciPy): one column per entry of s.
    Eigen::MatrixXd sol(const Eigen::VectorXd& s) const;
};

/// Like scipy.integrate.solve_bvp(fun, bc, s, Z0, tol=tol, max_nodes=max_nodes): 4th-order Lobatto IIIA
/// (Simpson) collocation solved by Newton's method.  fun(s, z) returns dz/ds; bc(za, zb) returns the boundary
/// residuals (as many as states).  s = initial mesh (increasing), Z0 = initial guess, one column per mesh node.
/// The mesh is refined by halving every interval until two successive solutions agree to tol (relative);
/// success = false (with a message) if Newton fails or the mesh would exceed max_nodes.
BvpResult solve_bvp(const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& fun,
                    const std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>& bc,
                    const Eigen::VectorXd& s, const Eigen::MatrixXd& Z0, double tol = 1e-8, int max_nodes = 300);

/// np.linspace(a, b, n) (n >= 2; the last value is exactly b).
Eigen::VectorXd linspace(double a, double b, int n);

}  // namespace lab

#endif  // SBEG46X_LAB7_HPP
