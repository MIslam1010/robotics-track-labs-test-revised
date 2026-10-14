// SBEG 461 Lab 3 -- Forward-dynamics simulator and integrator comparison (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Modern Robotics (MR) Sections 8.3-8.5:
//     tau = M(theta) ddtheta + h(theta, dtheta) + J^T(theta) Ftip,   h = c + g     (MR Eq. 8.77)
//     M(theta) = sum_i J_ib^T(theta) G_i J_ib(theta)                               (MR Eq. 8.57)
//     column j of M = InverseDynamics(theta, 0, e_j, g = 0, Ftip = 0)              (MR Sec. 8.5, p. 296)
//     forward dynamics: solve M ddtheta = tau - h - J^T Ftip                         (MR Eq. 8.81)
//     explicit Euler: theta += dtheta dt, dtheta += ddtheta dt                       (MR Sec. 8.5, p. 297)
// Twists are (omega, v), wrenches (m, f). The state used by the integrators is x = (theta, dtheta) in R^{2n}.
//
// Robots (data/robots.json): "mr_2r_point" (MR Fig. 8.1), "leg_2r" (thigh + transfemoral prosthesis, illustrative),
// "ur5" (MR Exercise 8.15; stand-in for the 6-DOF arm of an EOD robot).
//
// Python -> C++: Mlist -> mr::TransformList (std::vector<Eigen::Matrix4d>, n+1 frames), Glist -> mr::InertiaList
// (std::vector of 6x6), Slist -> Eigen::MatrixXd (6 x n, one screw axis per COLUMN), joint vectors ->
// Eigen::VectorXd, g -> Eigen::Vector3d, Ftip -> mr::Vector6d; the tuple returned by load_robot -> lab::Robot;
// the tuple (K, P) -> std::pair<double, double>; the tuple (t, theta, dtheta) of simulate -> lab::SimResult;
// the callables rnea, f, tau_fn, Ftip_fn -> the std::function types below (a lambda works); `None` -> an empty
// std::function ({}); an unknown method name (Python KeyError) -> std::invalid_argument.
#ifndef SBEG461_LAB3_HPP
#define SBEG461_LAB3_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList, mr::inverse_dynamics (default RNEA)

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// A robot of data/robots.json (Python: the tuple (Mlist, Glist, Slist, g) returned by load_robot).
struct Robot {
    mr::TransformList Mlist;  ///< n+1 frames M_{i-1,i} (CoM frames, MR Sec. 8.3.2)
    mr::InertiaList Glist;    ///< n spatial inertias G_i in the CoM frames (angular block first, MR Eq. 8.32)
    Eigen::MatrixXd Slist;    ///< 6 x n space-frame screw axes (columns)
    Eigen::Vector3d g;        ///< gravity vector in {s} [m/s^2]
};

/// An inverse-dynamics routine with the signature of mr::inverse_dynamics
/// (thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist) -> tau.
using RneaFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&,
                                                   const Eigen::VectorXd&, const Eigen::Vector3d&,
                                                   const mr::Vector6d&, const mr::TransformList&,
                                                   const mr::InertiaList&, const Eigen::MatrixXd&)>;

/// Right-hand side f(t, x) of the first-order ODE  dx/dt = f(t, x).
using StateFunction = std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>;

/// One integration step  x_{k+1} = step(f, t_k, x_k, h)  (euler_step, semi_implicit_euler_step, rk4_step).
using StepFunction =
    std::function<Eigen::VectorXd(const StateFunction&, double, const Eigen::VectorXd&, double)>;

/// tau_fn(t, theta, dtheta) -> joint torques (n).
using TorqueFunction = std::function<Eigen::VectorXd(double, const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/// Ftip_fn(t, theta, dtheta) -> tip wrench in {n+1} (6).
using WrenchFunction = std::function<mr::Vector6d(double, const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/// Result of simulate (Python: the tuple (t, theta, dtheta)).
struct SimResult {
    Eigen::VectorXd t;       ///< N+1 time instants k h
    Eigen::MatrixXd theta;   ///< (N+1) x n, row k = theta at t_k
    Eigen::MatrixXd dtheta;  ///< (N+1) x n, row k = dtheta at t_k
};

// ---------------------------------------------------------------------------
// Provided (complete in starter.cpp)
// ---------------------------------------------------------------------------

/// Return the robot `name` of data/robots.json. An empty path means the lab's data/robots.json.
/// Throws std::invalid_argument if there is no robot with that name.
Robot load_robot(const std::string& name, const std::string& path = "");

/// The stepper for a method name (Python: STEPPERS[method]): "euler" -> euler_step,
/// "semi_implicit" -> semi_implicit_euler_step, "rk4" -> rk4_step; any other name -> std::invalid_argument.
StepFunction stepper(const std::string& method);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- the mass matrix two ways
// ---------------------------------------------------------------------------

/// Mass matrix M(theta) by n calls of an inverse-dynamics (RNEA) routine (MR Sec. 8.5, p. 296).
///
/// Call j uses dtheta = 0, ddtheta = e_j, g = 0, Ftip = 0; the returned torque vector is column j of M.
/// `rnea` has the signature of mr::inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist,
/// Slist), so you can plug in your own Lab 2 RNEA.
Eigen::MatrixXd mass_matrix_rnea(const Eigen::VectorXd& theta, const mr::TransformList& Mlist,
                                 const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist,
                                 const RneaFunction& rnea = mr::inverse_dynamics);

/// List of the n link body Jacobians J_ib(theta) (6 x n, zero-padded beyond column i), i.e. V_i = J_ib dtheta
/// for the CoM frame {i} of link i, expressed in {i} (MR Sec. 8.4, text above Eq. 8.56).
std::vector<Eigen::MatrixXd> link_body_jacobians(const Eigen::VectorXd& theta, const mr::TransformList& Mlist,
                                                 const Eigen::MatrixXd& Slist);

/// M(theta) = sum_i J_ib^T G_i J_ib  (MR Eq. 8.57).
Eigen::MatrixXd mass_matrix_closed_form(const Eigen::VectorXd& theta, const mr::TransformList& Mlist,
                                        const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- forward dynamics and energy
// ---------------------------------------------------------------------------

/// ddtheta from MR Eq. 8.81 using n + 1 inverse-dynamics calls:
/// b = tau - InverseDynamics(theta, dtheta, 0, g, Ftip)   (= tau - h - J^T Ftip),  M from mass_matrix_rnea.
Eigen::VectorXd forward_dynamics(const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                                 const Eigen::VectorXd& tau, const Eigen::Vector3d& g, const mr::Vector6d& Ftip,
                                 const mr::TransformList& Mlist, const mr::InertiaList& Glist,
                                 const Eigen::MatrixXd& Slist);

/// n x 3 matrix: row i = position of the CoM frame {i} of link i in {s}: T_si = e^[S1]th1 ... e^[Si]thi M_0i.
Eigen::MatrixXd com_positions(const Eigen::VectorXd& theta, const mr::TransformList& Mlist,
                              const Eigen::MatrixXd& Slist);

/// (K, P): kinetic energy K = 1/2 dtheta^T M dtheta (MR Eq. 8.56) and potential energy P = -sum_i m_i g . p_ci
/// (zero at the {s} origin), with m_i = Glist[i](3, 3).
std::pair<double, double> energy(const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                                 const Eigen::Vector3d& g, const mr::TransformList& Mlist,
                                 const mr::InertiaList& Glist, const Eigen::MatrixXd& Slist);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- integrators and the simulator
// ---------------------------------------------------------------------------

/// Explicit (forward) Euler: x_{k+1} = x_k + h f(t_k, x_k)   (MR Sec. 8.5 algorithm).
Eigen::VectorXd euler_step(const StateFunction& f, double t, const Eigen::VectorXd& x, double h);

/// Semi-implicit (symplectic) Euler for x = (theta, dtheta):
/// dtheta_{k+1} = dtheta_k + h ddtheta(t_k, x_k);  theta_{k+1} = theta_k + h dtheta_{k+1}.
Eigen::VectorXd semi_implicit_euler_step(const StateFunction& f, double t, const Eigen::VectorXd& x, double h);

/// Classical fourth-order Runge-Kutta step.
Eigen::VectorXd rk4_step(const StateFunction& f, double t, const Eigen::VectorXd& x, double h);

/// Simulate the robot from (theta0, dtheta0) over [0, tf] with N = round(tf / h) fixed steps.
///
/// robot   : as returned by load_robot
/// method  : "euler", "semi_implicit" or "rk4"
/// tau_fn  : tau_fn(t, theta, dtheta) -> joint torques (empty = zero)
/// Ftip_fn : Ftip_fn(t, theta, dtheta) -> tip wrench in {n+1} (empty = zero)
/// Returns {t (N+1), theta ((N+1) x n), dtheta ((N+1) x n)}.
SimResult simulate(const Robot& robot, const Eigen::VectorXd& theta0, const Eigen::VectorXd& dtheta0, double tf,
                   double h, const std::string& method = "rk4", const TorqueFunction& tau_fn = {},
                   const WrenchFunction& Ftip_fn = {});

// ---------------------------------------------------------------------------
// Checkpoint 4 -- verification
// ---------------------------------------------------------------------------

/// Least-squares slope p of log(error) versus log(h): error ~ C h^p.
double observed_order(const Eigen::VectorXd& hs, const Eigen::VectorXd& errors);

}  // namespace lab

#endif  // SBEG461_LAB3_HPP
