// SBEG 461 Lab 3 -- Forward-dynamics simulator and integrator comparison (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 3 --cpp
//
// Notation follows Modern Robotics (MR) Sections 8.3-8.5 (see lab.hpp). As in the Python lab you may use mrcpp
// (the C++ port of modern_robotics): mr::inverse_dynamics, mr::adjoint, mr::trans_inv, mr::jacobian_body,
// mr::matrix_exp6, mr::vec_to_se3 -- but not mr::mass_matrix / mr::forward_dynamics, which the tests compare with.
// You may also plug in your own Lab 2 RNEA anywhere mr::inverse_dynamics is used (mass_matrix_rnea has an
// `rnea` argument for that).
//
// Eigen reminders:  VectorXd::Zero(n), MatrixXd::Zero(r, c), M.col(j) = v, M.leftCols(k), v.head(k), v.tail(k),
//   M.block<3, 1>(0, 3) (= M[0:3, 3]), A * B (matrix product), A.transpose(), v.dot(w),
//   M.partialPivLu().solve(b) (= np.linalg.solve(M, b); never invert M), stacking two vectors:
//   VectorXd x(2 * n);  x << a, b;   (= np.r_[a, b]).
// Never store an Eigen expression in `auto` (write the type: `MatrixXd C = A * B;`).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>

namespace lab {

using Eigen::Index;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---------------------------------------------------------------------------
// Provided (complete -- do not change)
// ---------------------------------------------------------------------------

Robot load_robot(const std::string& name, const std::string& path) {
    const sbeg::json all = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/robots.json") : path);
    if (!all.contains(name) || name.empty() || name[0] == '_')
        throw std::invalid_argument("load_robot: no robot named '" + name + "'");
    const sbeg::json& d = all.at(name);
    Robot r;
    r.Mlist = sbeg::as_list<Eigen::Matrix4d>(d.at("Mlist"));
    r.Glist = sbeg::as_list<mr::Matrix6d>(d.at("Glist"));
    r.Slist = sbeg::as_mat(d.at("Slist"));
    r.g = sbeg::as<Eigen::Vector3d>(d.at("g"));
    return r;
}

StepFunction stepper(const std::string& method) {
    if (method == "euler") return euler_step;
    if (method == "semi_implicit") return semi_implicit_euler_step;
    if (method == "rk4") return rk4_step;
    throw std::invalid_argument("unknown integration method '" + method + "' (use euler, semi_implicit or rk4)");
}

// ---------------------------------------------------------------------------
// Checkpoint 1 -- the mass matrix two ways
// ---------------------------------------------------------------------------

MatrixXd mass_matrix_rnea(const VectorXd& theta, const mr::TransformList& Mlist, const mr::InertiaList& Glist,
                          const MatrixXd& Slist, const RneaFunction& rnea) {
    // TODO: implement mass_matrix_rnea (see lab_handout.md)
    // Hint: for j in 0..n-1: ddtheta = e_j, dtheta = 0, g = Vector3d::Zero(), Ftip = mr::Vector6d::Zero();
    //       M.col(j) = rnea(theta, dtheta, ddtheta, g, Ftip, Mlist, Glist, Slist);
    throw std::logic_error("TODO: mass_matrix_rnea");
}

std::vector<MatrixXd> link_body_jacobians(const VectorXd& theta, const mr::TransformList& Mlist,
                                          const MatrixXd& Slist) {
    // TODO: implement link_body_jacobians (see lab_handout.md)
    // Hint: M_0i = Mlist[0] * ... * Mlist[i]; screw axes of joints 1..i in {i}: B = Ad(M_0i^-1) S
    //       (mr::adjoint, mr::trans_inv, Slist.leftCols(i + 1));
    //       J_ib = mr::jacobian_body(B, theta.head(i + 1)) padded with zero columns to 6 x n.
    throw std::logic_error("TODO: link_body_jacobians");
}

MatrixXd mass_matrix_closed_form(const VectorXd& theta, const mr::TransformList& Mlist,
                                 const mr::InertiaList& Glist, const MatrixXd& Slist) {
    // TODO: implement mass_matrix_closed_form (see lab_handout.md)
    throw std::logic_error("TODO: mass_matrix_closed_form");
}

// ---------------------------------------------------------------------------
// Checkpoint 2 -- forward dynamics and energy
// ---------------------------------------------------------------------------

VectorXd forward_dynamics(const VectorXd& theta, const VectorXd& dtheta, const VectorXd& tau,
                          const Eigen::Vector3d& g, const mr::Vector6d& Ftip, const mr::TransformList& Mlist,
                          const mr::InertiaList& Glist, const MatrixXd& Slist) {
    // TODO: implement forward_dynamics (see lab_handout.md)
    // Hint: one extra call mr::inverse_dynamics(theta, dtheta, 0, g, Ftip, ...) returns h + J^T Ftip;
    //       then M.partialPivLu().solve(...).
    throw std::logic_error("TODO: forward_dynamics");
}

MatrixXd com_positions(const VectorXd& theta, const mr::TransformList& Mlist, const MatrixXd& Slist) {
    // TODO: implement com_positions (see lab_handout.md)
    throw std::logic_error("TODO: com_positions");
}

std::pair<double, double> energy(const VectorXd& theta, const VectorXd& dtheta, const Eigen::Vector3d& g,
                                 const mr::TransformList& Mlist, const mr::InertiaList& Glist,
                                 const MatrixXd& Slist) {
    // TODO: implement energy (see lab_handout.md)
    throw std::logic_error("TODO: energy");
}

// ---------------------------------------------------------------------------
// Checkpoint 3 -- integrators and the simulator
// ---------------------------------------------------------------------------

VectorXd euler_step(const StateFunction& f, double t, const VectorXd& x, double h) {
    // TODO: implement euler_step (see lab_handout.md)
    throw std::logic_error("TODO: euler_step");
}

VectorXd semi_implicit_euler_step(const StateFunction& f, double t, const VectorXd& x, double h) {
    // TODO: implement semi_implicit_euler_step (see lab_handout.md)
    // Hint: update the velocity first, then use the NEW velocity to update theta.
    throw std::logic_error("TODO: semi_implicit_euler_step");
}

VectorXd rk4_step(const StateFunction& f, double t, const VectorXd& x, double h) {
    // TODO: implement rk4_step (see lab_handout.md)
    throw std::logic_error("TODO: rk4_step");
}

SimResult simulate(const Robot& robot, const VectorXd& theta0, const VectorXd& dtheta0, double tf, double h,
                   const std::string& method, const TorqueFunction& tau_fn, const WrenchFunction& Ftip_fn) {
    // TODO: implement simulate (see lab_handout.md)
    // Hint: build  StateFunction f = [&](double t, const VectorXd& x) { ... return (dtheta, forward_dynamics(...)); }
    //       (an empty tau_fn / Ftip_fn means zero: test with `if (tau_fn)`), then call
    //       stepper(method)(f, k * h, X[k], h) N = std::llround(tf / h) times.
    throw std::logic_error("TODO: simulate");
}

// ---------------------------------------------------------------------------
// Checkpoint 4 -- verification
// ---------------------------------------------------------------------------

double observed_order(const VectorXd& hs, const VectorXd& errors) {
    // TODO: implement observed_order (see lab_handout.md)
    // Hint: least-squares line through (log h, log error) (np.polyfit(..., 1)): slope = sum(dx dy) / sum(dx^2).
    throw std::logic_error("TODO: observed_order");
}

}  // namespace lab
