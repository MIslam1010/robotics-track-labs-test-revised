// SBEG 461 Lab 2 -- Newton-Euler inverse dynamics (RNEA), verified against modern_robotics and Pinocchio
// (C++ STARTER).
//
// Fill in every function marked  // TODO  (checkpoints 1-6, in order; replace the `throw std::logic_error("TODO: ...")`
// line). Functions marked "Given." are complete -- read them, do not change them. Do not change function names or
// signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 2 --cpp
//
// You may NOT call mr::inverse_dynamics, mr::mass_matrix, ... or pin_rnea inside checkpoints 1-4 and 6: those are the
// references your code is tested against.  You MAY use the mrcpp helpers mr::adjoint, mr::ad, mr::trans_inv,
// mr::matrix_exp6, mr::vec_to_se3, mr::rp_to_trans, mr::trans_to_rp (or the equivalents from your SBEG 361 library).
// Notation (MR Sections 8.2-8.3, twists and wrenches ANGULAR FIRST): see lab.hpp.
//
// Eigen reminders:  Matrix6d G = Matrix6d::Zero();  G.topLeftCorner<3, 3>() = I;  G.bottomRightCorner<3, 3>() = ...;
//   A.col(i) (column i, Python A[:, i]),  V.head<3>() / V.tail<3>() (Python V[:3] / V[3:]),  A.transpose(),
//   A * B (matrix product),  a.dot(b),  M.cwiseAbs(),  M.colwise().maxCoeff(),  VectorXd::Unit(n, j) (= e_j),
//   M.row(k).transpose() (row k as a column vector).  Never store an Eigen expression in `auto`: write the type.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lab {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

// ---------------------------------------------------------------------------
// Provided helpers
// ---------------------------------------------------------------------------

Robot load_robot(const std::string& name) {
    // Given.
    const sbeg::json all = sbeg::load_json(sbeg::lab_path("../data/robots.json"));
    if (!all.contains(name)) throw std::invalid_argument("unknown robot '" + name + "' in data/robots.json");
    const sbeg::json& r = all[name];
    Robot rb;
    rb.Mlist = sbeg::as_list<Matrix4d>(r["Mlist"]);
    rb.Glist = sbeg::as_list<Matrix6d>(r["Glist"]);
    rb.Slist = sbeg::as_mat(r["Slist"]);
    rb.g = sbeg::as<Vector3d>(r["g"]);
    rb.n = static_cast<int>(rb.Slist.cols());
    rb.description = r["description"].get<std::string>();
    rb.name = name;
    return rb;
}

std::vector<State> load_states(const std::string& name) {
    // Given.  Each line: robot, theta(n), dtheta(n), ddtheta(n), Ftip(6); lines starting with '#' are comments.
    std::ifstream in = sbeg::open_input(sbeg::lab_path("../data/states.csv"));
    std::vector<State> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        if (cell != name) continue;
        std::vector<double> x;
        while (std::getline(ss, cell, ',')) x.push_back(std::stod(cell));
        const Eigen::Index n = static_cast<Eigen::Index>((x.size() - 6) / 3);
        const VectorXd v = Eigen::Map<const VectorXd>(x.data(), static_cast<Eigen::Index>(x.size()));
        State s;
        s.theta = v.segment(0, n);
        s.dtheta = v.segment(n, n);
        s.ddtheta = v.segment(2 * n, n);
        s.Ftip = v.segment<6>(3 * n);
        out.push_back(s);
    }
    return out;
}

Trajectory load_trajectory(const std::string& fname) {
    // Given.  The first line is a '#' comment, the second the header.
    const sbeg::CsvTable t = sbeg::read_csv(sbeg::lab_path("../data/" + fname));
    const Eigen::Index n = (t.data.cols() - 1) / 3;
    Trajectory tr;
    tr.t = t.data.col(0);
    tr.theta = t.data.middleCols(1, n);
    tr.dtheta = t.data.middleCols(1 + n, n);
    tr.ddtheta = t.data.middleCols(1 + 2 * n, n);
    return tr;
}

// ---------------------------------------------------------------------------
// Checkpoint 1 -- one rigid body
// ---------------------------------------------------------------------------

Matrix6d spatial_inertia(const Matrix3d& I, double m) {
    // TODO: Build a 6x6 zero matrix and fill the two diagonal blocks.
    throw std::logic_error("TODO: spatial_inertia");
}

Matrix6d transform_spatial_inertia(const Matrix6d& G_b, const Matrix4d& T_ba) {
    // TODO: One line with mr::adjoint(T_ba).
    throw std::logic_error("TODO: transform_spatial_inertia");
}

Vector6d rigid_body_wrench(const Matrix6d& G, const Vector6d& V, const Vector6d& Vdot) {
    // TODO: mr::ad(V) returns [ad_V]; remember the TRANSPOSE in the gyroscopic term.
    throw std::logic_error("TODO: rigid_body_wrench");
}

// ---------------------------------------------------------------------------
// Checkpoint 2 -- forward iterations
// ---------------------------------------------------------------------------

MatrixXd link_screw_axes(const std::vector<Matrix4d>& Mlist, const MatrixXd& Slist) {
    // TODO: Accumulate M_i = M_{0,1} M_{1,2} ... M_{i-1,i} in a loop; use mr::adjoint and mr::trans_inv.
    throw std::logic_error("TODO: link_screw_axes");
}

ForwardPass forward_iterations(const VectorXd& theta, const VectorXd& dtheta, const VectorXd& ddtheta,
                               const Vector3d& g, const std::vector<Matrix4d>& Mlist, const MatrixXd& Slist) {
    // TODO: Loop i = 0..n-1 (0-based indexing!).  Column 0 of V and Vdot is the base.
    //       mr::matrix_exp6(mr::vec_to_se3(-A_i * theta(i))) gives exp(-[A_i] theta_i).
    //       Do not forget T[n] = M_{n+1,n} = mr::trans_inv(Mlist[n]).  out.T must hold n+1 matrices.
    throw std::logic_error("TODO: forward_iterations");
}

// ---------------------------------------------------------------------------
// Checkpoint 3 -- backward iterations and full RNEA
// ---------------------------------------------------------------------------

std::pair<VectorXd, MatrixXd> backward_iterations(const ForwardPass& fwd, const Vector6d& Ftip,
                                                  const std::vector<Matrix6d>& Glist) {
    // TODO: Loop i = n-1 down to 0.  The wrench from link i+1 is transported with [Ad_{T_{i+1,i}}]^T =
    //       mr::adjoint(fwd.T[i + 1]).transpose().  Use your rigid_body_wrench.  return {tau, F};
    throw std::logic_error("TODO: backward_iterations");
}

VectorXd rnea(const VectorXd& theta, const VectorXd& dtheta, const VectorXd& ddtheta, const Vector3d& g,
              const Vector6d& Ftip, const std::vector<Matrix4d>& Mlist, const std::vector<Matrix6d>& Glist,
              const MatrixXd& Slist) {
    // TODO: Two lines: forward_iterations then backward_iterations (.first is tau).
    throw std::logic_error("TODO: rnea");
}

// ---------------------------------------------------------------------------
// Checkpoint 4 -- M, c, g and J^T F by RNEA calls
// ---------------------------------------------------------------------------

DynamicsTerms dynamics_terms(const VectorXd& theta, const VectorXd& dtheta, const Vector3d& g,
                             const std::vector<Matrix4d>& Mlist, const std::vector<Matrix6d>& Glist,
                             const MatrixXd& Slist) {
    // TODO: n + 2 calls of your rnea; switch gravity off (Vector3d::Zero()) where needed.
    //       Fill out.M (n x n, column j = rnea(theta, 0, e_j, 0, 0)), out.c and out.grav.
    throw std::logic_error("TODO: dynamics_terms");
}

VectorXd tip_wrench_torques(const VectorXd& theta, const Vector6d& Ftip, const std::vector<Matrix4d>& Mlist,
                            const std::vector<Matrix6d>& Glist, const MatrixXd& Slist) {
    // TODO: One rnea call with zero velocity, acceleration and gravity.
    throw std::logic_error("TODO: tip_wrench_torques");
}

// ---------------------------------------------------------------------------
// Checkpoint 5 -- Pinocchio conventions (pinocchio_model and pin_rnea are given in pinlite.cpp -- read them)
// ---------------------------------------------------------------------------

VectorXd pinocchio_inverse_dynamics(const PinModel& model, const PinInfo& info, const VectorXd& theta,
                                    const VectorXd& dtheta, const VectorXd& ddtheta,
                                    const std::optional<Vector6d>& Ftip) {
    // TODO: if (!Ftip) return pin_rnea(model, theta, dtheta, ddtheta);   (*Ftip is the wrench when it is given)
    //       std::vector<Vector6d> fext(model.njoints, Vector6d::Zero());  fext[model.njoints - 1] = ... (LINEAR first);
    //       then pin_rnea(model, theta, dtheta, ddtheta, fext).  Mind the three convention changes listed in lab.hpp.
    throw std::logic_error("TODO: pinocchio_inverse_dynamics");
}

// ---------------------------------------------------------------------------
// Checkpoint 6 -- application: torque profiles
// ---------------------------------------------------------------------------

MatrixXd torque_profile(const Robot& robot, const MatrixXd& theta, const MatrixXd& dtheta, const MatrixXd& ddtheta,
                        const std::optional<Vector6d>& Ftip) {
    // TODO: Loop over the rows (time samples) and call your rnea (row k as a vector: theta.row(k).transpose()).
    //       Ftip empty (std::nullopt) means a zero tip wrench.
    throw std::logic_error("TODO: torque_profile");
}

VectorXd peak_torques(const MatrixXd& tau) {
    // TODO: maximum over the rows of |tau|, one value per joint (column).
    throw std::logic_error("TODO: peak_torques");
}

}  // namespace lab
