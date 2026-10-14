// Auto-checks for SBEG 461 Lab 2 (RNEA, verified against modern_robotics and Pinocchio) -- C++ track.
// Mirrors test_lab.py test by test (a parametrised Python test = one C++ test case that loops over the robots).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*spatial_inertia*,*rigid_body*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: inverse_dynamics, mass_matrix, vel_quadratic_forces,
// gravity_forces, jacobian_body, fkin_space, inverse_dynamics_trajectory), the classical Newton/Euler equations
// (MR Eqs. 8.22-8.23), the closed-form 2R dynamics (MR Eq. 8.10), the Week 2 workshop hand calculation (W3) and
// cpp_cases.json, which holds the numbers of the Python reference and of the real Pinocchio (pin.rnea).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using lab::Matrix6d;
using lab::Vector6d;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double TOL = 1e-9;
const std::vector<std::string> ROBOTS = {"mr_2r_point", "arm_exo_2r", "eod_3r", "ur5"};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// Provided (identical) in starter.cpp and the reference implementation.
lab::Robot robot(const std::string& name) { return lab::load_robot(name); }

Matrix4d translation(double x, double y, double z) { return mr::rp_to_trans(Matrix3d::Identity(), Vector3d(x, y, z)); }

Vector6d vec6(double a, double b, double c, double d, double e, double f) {
    Vector6d v;
    v << a, b, c, d, e, f;
    return v;
}

/// Workshop W3 model: planar 2R arm, point masses 2 kg and 1.5 kg, L1 = 0.30 m, L2 = 0.35 m.
struct W3 {
    std::vector<Matrix4d> Ml{translation(0.30, 0, 0), translation(0.35, 0, 0), Matrix4d::Identity()};
    std::vector<Matrix6d> Gl{Matrix6d(vec6(0, 0, 0, 2.0, 2.0, 2.0).asDiagonal()),
                             Matrix6d(vec6(0, 0, 0, 1.5, 1.5, 1.5).asDiagonal())};
    MatrixXd S = MatrixXd::Zero(6, 2);
    W3() {
        S.col(0) = vec6(0, 0, 1, 0, 0, 0);
        S.col(1) = vec6(0, 0, 1, 0, -0.30, 0);
    }
};

lab::ForwardPass w3_forward(const W3& w) {
    return lab::forward_iterations(Eigen::Vector2d(-PI / 2, PI / 2), Eigen::Vector2d(1, 2), Eigen::Vector2d(0, -1),
                                   Vector3d(0, -9.81, 0), w.Ml, w.S);
}
}  // namespace

// ---------------------------------------------------------------- checkpoint 1: one rigid body
TEST_CASE("test_spatial_inertia_blocks") {
    Matrix3d I;
    I << 0.2, 0.01, 0, 0.01, 0.3, 0.02, 0, 0.02, 0.4;
    const Matrix6d G = lab::spatial_inertia(I, 2.5);
    CHECK_MAT_NEAR((G.topLeftCorner<3, 3>()), I, 1e-15);
    CHECK_MAT_NEAR((G.bottomRightCorner<3, 3>()), 2.5 * Matrix3d::Identity(), 1e-15);
    CHECK_MAT_NEAR((G.topRightCorner<3, 3>()), Matrix3d::Zero(), 0.0);
    CHECK_MAT_NEAR((G.bottomLeftCorner<3, 3>()), Matrix3d::Zero(), 0.0);
}

TEST_CASE("test_transform_spatial_inertia_workshop_W1") {
    // Prosthetic shank + foot about the knee (workshop W1): rotational block = Steiner, off-diagonal = m[q]^T.
    Matrix3d Ic;
    Ic << 0.04412, 0, 0.00564, 0, 0.04895, 0, 0.00564, 0, 0.00741;
    const double m = 2.0;
    const Vector3d c(0.02, 0, -0.294);
    const Matrix6d Gk = lab::transform_spatial_inertia(lab::spatial_inertia(Ic, m), mr::rp_to_trans(Matrix3d::Identity(), -c));
    const Matrix3d steiner = Ic + m * (c.dot(c) * Matrix3d::Identity() - c * c.transpose());
    CHECK_MAT_NEAR((Gk.topLeftCorner<3, 3>()), steiner, 1e-12);
    Matrix3d off;
    off << 0, 0.588, 0, -0.588, 0, -0.04, 0, 0.04, 0;
    CHECK_MAT_NEAR((Gk.topRightCorner<3, 3>()), off, 1e-12);
    CHECK_MAT_NEAR(Gk, Gk.transpose(), 1e-14);
    CHECK_MAT_NEAR(Gk, as<Matrix6d>(cases()["transform_spatial_inertia_workshop_W1"]["Gk"]), 1e-12);
}

TEST_CASE("test_transform_spatial_inertia_kinetic_energy_invariant") {
    const json& c = cases()["transform_spatial_inertia_kinetic_energy_invariant"];  // rng 1 of test_lab.py
    const Matrix6d Gb = lab::spatial_inertia(as<Matrix3d>(c["I"]), c["m"].get<double>());
    const Matrix4d Tba = as<Matrix4d>(c["T_ba"]);
    const Matrix6d Ga = lab::transform_spatial_inertia(Gb, Tba);
    const Vector6d Va = as<Vector6d>(c["Va"]);
    const Vector6d Vb = mr::adjoint(Tba) * Va;
    CHECK_SCALAR_NEAR(0.5 * Va.dot(Ga * Va), 0.5 * Vb.dot(Gb * Vb), 1e-12);  // MR p. 288
    CHECK_MAT_NEAR(Ga, as<Matrix6d>(c["Ga"]), 1e-12);
}

TEST_CASE("test_rigid_body_wrench_matches_newton_euler_mr_8_22_8_23") {
    const Matrix3d I = Vector3d(0.8667, 0.7042, 0.3708).asDiagonal();
    const double m = 20.0;
    for (const auto& c : cases()["rigid_body_wrench_matches_newton_euler_mr_8_22_8_23"]) {  // rng 2, 5 cases
        const Vector6d V = as<Vector6d>(c["V"]), Vd = as<Vector6d>(c["Vdot"]);
        const Vector6d F = lab::rigid_body_wrench(lab::spatial_inertia(I, m), V, Vd);
        const Vector3d w = V.head<3>(), v = V.tail<3>(), wd = Vd.head<3>(), vd = Vd.tail<3>();
        CHECK_MAT_NEAR(F.head<3>(), (I * wd + w.cross(I * w)).eval(), 1e-12);
        CHECK_MAT_NEAR(F.tail<3>(), (m * (vd + w.cross(v))).eval(), 1e-12);
        CHECK_MAT_NEAR(F, as<Vector6d>(c["F"]), 1e-12);
    }
}

TEST_CASE("test_rigid_body_wrench_quiz_Q2_and_frame_invariance_mr_8_43") {
    const Matrix6d G = vec6(2, 3, 4, 50, 50, 50).asDiagonal();
    CHECK_MAT_NEAR(lab::rigid_body_wrench(G, vec6(0, 0, 0.5, 2, 0, 0), Vector6d::Zero()), vec6(0, 0, 0, 0, 50, 0),
                   1e-12);
    const json& c = cases()["rigid_body_wrench_quiz_Q2_and_frame_invariance_mr_8_43"];  // rng 3
    const Matrix6d Gb = lab::spatial_inertia(Vector3d(0.1, 0.2, 0.3).asDiagonal(), 1.5);
    const Matrix4d Tba = as<Matrix4d>(c["T_ba"]);
    const Vector6d Vb = as<Vector6d>(c["Vb"]), Vdb = as<Vector6d>(c["Vdotb"]);
    const Matrix6d Ad = mr::adjoint(Tba);
    const Vector6d Va = Ad.partialPivLu().solve(Vb), Vda = Ad.partialPivLu().solve(Vdb);
    const Vector6d Fa = lab::rigid_body_wrench(lab::transform_spatial_inertia(Gb, Tba), Va, Vda);
    CHECK_MAT_NEAR(Fa, (Ad.transpose() * lab::rigid_body_wrench(Gb, Vb, Vdb)).eval(), 1e-10);
    CHECK_MAT_NEAR(Fa, as<Vector6d>(c["Fa"]), 1e-10);
}

// ---------------------------------------------------------------- checkpoint 2: forward iterations
TEST_CASE("test_link_screw_axes") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const MatrixXd A = lab::link_screw_axes(r.Mlist, r.Slist);
        REQUIRE(A.rows() == 6);
        REQUIRE(A.cols() == r.n);
        Matrix4d Mi = Matrix4d::Identity();
        for (int i = 0; i < r.n; ++i) {
            Mi = (Mi * r.Mlist[static_cast<std::size_t>(i)]).eval();
            CHECK_MAT_NEAR((mr::adjoint(Mi) * A.col(i)).eval(), r.Slist.col(i), 1e-12);
        }
        CHECK_MAT_NEAR(A, as_mat(cases()["link_screw_axes"][name]), 1e-12);
    }
}

TEST_CASE("test_forward_transforms_and_twists_vs_fk_and_body_jacobian") {
    // T_{i,i-1} = T_{0,i}^{-1} T_{0,i-1}  and  V_i = J_{b,i}(theta) dtheta  (MR p. 293, Eq. 8.56 notation).
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const int n = r.n;
        const json& c = cases()["forward_transforms_and_twists_vs_fk_and_body_jacobian"][name];  // rng 4
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]), ddth = as_vec(c["ddtheta"]);
        const lab::ForwardPass out = lab::forward_iterations(th, dth, ddth, r.g, r.Mlist, r.Slist);
        REQUIRE(out.V.rows() == 6);
        REQUIRE(out.V.cols() == n + 1);
        REQUIRE(out.Vdot.rows() == 6);
        REQUIRE(out.Vdot.cols() == n + 1);
        REQUIRE(out.T.size() == static_cast<std::size_t>(n + 1));
        Vector6d Vd0;
        Vd0 << 0, 0, 0, -r.g;
        CHECK_MAT_NEAR(out.Vdot.col(0), Vd0, 0.0);
        Matrix4d Mi = Matrix4d::Identity(), T_prev = Matrix4d::Identity();
        for (int i = 0; i < n; ++i) {
            INFO("i = " << i);
            Mi = (Mi * r.Mlist[static_cast<std::size_t>(i)]).eval();
            const Matrix4d T0i = mr::fkin_space(Mi, r.Slist.leftCols(i + 1), th.head(i + 1));
            CHECK_MAT_NEAR(out.T[static_cast<std::size_t>(i)], (mr::trans_inv(T0i) * T_prev).eval(), 1e-10);
            const MatrixXd Jb = mr::jacobian_body(mr::adjoint(mr::trans_inv(Mi)) * r.Slist.leftCols(i + 1),
                                                  th.head(i + 1));
            CHECK_MAT_NEAR(out.V.col(i + 1), (Jb * dth.head(i + 1)).eval(), 1e-10);
            T_prev = T0i;
        }
        CHECK_MAT_NEAR(out.T[static_cast<std::size_t>(n)], mr::trans_inv(r.Mlist[static_cast<std::size_t>(n)]), 1e-12);
        CHECK_MAT_NEAR(out.V, as_mat(c["V"]), 1e-10);
    }
}

TEST_CASE("test_forward_accelerations_are_time_derivative_of_twists") {
    // With g = 0, Vdot_i must equal dV_i/dt along theta(t) = theta0 + dtheta t + ddtheta t^2/2 (central difference).
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const json& c = cases()["forward_accelerations_are_time_derivative_of_twists"][name];  // rng 5
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]), ddth = as_vec(c["ddtheta"]);
        const Vector3d z3 = Vector3d::Zero();
        const lab::ForwardPass out = lab::forward_iterations(th, dth, ddth, z3, r.Mlist, r.Slist);
        const double h = 1e-5;
        auto V_at = [&](double t) -> MatrixXd {
            const VectorXd tht = th + dth * t + 0.5 * ddth * t * t;
            const VectorXd dtht = dth + ddth * t;
            return lab::forward_iterations(tht, dtht, ddth, z3, r.Mlist, r.Slist).V;
        };
        const MatrixXd fd = (V_at(h) - V_at(-h)) / (2 * h);
        CHECK_MAT_NEAR(out.Vdot, fd, 1e-6);
        CHECK_MAT_NEAR(out.Vdot, as_mat(c["Vdot"]), 1e-10);
    }
}

TEST_CASE("test_forward_workshop_W3_hand_values") {
    const W3 w;
    const lab::ForwardPass out = w3_forward(w);
    REQUIRE(out.V.cols() == 3);
    REQUIRE(out.Vdot.cols() == 3);
    CHECK_MAT_NEAR(out.V.col(2), vec6(0, 0, 3, 0.3, 1.05, 0), 1e-12);
    CHECK_MAT_NEAR(out.Vdot.col(1), vec6(0, 0, 0, -9.81, 0, 0), 1e-12);
    CHECK_MAT_NEAR(out.Vdot.col(2), vec6(0, 0, -1, 0, 8.86, 0), 1e-12);
}

// ---------------------------------------------------------------- checkpoint 3: backward iterations and full RNEA
TEST_CASE("test_backward_workshop_W3_hand_values") {
    const W3 w;
    const lab::ForwardPass fwd = w3_forward(w);
    const auto [tau, F] = lab::backward_iterations(fwd, Vector6d::Zero(), w.Gl);
    REQUIRE(F.rows() == 6);
    REQUIRE(F.cols() == 3);
    CHECK_MAT_NEAR(F.col(1), vec6(0, 0, 0, -4.725, 14.64, 0), 1e-12);
    CHECK_MAT_NEAR(F.col(0), vec6(0, 0, 5.124, -34.86, -4.725, 0), 1e-12);
    CHECK_MAT_NEAR(tau, Eigen::Vector2d(3.7065, 5.124), 1e-12);
}

TEST_CASE("test_rnea_mr_2r_closed_form_mr_eq_8_10") {
    const lab::Robot r = robot("mr_2r_point");
    for (const auto& c : cases()["rnea_mr_2r_closed_form_mr_eq_8_10"]) {  // rng 6, 5 states
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]), ddth = as_vec(c["ddtheta"]);
        const double c2 = std::cos(th(1)), s2 = std::sin(th(1));
        Eigen::Matrix2d M;
        M << 3 + 2 * c2, 1 + c2, 1 + c2, 1;
        const Eigen::Vector2d cv(-s2 * (2 * dth(0) * dth(1) + dth(1) * dth(1)), s2 * dth(0) * dth(0));
        const Eigen::Vector2d gv =
            9.81 * Eigen::Vector2d(2 * std::cos(th(0)) + std::cos(th.sum()), std::cos(th.sum()));
        const VectorXd tau = lab::rnea(th, dth, ddth, r.g, Vector6d::Zero(), r.Mlist, r.Glist, r.Slist);
        CHECK_MAT_NEAR(tau, (M * ddth + cv + gv).eval(), 1e-10);
        CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-10);
    }
}

TEST_CASE("test_rnea_vs_modern_robotics_states_file") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const std::vector<lab::State> sts = lab::load_states(name);
        const json& ref = cases()["rnea_vs_modern_robotics_states_file"][name];
        REQUIRE(sts.size() == ref.size());
        for (std::size_t k = 0; k < sts.size(); ++k) {
            INFO("state " << k);
            const lab::State& s = sts[k];
            const VectorXd tau = lab::rnea(s.theta, s.dtheta, s.ddtheta, r.g, s.Ftip, r.Mlist, r.Glist, r.Slist);
            CHECK_MAT_NEAR(tau, mr::inverse_dynamics(s.theta, s.dtheta, s.ddtheta, r.g, s.Ftip, r.Mlist, r.Glist, r.Slist),
                           TOL);
            CHECK_MAT_NEAR(tau, as_vec(ref[k]), TOL);
        }
    }
}

TEST_CASE("test_rnea_vs_modern_robotics_docstring_example") {
    // The 3-link example printed in the modern_robotics InverseDynamics docstring.
    Matrix4d M01, M12, M23, M34;
    M01 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.089159, 0, 0, 0, 1;
    M12 << 0, 0, 1, 0.28, 0, 1, 0, 0.13585, -1, 0, 0, 0, 0, 0, 0, 1;
    M23 << 1, 0, 0, 0, 0, 1, 0, -0.1197, 0, 0, 1, 0.395, 0, 0, 0, 1;
    M34 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.14225, 0, 0, 0, 1;
    const std::vector<Matrix6d> Gl = {Matrix6d(vec6(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7).asDiagonal()),
                                      Matrix6d(vec6(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393).asDiagonal()),
                                      Matrix6d(vec6(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275).asDiagonal())};
    MatrixXd S(6, 3);
    S.col(0) = vec6(1, 0, 1, 0, 1, 0);
    S.col(1) = vec6(0, 1, 0, -0.089, 0, 0);
    S.col(2) = vec6(0, 1, 0, -0.089, 0, 0.425);
    const VectorXd tau = lab::rnea(Vector3d(0.1, 0.1, 0.1), Vector3d(0.1, 0.2, 0.3), Vector3d(2, 1.5, 1),
                                   Vector3d(0, 0, -9.8), vec6(1, 1, 1, 1, 1, 1), {M01, M12, M23, M34}, Gl, S);
    CHECK_MAT_NEAR(tau, Vector3d(74.69616155, -33.06766016, -3.23057314), 1e-6);
}

// ---------------------------------------------------------------- checkpoint 4: M, c, g and J^T F by RNEA calls
TEST_CASE("test_dynamics_terms_vs_modern_robotics") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const std::vector<lab::State> sts = lab::load_states(name);
        const json& ref = cases()["dynamics_terms_vs_modern_robotics"][name];
        REQUIRE(sts.size() == ref.size());
        for (std::size_t k = 0; k < sts.size(); ++k) {
            INFO("state " << k);
            const lab::State& s = sts[k];
            const lab::DynamicsTerms d = lab::dynamics_terms(s.theta, s.dtheta, r.g, r.Mlist, r.Glist, r.Slist);
            CHECK_MAT_NEAR(d.M, mr::mass_matrix(s.theta, r.Mlist, r.Glist, r.Slist), TOL);
            CHECK_MAT_NEAR(d.c, mr::vel_quadratic_forces(s.theta, s.dtheta, r.Mlist, r.Glist, r.Slist), TOL);
            CHECK_MAT_NEAR(d.grav, mr::gravity_forces(s.theta, r.g, r.Mlist, r.Glist, r.Slist), TOL);
            CHECK_MAT_NEAR(d.M, as_mat(ref[k]["M"]), TOL);
            REQUIRE(d.M.rows() == r.n);
            REQUIRE(d.M.cols() == r.n);
            CHECK_MAT_NEAR(d.M, d.M.transpose(), 1e-12);
            const Eigen::SelfAdjointEigenSolver<MatrixXd> eig(d.M, Eigen::EigenvaluesOnly);
            CHECK(eig.eigenvalues().minCoeff() > 0.0);
            const VectorXd sum = d.M * s.ddtheta + d.c + d.grav +
                                 lab::tip_wrench_torques(s.theta, s.Ftip, r.Mlist, r.Glist, r.Slist);
            CHECK_MAT_NEAR(sum, lab::rnea(s.theta, s.dtheta, s.ddtheta, r.g, s.Ftip, r.Mlist, r.Glist, r.Slist), TOL);
        }
    }
}

TEST_CASE("test_tip_wrench_torques_equal_JbT_F_mr_eq_5_26") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const json& c = cases()["tip_wrench_torques_equal_JbT_F_mr_eq_5_26"][name];  // rng 7
        const VectorXd th = as_vec(c["theta"]);
        const Vector6d Ft = as<Vector6d>(c["Ftip"]);
        Matrix4d Mee = Matrix4d::Identity();
        for (const Matrix4d& M : r.Mlist) Mee = (Mee * M).eval();
        const MatrixXd Jb = mr::jacobian_body(mr::adjoint(mr::trans_inv(Mee)) * r.Slist, th);
        const VectorXd tau = lab::tip_wrench_torques(th, Ft, r.Mlist, r.Glist, r.Slist);
        CHECK_MAT_NEAR(tau, (Jb.transpose() * Ft).eval(), TOL);
        CHECK_MAT_NEAR(tau, as_vec(c["tau"]), TOL);
    }
}

// ---------------------------------------------------------------- checkpoint 5: Pinocchio conventions
TEST_CASE("test_pinocchio_matches_modern_robotics") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot r = robot(name);
        const auto [model, info] = lab::pinocchio_model(r);  // provided
        const std::vector<lab::State> sts = lab::load_states(name);
        const json& ref = cases()["pinocchio_matches_modern_robotics"][name];  // values of the real pin.rnea
        REQUIRE(sts.size() == ref["tau"].size());
        REQUIRE(sts.size() >= 2);
        for (std::size_t k = 0; k < sts.size(); ++k) {
            INFO("state " << k);
            const lab::State& s = sts[k];
            const VectorXd want = mr::inverse_dynamics(s.theta, s.dtheta, s.ddtheta, r.g, s.Ftip, r.Mlist, r.Glist,
                                                       r.Slist);
            const VectorXd got = lab::pinocchio_inverse_dynamics(model, info, s.theta, s.dtheta, s.ddtheta, s.Ftip);
            CHECK_MAT_NEAR(got, want, TOL);
            CHECK_MAT_NEAR(got, as_vec(ref["tau"][k]), TOL);
        }
        const lab::State& s = sts[1];
        const VectorXd got = lab::pinocchio_inverse_dynamics(model, info, s.theta, s.dtheta, s.ddtheta);
        CHECK_MAT_NEAR(got, mr::inverse_dynamics(s.theta, s.dtheta, s.ddtheta, r.g, Vector6d::Zero(), r.Mlist, r.Glist,
                                                 r.Slist), TOL);
        CHECK_MAT_NEAR(got, as_vec(ref["tau_no_tip"]), TOL);
    }
}

// ---------------------------------------------------------------- checkpoint 6: application -- torque profiles
TEST_CASE("test_torque_profile_vs_modern_robotics_trajectory") {
    const std::vector<std::pair<std::string, std::string>> params = {{"eod_3r", "eod_lift.csv"},
                                                                     {"arm_exo_2r", "exo_reach.csv"}};
    for (const auto& param : params) {
        const std::string& name = param.first;
        const std::string& fname = param.second;
        INFO("robot = " << name << ", file = " << fname);
        const lab::Robot r = robot(name);
        const lab::Trajectory tr = lab::load_trajectory(fname);
        const Eigen::Index N = tr.t.size();
        const MatrixXd want = mr::inverse_dynamics_trajectory(tr.theta, tr.dtheta, tr.ddtheta, r.g,
                                                              MatrixXd::Zero(N, 6), r.Mlist, r.Glist, r.Slist);
        const MatrixXd tau = lab::torque_profile(r, tr.theta, tr.dtheta, tr.ddtheta);
        REQUIRE(tau.rows() == N);
        REQUIRE(tau.cols() == r.n);
        CHECK_MAT_NEAR(tau, want, TOL);
        const VectorXd want_peak = want.cwiseAbs().colwise().maxCoeff().transpose();
        CHECK_MAT_NEAR(lab::peak_torques(tau), want_peak, TOL);
        const json& c = cases()["torque_profile_vs_modern_robotics_trajectory"][name];
        CHECK(N == c["N"].get<Eigen::Index>());
        CHECK_MAT_NEAR(lab::peak_torques(tau), as_vec(c["peak"]), TOL);
    }
}

TEST_CASE("test_peak_torques_simple") {
    Eigen::Matrix2d tau;
    tau << 1.0, -5.0, -3.0, 2.0;
    CHECK_MAT_NEAR(lab::peak_torques(tau), Eigen::Vector2d(3.0, 5.0), 1e-12);
}
