// Auto-checks for SBEG 361 Lab 2 (angular velocities and twists) -- C++ track. Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*angular*,*adjoint*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics), the results of MR Sections 3.2.2 and 3.3.2
// (Prop. 3.8, Prop. 3.9, Prop. 3.21, Prop. 3.22, Example 3.23), finite differences of the data, and
// cpp_cases.json, which holds the numbers (and the random inputs) of the Python reference.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using lab::Matrix6d;
using lab::Vector6d;
using sbeg::as;
using sbeg::as_list;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// RANDOM_T of test_lab.py (20 random poses, stored in cpp_cases.json).
const std::vector<Matrix4d>& random_T() {
    static const std::vector<Matrix4d> Ts = as_list<Matrix4d>(cases()["random_T"]);
    return Ts;
}

Matrix4d mat4(double a00, double a01, double a02, double a03, double a10, double a11, double a12, double a13,
              double a20, double a21, double a22, double a23) {
    Matrix4d T;
    T << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, 0.0, 0.0, 0.0, 1.0;
    return T;
}

Vector6d vec6(double a, double b, double c, double d, double e, double f) {
    Vector6d V;
    V << a, b, c, d, e, f;
    return V;
}

/// np.isclose(a, b, rtol, atol): |a - b| <= atol + rtol |b|.
double isclose_tol(double b, double rtol = 1e-5, double atol = 1e-8) { return atol + rtol * std::abs(b); }

lab::TransformTree load_nav() { return lab::load_tree_json(sbeg::lab_path("../nav_tree.json")); }

/// The tracked screwdriver (given loader; read once).
const std::pair<Eigen::VectorXd, std::vector<Matrix4d>>& track() {
    static const std::pair<Eigen::VectorXd, std::vector<Matrix4d>> tr =
        lab::load_track_csv(sbeg::lab_path("../tool_track.csv"));
    return tr;
}

/// Row-wise mean of an N x 6 matrix as a 6-vector.
Vector6d row_mean(const MatrixXd& M) {
    const Eigen::RowVectorXd m = M.colwise().mean();
    Vector6d V = m.transpose();
    return V;
}

const Matrix4d T_SB_323 = mat4(-1, 0, 0, 4, 0, 1, 0, 0.4, 0, 0, -1, 0);  // MR Example 3.23
}  // namespace

// ------------------------------------------------------------- Checkpoint 1: angular velocity
TEST_CASE("test_angular_velocities_space_and_body") {
    const json& c = cases()["angular_velocities_space_and_body"];
    REQUIRE(c.size() == 10);
    for (std::size_t i = 0; i < 10; ++i) {
        const Matrix3d R = random_T()[i].topLeftCorner<3, 3>();
        const Vector3d w_b = as<Vector3d>(c[i]["w_b"]);
        INFO("RANDOM_T[" << i << "], w_b = " << w_b.transpose());
        const Matrix3d Rdot = R * mr::vec_to_so3(w_b);
        const auto [w_s, w_b2] = lab::angular_velocities(R, Rdot);
        CHECK_MAT_NEAR(w_b2, w_b, 1e-12);
        const Vector3d Rw = R * w_b;
        CHECK_MAT_NEAR(w_s, Rw, 1e-12);
        const Matrix3d rhs = R * mr::vec_to_so3(w_b) * R.transpose();  // Prop. 3.8
        CHECK_MAT_NEAR(mr::vec_to_so3(w_s), rhs, 1e-12);
    }
}

TEST_CASE("test_angular_velocity_turntable_example") {
    const double a = 0.5, b = 2.0, t = 1.0;
    const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, a * t))) *
                       mr::matrix_exp3(mr::vec_to_so3(Vector3d(b * t, 0, 0)));
    const Matrix3d Rdot = mr::vec_to_so3(Vector3d(0, 0, a)) * R + R * mr::vec_to_so3(Vector3d(b, 0, 0));
    const auto [w_s, w_b] = lab::angular_velocities(R, Rdot);
    const Vector3d ws_exp = Vector3d(0, 0, a) + b * R.col(0);
    const Vector3d wb_exp = R.transpose() * Vector3d(0, 0, a) + Vector3d(b, 0, 0);
    CHECK_MAT_NEAR(w_s, ws_exp, 1e-12);
    CHECK_MAT_NEAR(w_b, wb_exp, 1e-12);
    CHECK_MAT_NEAR(w_s, Vector3d(1.755165, 0.958851, 0.5), 1e-6);
    CHECK_MAT_NEAR(w_b, Vector3d(2.0, 0.454649, -0.208073), 1e-6);
    CHECK_SCALAR_NEAR(w_s.norm(), w_b.norm(), isclose_tol(w_b.norm()));
    const json& c = cases()["angular_velocity_turntable_example"];
    CHECK_MAT_NEAR(w_s, as<Vector3d>(c["w_s"]), 1e-12);
}

// ------------------------------------------------------------- Checkpoint 2: se(3) and adjoint
TEST_CASE("test_vec_to_se3_matches_mr") {
    for (const auto& c : cases()["vec_to_se3_matches_mr"]) {
        const Vector6d V = as<Vector6d>(c["V"]);
        CHECK_MAT_NEAR(lab::vec_to_se3(V), mr::vec_to_se3(V), 1e-12);
        CHECK_MAT_NEAR(lab::se3_to_vec(mr::vec_to_se3(V)), V, 1e-12);
    }
}

TEST_CASE("test_adjoint_matches_mr_and_properties") {
    const json& c = cases()["adjoint_matches_mr_and_properties"];
    REQUIRE(c.size() == 10);
    for (std::size_t i = 0; i < 10; ++i) {
        const Matrix4d& T1 = random_T()[i];
        const Matrix4d& T2 = random_T()[10 + i];
        INFO("T1 = RANDOM_T[" << i << "], T2 = RANDOM_T[" << 10 + i << "]");
        const Matrix6d A1 = lab::adjoint(T1);
        CHECK_MAT_NEAR(A1, mr::adjoint(T1), 1e-12);
        const Matrix4d T12 = T1 * T2;
        CHECK_MAT_NEAR(A1 * lab::adjoint(T2), lab::adjoint(T12), 1e-10);                  // (3.77)
        CHECK_MAT_NEAR(Matrix6d(A1.inverse()), lab::adjoint(mr::trans_inv(T1)), 1e-10);  // (3.78)
        const Vector6d V = as<Vector6d>(c[i]["V"]);
        const Vector6d AV = A1 * V;
        // the adjoint law in its two forms: T [V] T^-1 = [Ad_T V] and R [w] R^T = [R w]
        const Matrix4d rhs = T1 * mr::vec_to_se3(V) * mr::trans_inv(T1);
        CHECK_MAT_NEAR(lab::vec_to_se3(AV), rhs, 1e-10);
        const Matrix3d R = T1.topLeftCorner<3, 3>();
        const Vector3d Rw = R * V.head<3>();
        CHECK_MAT_NEAR(AV.head<3>(), Rw, 1e-12);
        const Matrix3d top = lab::vec_to_se3(AV).topLeftCorner<3, 3>();
        const Matrix3d rot_law = R * mr::vec_to_so3(V.head<3>()) * R.transpose();
        CHECK_MAT_NEAR(top, rot_law, 1e-10);
    }
}

// ------------------------------------------------------------- Checkpoint 3: twists and point velocities
TEST_CASE("test_twists_from_T_Tdot_mr_example_3_23") {
    const Vector6d V_b = vec6(0, 0, -2, 2.8, 4, 0);
    const Matrix4d Tdot = T_SB_323 * mr::vec_to_se3(V_b);
    const auto [V_s, V_b2] = lab::twists_from_T_Tdot(T_SB_323, Tdot);
    CHECK_MAT_NEAR(V_s, vec6(0, 0, 2, -2, -4, 0), 1e-12);
    CHECK_MAT_NEAR(V_b2, V_b, 1e-12);
    const Vector6d AdVb = mr::adjoint(T_SB_323) * V_b2;
    CHECK_MAT_NEAR(AdVb, V_s, 1e-12);
}

TEST_CASE("test_twists_and_angular_velocities_agree") {
    const json& c = cases()["twists_and_angular_velocities_agree"];
    REQUIRE(c.size() == 10);
    for (std::size_t i = 0; i < 10; ++i) {
        const Matrix4d& T = random_T()[i];
        const Vector6d V_s = as<Vector6d>(c[i]["V_s"]);
        INFO("RANDOM_T[" << i << "]");
        const Matrix4d Tdot = mr::vec_to_se3(V_s) * T;
        const auto [V_s2, V_b] = lab::twists_from_T_Tdot(T, Tdot);
        CHECK_MAT_NEAR(V_s2, V_s, 1e-12);
        const Vector6d Vb_exp = mr::adjoint(mr::trans_inv(T)) * V_s;  // (3.84)
        CHECK_MAT_NEAR(V_b, Vb_exp, 1e-10);
        const Matrix3d R = T.topLeftCorner<3, 3>(), Rdot = Tdot.topLeftCorner<3, 3>();
        const auto [w_s, w_b] = lab::angular_velocities(R, Rdot);
        CHECK_MAT_NEAR(w_s, V_s2.head<3>(), 1e-12);
        CHECK_MAT_NEAR(w_b, V_b.head<3>(), 1e-12);
        const Vector3d vb_exp = R.transpose() * Tdot.topRightCorner<3, 1>();  // v_b = R^T pdot
        CHECK_MAT_NEAR(V_b.tail<3>(), vb_exp, 1e-12);
    }
}

TEST_CASE("test_point_velocity") {
    const Vector6d V_b = vec6(0, 0, -2, 2.8, 4, 0);
    const Vector6d V_s = mr::adjoint(T_SB_323) * V_b;
    const Matrix4d Tdot = T_SB_323 * mr::vec_to_se3(V_b);
    CHECK_MAT_NEAR(lab::point_velocity(V_s, Vector3d::Zero()), V_s.tail<3>(), 1e-12);
    const Vector3d p = T_SB_323.topRightCorner<3, 1>(), pdot = Tdot.topRightCorner<3, 1>();
    CHECK_MAT_NEAR(lab::point_velocity(V_s, p), pdot, 1e-12);
    CHECK_MAT_NEAR(lab::point_velocity(V_s, p), Vector3d(-2.8, 4, 0), 1e-12);
    CHECK_MAT_NEAR(lab::point_velocity(V_s, Vector3d(2, -1, 0)), Vector3d::Zero(), 1e-12);  // on the screw axis
    const json& c = cases()["point_velocity"];
    const MatrixXd x_b = sbeg::as_mat(c["x_b"]);  // six body points, one per row
    REQUIRE(x_b.rows() == 6);
    const MatrixXd x_s = lab::transform_points(T_SB_323, x_b);
    MatrixXd hom(4, 6);
    hom.topRows(3) = x_b.transpose();
    hom.row(3).setOnes();
    const MatrixXd xdot_s = (Tdot * hom).topRows(3).transpose();  // d/dt (T x_b) = Tdot x_b
    CHECK_MAT_NEAR(lab::point_velocities(V_s, x_s), xdot_s, 1e-12);
    const MatrixXd xdot_b = xdot_s * T_SB_323.topLeftCorner<3, 3>();  // rows: (R^T xdot_s)^T
    CHECK_MAT_NEAR(lab::point_velocities(V_b, x_b), xdot_b, 1e-12);
    CHECK_MAT_NEAR(lab::point_velocities(V_s, x_s), sbeg::as_mat(c["xdot_s"]), 1e-12);
}

// ------------------------------------------------------------- Checkpoint 4: screws
TEST_CASE("test_screw_to_twist_and_back") {
    const Vector6d V = lab::screw_to_twist(Vector3d(2, -1, 0), Vector3d(0, 0, 1), 0.0, 2.0);  // MR Example 3.23, V_s
    CHECK_MAT_NEAR(V, vec6(0, 0, 2, -2, -4, 0), 1e-12);
    const lab::Screw sc = lab::twist_to_screw(V);
    CHECK_SCALAR_NEAR(sc.thetadot, 2.0, isclose_tol(2.0));
    CHECK_SCALAR_NEAR(sc.h, 0.0, isclose_tol(0.0));
    REQUIRE(sc.q.has_value());
    CHECK_MAT_NEAR(*sc.q, Vector3d(2, -1, 0), 1e-12);
    const Vector6d St = sc.S * sc.thetadot;
    CHECK_MAT_NEAR(St, V, 1e-12);
    for (const auto& c : cases()["screw_to_twist_and_back"]["random"]) {  // random screws, q off the origin
        const Vector3d s = as<Vector3d>(c["s"]), q = as<Vector3d>(c["q"]);
        const double h = c["h"].get<double>(), td = c["thetadot"].get<double>();
        INFO("s = " << s.transpose() << ", q = " << q.transpose() << ", h = " << h << ", thetadot = " << td);
        const lab::Screw r = lab::twist_to_screw(lab::screw_to_twist(q, s, h, td));
        CHECK_SCALAR_NEAR(r.h, h, isclose_tol(h));
        CHECK_SCALAR_NEAR(r.thetadot, td, isclose_tol(td));
        CHECK_MAT_NEAR(r.s_hat, s, 1e-10);
        REQUIRE(r.q.has_value());
        CHECK(lab::point_line_distance(q, *r.q, s) < 1e-10);  // same line
        CHECK(std::abs(r.q->dot(s)) < 1e-10);                // q is the point closest to the origin
    }
}

TEST_CASE("test_pure_translation_and_zero_twist") {
    const Vector6d V = vec6(0, 0, 0, 0, 0, 0.05);
    const lab::Screw sc = lab::twist_to_screw(V);
    CHECK(std::isinf(sc.h));
    CHECK_FALSE(sc.q.has_value());
    CHECK_SCALAR_NEAR(sc.thetadot, 0.05, isclose_tol(0.05));
    CHECK_MAT_NEAR(sc.S, vec6(0, 0, 0, 0, 0, 1), 1e-12);
    // q is ignored for h = INF (Python passes None)
    CHECK_MAT_NEAR(lab::screw_to_twist(Vector3d::Zero(), Vector3d(0, 0, 2), lab::INF, 0.05), V, 1e-12);
    CHECK_THROWS_AS(lab::twist_to_screw(Vector6d::Zero()), std::invalid_argument);
}

TEST_CASE("test_pitch_is_frame_invariant") {
    const json& c = cases()["pitch_is_frame_invariant"];
    const Vector6d V = as<Vector6d>(c["V"]);
    const double h = lab::twist_to_screw(V).h;
    CHECK_SCALAR_NEAR(h, c["h"].get<double>(), isclose_tol(c["h"].get<double>()));
    for (std::size_t i = 0; i < 5; ++i) {
        const Vector6d AdV = mr::adjoint(random_T()[i]) * V;
        CHECK_SCALAR_NEAR(lab::twist_to_screw(AdV).h, h, isclose_tol(h));
    }
}

// ------------------------------------------------------------- Checkpoint 5: navigated screwdriver, EOD gripper
TEST_CASE("test_body_twist_from_tracking_data") {
    const Eigen::VectorXd& t = track().first;
    const std::vector<Matrix4d>& Ts = track().second;
    REQUIRE(Ts.size() > 2);
    const double dt = t(1) - t(0);
    const auto [Vs, Vb] = lab::twists_from_samples(Ts, dt);
    const Eigen::Index n = static_cast<Eigen::Index>(Ts.size()) - 2;
    REQUIRE(Vs.rows() == n);
    REQUIRE(Vs.cols() == 6);
    REQUIRE(Vb.rows() == n);
    REQUIRE(Vb.cols() == 6);
    const Matrix4d step = mr::trans_inv(Ts[0]) * Ts[1];
    const Vector6d exact = mr::se3_to_vec(mr::matrix_log6(step)) / dt;  // constant screw motion
    CHECK_MAT_NEAR(exact, as<Vector6d>(cases()["body_twist_from_tracking_data"]["exact"]), 1e-9);
    const MatrixXd tiled = exact.transpose().replicate(n, 1);
    CHECK_MAT_NEAR(Vb, tiled, 5e-4);
    for (Eigen::Index k : {Eigen::Index(0), Eigen::Index(150), n - 1}) {  // V_s = [Ad_Tsb] V_b
        INFO("k = " << k);
        const Vector6d vb = Vb.row(k).transpose();
        const Vector6d expected = mr::adjoint(Ts[static_cast<std::size_t>(k + 1)]) * vb;
        CHECK_MAT_NEAR(Vs.row(k), expected, 1e-9);
    }
}

TEST_CASE("test_screw_axis_and_lead_of_the_screwdriver") {
    const lab::TransformTree nav = load_nav();
    const Eigen::VectorXd& t = track().first;
    const std::vector<Matrix4d>& Ts = track().second;
    REQUIRE(Ts.size() > 2);
    const auto [Vs, Vb] = lab::twists_from_samples(Ts, t(1) - t(0));
    REQUIRE(Vs.rows() > 0);
    REQUIRE(Vs.cols() == 6);
    REQUIRE(Vb.rows() > 0);
    REQUIRE(Vb.cols() == 6);
    const lab::Screw sc_b = lab::twist_to_screw(row_mean(Vb));
    CHECK_SCALAR_NEAR(sc_b.h * 2 * PI * 1000, 2.5, isclose_tol(2.5, 1e-5, 0.01));  // lead in mm/rev
    CHECK_SCALAR_NEAR(sc_b.thetadot, PI / 2, isclose_tol(PI / 2, 1e-3));
    CHECK_MAT_NEAR(sc_b.s_hat, Vector3d(0, 0, -1), 1e-6);
    REQUIRE(sc_b.q.has_value());
    CHECK_MAT_NEAR(*sc_b.q, Vector3d(0.05, 0, 0), 1e-6);
    const Vector6d V_ct = lab::adjoint(nav.lookup("ct_image", "tracker")) * row_mean(Vs);
    const lab::Screw sc = lab::twist_to_screw(V_ct);
    REQUIRE(sc.q.has_value());
    const Matrix4d T_plan = nav.lookup("ct_image", "plan_entry");
    const double angle_deg = lab::axis_angle_between(sc.s_hat, T_plan.block<3, 1>(0, 2)) * 180.0 / PI;
    CHECK_SCALAR_NEAR(angle_deg, 2.0, isclose_tol(2.0, 1e-5, 0.01));
    const double dist_mm = lab::point_line_distance(T_plan.topRightCorner<3, 1>(), *sc.q, sc.s_hat) * 1000;
    CHECK_SCALAR_NEAR(dist_mm, 1.0, isclose_tol(1.0, 1e-5, 0.01));
}

TEST_CASE("test_tool_tip_velocity") {
    lab::TransformTree nav = load_nav();
    const Eigen::VectorXd& t = track().first;
    const std::vector<Matrix4d>& Ts = track().second;
    REQUIRE(Ts.size() > 200);
    const double dt = t(1) - t(0);
    const auto [Vs, Vb] = lab::twists_from_samples(Ts, dt);
    const Eigen::Index n = static_cast<Eigen::Index>(Ts.size()) - 2;
    REQUIRE(Vs.rows() == n);
    REQUIRE(Vs.cols() == 6);
    REQUIRE(Vb.cols() == 6);
    const Vector3d tip_m = nav.lookup("tool_marker", "tool_tip").topRightCorner<3, 1>();
    const Vector3d v0 = lab::point_velocity(row_mean(Vb), Vector3d::Zero()) * 1000.0;
    CHECK_MAT_NEAR(v0, Vector3d(0, 78.527, -0.625), 1e-3);
    const Vector3d vt = lab::point_velocity(row_mean(Vb), tip_m) * 1000.0;
    CHECK_MAT_NEAR(vt, Vector3d(0, 0, -0.625), 1e-3);
    std::vector<Vector3d> tips;
    for (const Matrix4d& T : Ts) tips.push_back(lab::transform_point(T, tip_m));  // tip track, tracker coordinates
    MatrixXd v_tip(n, 3), fd(n, 3);
    for (Eigen::Index k = 0; k < n; ++k) {
        const std::size_t i = static_cast<std::size_t>(k);
        const Vector6d V = Vs.row(k).transpose();
        v_tip.row(k) = lab::point_velocity(V, tips[i + 1]).transpose();
        fd.row(k) = ((tips[i + 2] - tips[i]) / (2 * dt)).transpose();  // central difference of the tip track
    }
    CHECK_MAT_NEAR(v_tip, fd, 1e-9);
    nav.add("tracker", "tool_marker", Ts[0]);
    const Matrix3d R_plan_tracker = nav.lookup("plan_entry", "tracker").topLeftCorner<3, 3>();
    const Vector3d v199 = v_tip.row(199).transpose();
    const Vector3d v_plan = R_plan_tracker * v199 * 1000.0;  // mm/s at t = 4 s, plan coordinates
    CHECK_MAT_NEAR(v_plan, Vector3d(0, -0.021812, 0.624619), 1e-5);
}

TEST_CASE("test_eod_gripper_twist_in_camera_frame") {
    const lab::TransformTree eod = lab::load_tree_json(sbeg::lab_path("../eod_tree.json"));
    const Vector6d V_g = vec6(0, 0, 0.5, 0, 0, 0);
    const Matrix4d T_cg = eod.lookup("mast_camera", "gripper");
    const Vector6d V_c = lab::adjoint(T_cg) * V_g;
    CHECK_MAT_NEAR(V_c, vec6(0, 0, 0.5, 0.2, 0, 0), 1e-9);
    const lab::Screw sc = lab::twist_to_screw(V_c);
    CHECK_SCALAR_NEAR(sc.h, 0.0, 1e-12);
    CHECK_SCALAR_NEAR(sc.thetadot, 0.5, isclose_tol(0.5));
    const Vector3d z_g = T_cg.block<3, 1>(0, 2);
    CHECK_MAT_NEAR(sc.s_hat, z_g, 1e-9);
    REQUIRE(sc.q.has_value());
    CHECK(lab::point_line_distance(T_cg.topRightCorner<3, 1>(), *sc.q, sc.s_hat) < 1e-9);  // through the gripper origin
    const Vector6d V_base = lab::adjoint(eod.lookup("base_link", "gripper")) * V_g;
    CHECK_MAT_NEAR(V_base, vec6(0.5, 0, 0, 0, 0.2, 0), 1e-9);
    const Vector3d r_g(0, 0.04, 0.1);  // a point on a jaw, gripper coordinates
    const Vector3d v_g = lab::point_velocity(V_g, r_g);
    CHECK_MAT_NEAR(v_g, Vector3d(-0.02, 0, 0), 1e-12);
    const Vector3d r_c = T_cg.topLeftCorner<3, 3>() * r_g + T_cg.topRightCorner<3, 1>();
    const Vector3d v_c_exp = T_cg.topLeftCorner<3, 3>() * v_g;
    CHECK_MAT_NEAR(lab::point_velocity(V_c, r_c), v_c_exp, 1e-12);
}
