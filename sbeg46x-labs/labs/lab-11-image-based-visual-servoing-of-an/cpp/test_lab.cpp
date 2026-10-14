// Auto-checks for SBEG 46X Lab 11 (image-based visual servoing of an endoscope-holding arm, simulated pinhole
// camera) -- C++ track. Mirrors test_lab.py test by test (27 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*intrinsic*,*project_points*"      (like pytest -k)
// Ground truth: the worked numbers of Corke, Robotics, Vision and Control (Sec. 11.1, pp. 254-258, and Sec. 15.2.1,
// pp. 460-462), the camera matrix K [R | p], finite differences of the projection under the matrix exponential of the
// camera twist (values of the Python test, cpp_cases.json), mrcpp (matrix_exp6, matrix_log6, jacobian_body,
// fkin_body), the course model ur5_laparoscope (course_models.cpp), closed-form cases, data/expected.json and
// cpp_cases.json (numbers of the Python reference, including the random inputs that test_lab.py draws), so both
// languages are checked against the same values.
//
// Plot files: when the functions they use work, the tests write lab11_cpp_ibvs.csv and lab11_cpp_pbvs.csv (Corke
// start, Figures 1-2) and lab11_cpp_endoscope.csv (default endoscope run, Figure 4) into the folder you run lab_tests
// from; plot them with  python cpp/plot_results.py
#include "lab.hpp"
#include "course_models.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using lab::Vector6d;
using sbeg::as;
using sbeg::json;

namespace {
const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}
const json& EXP() {
    static const json E = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return E;
}
const lab::Scene& SC() {
    static const lab::Scene s = lab::load_scene();
    return s;
}
Matrix3d K_I() {
    Matrix3d K;
    K << 800, 0, 512, 0, 800, 512, 0, 0, 1;
    return K;
}
Matrix3d K_E() {
    Matrix3d K;
    K << 600, 0, 320, 0, 600, 240, 0, 0, 1;
    return K;
}
MatrixXd row2(double u, double v) {
    MatrixXd p(1, 2);
    p << u, v;
    return p;
}
MatrixXd row3(double x, double y, double z) {
    MatrixXd p(1, 3);
    p << x, y, z;
    return p;
}
/// pixels of camera-frame points (rows) with K [I | 0]
MatrixXd pix(const Matrix3d& K, const MatrixXd& Pc) {
    MatrixXd p(Pc.rows(), 2);
    for (Eigen::Index i = 0; i < Pc.rows(); ++i) {
        const Vector3d q = K * (Pc.row(i).transpose() / Pc(i, 2));
        p(i, 0) = q(0);
        p(i, 1) = q(1);
    }
    return p;
}
/// points (rows) transformed by T
MatrixXd transform(const Matrix4d& T, const MatrixXd& P) {
    MatrixXd Q(P.rows(), 3);
    for (Eigen::Index i = 0; i < P.rows(); ++i)
        Q.row(i) = (T.topLeftCorner<3, 3>() * P.row(i).transpose() + T.block<3, 1>(0, 3)).transpose();
    return Q;
}
VectorXd stack(const MatrixXd& e) {
    VectorXd v(2 * e.rows());
    for (Eigen::Index i = 0; i < e.rows(); ++i) {
        v(2 * i) = e(i, 0);
        v(2 * i + 1) = e(i, 1);
    }
    return v;
}
struct Setup {
    MatrixXd P, xy, ps;
    Matrix4d Tg, T_cst;
};
Setup scene_setup() {
    const lab::Target t = lab::target_points(SC());
    Setup s;
    s.P = t.P_w;
    s.xy = t.xy_t;
    s.Tg = SC().ibvs.T_wc_star;
    s.T_cst = mr::trans_inv(s.Tg) * SC().T_wt;
    s.ps = sbeg::as_mat(EXP().at("p_star"));
    return s;
}
double rel_tol(double expected, double rel) { return std::max(rel * std::abs(expected), 1e-12); }
double rel_tol(const json& expected, double rel) { return rel_tol(expected.get<double>(), rel); }
void write_run_csv(const std::string& name, const lab::ServoRun& r) {
    const Eigen::Index n1 = static_cast<Eigen::Index>(r.T.size());
    MatrixXd D(n1, 13);
    for (Eigen::Index k = 0; k < n1; ++k) {
        const MatrixXd& p = r.p[static_cast<std::size_t>(k)];
        D(k, 0) = static_cast<double>(k);
        D(k, 1) = r.err(k);
        for (int i = 0; i < 4; ++i) {
            D(k, 2 + 2 * i) = p(i, 0);
            D(k, 3 + 2 * i) = p(i, 1);
        }
        D.block<1, 3>(k, 10) = r.T[static_cast<std::size_t>(k)].block<3, 1>(0, 3).transpose();
    }
    sbeg::write_csv(name, {"k", "err", "u1", "v1", "u2", "v2", "u3", "v3", "u4", "v4", "x", "y", "z"}, D);
}
}  // namespace

// ------------------------------------------------------------------------------ data
TEST_CASE("test_data_files_consistent") {
    const auto Pp = lab::load_calib();
    const int n = EXP().at("calib").at("n_points").get<int>();
    REQUIRE(Pp.first.rows() == n);
    CHECK(Pp.second.rows() == n);
    const lab::Target t = lab::target_points(SC());
    CHECK(t.P_w.rows() == 4);
    CHECK_MAT_NEAR(t.P_w.col(2), VectorXd::Constant(4, 3.0), 1e-12);
    const Matrix4d T = SC().calib.T_wc;
    const Matrix3d R = T.topLeftCorner(3, 3);
    CHECK_MAT_NEAR(R.transpose() * R, Matrix3d::Identity(), 1e-12);
    // C++ only: the measured pixels lie within 2 px of the true projection (calls your project_points)
    const auto pz = lab::project_points(K_E(), T, Pp.first);
    CHECK_MAT_NEAR(pz.first, Pp.second, 2.0);
}

// ------------------------------------------------------------------------------ Part A
TEST_CASE("test_intrinsic_matrix") {
    Matrix3d K;
    K << 1500, 0, 640, 0, 1500, 512, 0, 0, 1;
    CHECK_MAT_NEAR(lab::intrinsic_matrix(0.015, 10e-6, 10e-6, 640, 512), K, 1e-9);
    CHECK_MAT_NEAR(lab::camera_K(SC().endoscope_camera), K_E(), 1e-9);
    Matrix3d K2;
    K2 << 800, 0, 100, 0, 1000, 50, 0, 0, 1;
    CHECK_MAT_NEAR(lab::intrinsic_matrix(0.004, 5e-6, 4e-6, 100, 50), K2, 1e-9);
}

TEST_CASE("test_project_points_corke_examples") {
    Matrix3d K;
    K << 1500, 0, 640, 0, 1500, 512, 0, 0, 1;
    auto pz = lab::project_points(K, Matrix4d::Identity(), row3(0.3, 0.4, 3.0));
    REQUIRE(pz.first.rows() == 1);
    REQUIRE(pz.second.size() == 1);
    CHECK_MAT_NEAR(pz.first, row2(790, 712), 1e-9);
    CHECK_SCALAR_NEAR(pz.second(0), 3.0, 1e-12);
    pz = lab::project_points(K, lab::trans(Vector3d(-0.5, 0, 0)), row3(0.3, 0.4, 3.0));
    CHECK_MAT_NEAR(pz.first, row2(1040, 712), 1e-9);
    MatrixXd grid(9, 3);
    int r = 0;
    for (double x : {-0.1, 0.0, 0.1})
        for (double y : {-0.1, 0.0, 0.1}) grid.row(r++) << x, y, 1.0;
    pz = lab::project_points(K, Matrix4d::Identity(), grid);
    VectorXd u(9), v(9);
    u << 490, 490, 490, 640, 640, 640, 790, 790, 790;
    v << 362, 512, 662, 362, 512, 662, 362, 512, 662;
    CHECK_MAT_NEAR(pz.first.col(0), u, 1e-9);
    CHECK_MAT_NEAR(pz.first.col(1), v, 1e-9);
    const auto p0 = lab::project_points(K_I(), Matrix4d::Identity(), row3(1, 1, 5));
    const auto px = lab::project_points(K_I(), lab::trans(Vector3d(0.1, 0, 0)), row3(1, 1, 5));
    CHECK_MAT_NEAR(p0.first, row2(672, 672), 1e-9);
    CHECK_MAT_NEAR(px.first, row2(656, 672), 1e-9);
}

TEST_CASE("test_project_points_rejects_points_behind") {
    MatrixXd P(2, 3);
    P << 0, 0, 1.0, 0.2, 0.1, -0.5;
    CHECK_THROWS_AS(lab::project_points(K_I(), Matrix4d::Identity(), P), std::invalid_argument);
    CHECK_THROWS_AS(lab::project_points(K_I(), lab::trans(Vector3d(0, 0, 2.0)), row3(0.1, 0.1, 1.0)),
                    std::invalid_argument);
}

TEST_CASE("test_project_points_matches_camera_matrix") {
    for (const json& c : cases().at("project_points_matches_camera_matrix")) {
        const Matrix4d T = as<Matrix4d>(c.at("T"));
        const MatrixXd P = sbeg::as_mat(c.at("P"));
        const auto pz = lab::project_points(K_I(), T, P);
        const MatrixXd Pc = transform(mr::trans_inv(T), P);
        CHECK_MAT_NEAR(pz.first, pix(K_I(), Pc), 1e-9);
        CHECK_MAT_NEAR(pz.second, Pc.col(2), 1e-12);
        CHECK_MAT_NEAR(pz.first, sbeg::as_mat(c.at("p")), 1e-9);
    }
}

TEST_CASE("test_camera_matrix_dlt_exact_data") {
    const MatrixXd P = lab::load_calib().first;
    const Matrix4d T = SC().calib.T_wc;
    const lab::Matrix34d C_true = K_E() * mr::trans_inv(T).topRows<3>();
    CHECK_MAT_NEAR(C_true, sbeg::as_mat(cases().at("camera_matrix_dlt_exact_data").at("C_true")), 1e-9);
    const MatrixXd p = pix(K_E(), transform(mr::trans_inv(T), P));
    const lab::Matrix34d C = lab::camera_matrix_dlt(P, p);
    CHECK_MAT_NEAR(C, C_true, 1e-7 * C_true.cwiseAbs().maxCoeff());
    const auto KT = lab::decompose_camera_matrix(C);
    CHECK_MAT_NEAR(KT.first, K_E(), 1e-6);
    CHECK_MAT_NEAR(KT.second, mr::trans_inv(T), 1e-9);
    CHECK(lab::reprojection_rms(C, P, p) < 1e-6);
    CHECK_THROWS_AS(lab::camera_matrix_dlt(P.topRows(5), p.topRows(5)), std::invalid_argument);
}

TEST_CASE("test_camera_matrix_dlt_noisy_data_and_reprojection") {
    const auto Pp = lab::load_calib();
    const lab::Matrix34d C = lab::camera_matrix_dlt(Pp.first, Pp.second);
    const json& E = EXP().at("calib");
    const MatrixXd C_exp = sbeg::as_mat(E.at("C"));
    CHECK_MAT_NEAR(C, C_exp, 1e-8 * C_exp.cwiseAbs().maxCoeff());
    CHECK_MAT_NEAR(lab::decompose_camera_matrix(C).first, sbeg::as_mat(E.at("K")), 1e-6);
    CHECK_APPROX(lab::reprojection_rms(C, Pp.first, Pp.second), E.at("rms").get<double>());
    const std::vector<int> i6 = E.at("idx6").get<std::vector<int>>();
    MatrixXd P6(6, 3), p6(6, 2);
    for (int k = 0; k < 6; ++k) {
        P6.row(k) = Pp.first.row(i6[static_cast<std::size_t>(k)]);
        p6.row(k) = Pp.second.row(i6[static_cast<std::size_t>(k)]);
    }
    CHECK_APPROX(lab::reprojection_rms(lab::camera_matrix_dlt(P6, p6), Pp.first, Pp.second),
                 E.at("rms6_all").get<double>());
}

TEST_CASE("test_reprojection_rms_hand_value") {
    lab::Matrix34d C = lab::Matrix34d::Zero();
    C.leftCols<3>() = K_E();
    MatrixXd P(2, 3), p(2, 2);
    P << 0, 0, 1, 0.1, 0, 1;
    p << 323, 244, 380, 240;
    CHECK_APPROX(lab::reprojection_rms(C, P, p), std::sqrt(12.5));
    const lab::Matrix34d C2 = -2.0 * C;
    CHECK_APPROX(lab::reprojection_rms(C2, P, p), std::sqrt(12.5));
}

// ------------------------------------------------------------------------------ Part B
TEST_CASE("test_interaction_matrix_corke_values") {
    MatrixXd Lc(2, 6);
    Lc << 32, -832, 160, -160, 0, 32, 832, -32, -160, 0, -160, 32;
    const MatrixXd L = lab::interaction_matrix(row2(672, 672), VectorXd::Constant(1, 5.0), K_I());
    CHECK_MAT_NEAR(L, Lc, 1e-9);
    CHECK_MAT_NEAR(L, sbeg::as_mat(cases().at("interaction_matrix_corke_values").at("L")), 1e-9);
    MatrixXd p(2, 2);
    p << 672, 672, 512, 512;
    const MatrixXd L2 = lab::interaction_matrix(p, Eigen::Vector2d(5.0, 1.0), K_I());
    REQUIRE(L2.rows() == 4);
    MatrixXd Lb(2, 6);
    Lb << 0, -800, 0, -800, 0, 0, 800, 0, 0, 0, -800, 0;
    CHECK_MAT_NEAR(L2.bottomRows(2), Lb, 1e-9);
}

TEST_CASE("test_interaction_matrix_finite_differences") {
    for (const json& c : cases().at("interaction_matrix_finite_differences")) {
        const MatrixXd p = sbeg::as_mat(c.at("p"));
        const VectorXd Z = sbeg::as_vec(c.at("Z"));
        const VectorXd V = sbeg::as_vec(c.at("V"));
        CHECK_MAT_NEAR(lab::interaction_matrix(p, Z, K_I()) * V, sbeg::as_vec(c.at("fd")), 1e-4);
    }
}

TEST_CASE("test_interaction_matrix_rank_and_null_space") {
    const MatrixXd L = lab::interaction_matrix(row2(512, 512), VectorXd::Constant(1, 2.0), K_I());
    REQUIRE(L.rows() == 2);
    CHECK(Eigen::FullPivLU<MatrixXd>(L).rank() == 2);
    const double nulls[4][6] = {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 1, 0, -2.0, 0, 0}, {1, 0, 0, 0, 2.0, 0}};
    for (const auto& n : nulls) {
        Vector6d V;
        V << n[0], n[1], n[2], n[3], n[4], n[5];
        CHECK_MAT_NEAR(L * V, Vector2d::Zero(), 1e-9);
    }
    MatrixXd p3(3, 2), pc(3, 2);
    p3 << 400, 300, 700, 450, 480, 720;
    pc << 300, 300, 500, 500, 700, 700;
    Eigen::JacobiSVD<MatrixXd> s3(lab::interaction_matrix(p3, VectorXd::Constant(3, 2.0), K_I()));
    Eigen::JacobiSVD<MatrixXd> sc(lab::interaction_matrix(pc, VectorXd::Constant(3, 2.0), K_I()));
    CHECK(s3.singularValues()(5) > 1e-6 * s3.singularValues()(0));        // rank 6
    CHECK(sc.singularValues()(5) < 1e-6);                                   // collinear: singular
}

// ------------------------------------------------------------------------------ Part C
TEST_CASE("test_ibvs_velocity_law") {
    MatrixXd p3(3, 2), ps3(3, 2);
    p3 << 400, 300, 700, 450, 480, 720;
    ps3 << 420, 330, 690, 430, 500, 700;
    const VectorXd Z = Eigen::Vector3d(2.0, 2.5, 1.8);
    const Vector6d V = lab::ibvs_velocity(p3, ps3, Z, K_I(), 0.7);
    CHECK_MAT_NEAR(lab::interaction_matrix(p3, Z, K_I()) * V, -0.7 * stack(p3 - ps3), 1e-8);
    CHECK_MAT_NEAR(lab::ibvs_velocity(p3, p3, Z, K_I(), 0.7), Vector6d::Zero(), 1e-12);
    const Setup s = scene_setup();
    const auto pz = lab::project_points(K_I(), SC().start_poses.corke, s.P);
    const MatrixXd L = lab::interaction_matrix(pz.first, pz.second, K_I());
    CHECK_MAT_NEAR(lab::ibvs_velocity(pz.first, s.ps, pz.second, K_I(), 1.0), -mr::pinv(L) * stack(pz.first - s.ps),
                   1e-9);
}

TEST_CASE("test_camera_step_matches_expm") {
    const Matrix4d T = SC().start_poses.corke;
    Vector6d V;
    V << 0.3, -0.2, 0.5, 1.0, -0.4, 0.8;
    const Vector6d Vh = V * 0.1;
    CHECK_MAT_NEAR(lab::camera_step(T, V, 0.1), T * mr::matrix_exp6(mr::vec_to_se3(Vh)), 1e-12);
    CHECK_MAT_NEAR(lab::camera_step(T, Vector6d::Zero(), 0.1), T, 1e-15);
}

TEST_CASE("test_run_ibvs_corke_example") {
    const Setup s = scene_setup();
    const Matrix4d T0 = SC().start_poses.corke;
    const lab::ServoRun r = lab::run_ibvs(T0, s.P, s.ps, K_I(), SC().ibvs.lam, SC().ibvs.h, std::nullopt,
                                          SC().ibvs.n_max, SC().ibvs.tol);
    const json& E = EXP().at("ibvs_depth").at(0);
    const int n = r.n_iter;
    CHECK(r.converged);
    CHECK(std::abs(n - E.at("n_iter").get<int>()) <= 1);
    REQUIRE(r.T.size() == static_cast<std::size_t>(n + 1));
    REQUIRE(r.p.size() == static_cast<std::size_t>(n + 1));
    REQUIRE(r.err.size() == n + 1);
    CHECK(r.Z.rows() == n + 1);
    CHECK(r.V.rows() == n);
    REQUIRE(r.cond.size() == n);
    REQUIRE(n >= 1);
    CHECK(r.err(n) <= SC().ibvs.tol);
    CHECK(r.err(n - 1) > SC().ibvs.tol);
    CHECK_MAT_NEAR(r.T.front(), T0, 1e-15);
    CHECK((r.T.back().block<3, 1>(0, 3) - s.Tg.block<3, 1>(0, 3)).norm() < 3e-3);
    const Matrix3d R_end = r.T.back().topLeftCorner(3, 3);
    CHECK_MAT_NEAR(R_end, Matrix3d::Identity(), 3e-3);
    const lab::ServoMetrics m = lab::servo_metrics(r, SC().inspection_camera);
    CHECK_SCALAR_NEAR(m.straightness, E.at("straightness").get<double>(), rel_tol(E.at("straightness"), 1e-4));
    CHECK_SCALAR_NEAR(r.cond(0), E.at("cond0").get<double>(), rel_tol(E.at("cond0"), 1e-6));
    write_run_csv("lab11_cpp_ibvs.csv", r);
}

TEST_CASE("test_run_ibvs_depth_estimates") {
    const Setup s = scene_setup();
    const json& runs = EXP().at("ibvs_depth");
    for (std::size_t i = 1; i < runs.size(); ++i) {
        const json& E = runs.at(i);
        const double Zh = E.at("Z_hat").get<double>();
        INFO("Z_hat = " << Zh);
        const lab::ServoRun r = lab::run_ibvs(SC().start_poses.corke, s.P, s.ps, K_I(), SC().ibvs.lam, SC().ibvs.h,
                                              VectorXd::Constant(4, Zh), SC().ibvs.n_max, SC().ibvs.tol);
        CHECK(r.converged);
        CHECK(std::abs(r.n_iter - E.at("n_iter").get<int>()) <= 1);
        REQUIRE(r.T.size() == static_cast<std::size_t>(r.n_iter + 1));
        REQUIRE(r.p.size() == r.T.size());
        REQUIRE(r.Z.rows() == r.n_iter + 1);
        CHECK_SCALAR_NEAR(lab::servo_metrics(r, SC().inspection_camera).straightness,
                          E.at("straightness").get<double>(), rel_tol(E.at("straightness"), 1e-3));
    }
}

TEST_CASE("test_run_ibvs_stops_and_model_errors") {
    const Setup s = scene_setup();
    const Matrix4d T0 = SC().start_poses.corke;
    const lab::ServoRun r = lab::run_ibvs(T0, s.P, s.ps, K_I(), SC().ibvs.lam, SC().ibvs.h, std::nullopt, 5,
                                          SC().ibvs.tol);
    CHECK(!r.converged);
    CHECK(r.n_iter == 5);
    CHECK(r.T.size() == 6);
    CHECK(r.V.rows() == 5);
    const lab::ServoRun r0 = lab::run_ibvs(s.Tg, s.P, s.ps, K_I(), SC().ibvs.lam, SC().ibvs.h, std::nullopt, 5,
                                           SC().ibvs.tol);
    CHECK(r0.converged);
    CHECK(r0.n_iter == 0);
    CHECK(r0.V.rows() == 0);
    const Matrix3d Kb = as<Matrix3d>(EXP().at("k_bad").at("K_hat"));
    const lab::ServoRun rb = lab::run_ibvs(T0, s.P, s.ps, K_I(), SC().ibvs.lam, SC().ibvs.h, std::nullopt,
                                           SC().ibvs.n_max, SC().ibvs.tol, Kb);
    CHECK(rb.converged);
    REQUIRE(!rb.T.empty());
    CHECK((rb.T.back().block<3, 1>(0, 3) - s.Tg.block<3, 1>(0, 3)).norm() < 3e-3);
}

// ------------------------------------------------------------------------------ Part D
TEST_CASE("test_homography_dlt_exact") {
    const Setup s = scene_setup();
    const Matrix4d T_ct = mr::trans_inv(SC().start_poses.corke) * SC().T_wt;
    MatrixXd Pt(4, 3);
    Pt << s.xy, MatrixXd::Zero(4, 1);
    const MatrixXd p = pix(K_I(), transform(T_ct, Pt));
    const Matrix3d H = lab::homography_dlt(s.xy, p);
    Matrix3d B;
    B << T_ct.block<3, 1>(0, 0), T_ct.block<3, 1>(0, 1), T_ct.block<3, 1>(0, 3);
    const Matrix3d H_true = K_I() * B;
    const Matrix3d Hn = H_true / H_true(2, 2);
    CHECK_MAT_NEAR(H, Hn, 1e-8 * (1.0 + Hn.cwiseAbs().maxCoeff()));
    CHECK_THROWS_AS(lab::homography_dlt(s.xy.topRows(3), p.topRows(3)), std::invalid_argument);
}

TEST_CASE("test_pose_from_homography_recovers_pose") {
    const json& c = cases().at("pose_from_homography_recovers_pose");
    const MatrixXd xy = sbeg::as_mat(c.at("xy"));
    for (const json& k : c.at("cases")) {
        const Matrix4d T_ct = as<Matrix4d>(k.at("T_ct"));
        const MatrixXd p = sbeg::as_mat(k.at("p"));
        CHECK_MAT_NEAR(lab::pose_from_homography(lab::homography_dlt(xy, p), K_I()), T_ct, 1e-8);
    }
    Matrix3d B;
    B << 1, 0, 0.1, 0, 1, 0.2, 0, 0, 2.0;
    const Matrix3d H = -3.0 * (K_I() * B);
    CHECK_MAT_NEAR(lab::pose_from_homography(H, K_I()), lab::trans(Vector3d(0.1, 0.2, 2.0)), 1e-12);
}

TEST_CASE("test_pbvs_velocity_one_unit_step") {
    const Setup s = scene_setup();
    const Matrix4d Ts[2] = {SC().start_poses.corke, s.Tg * lab::rotz(2.0) * lab::trans(Vector3d(0.2, -0.1, 0.3))};
    for (const Matrix4d& T : Ts) {
        const Matrix4d T_ct = mr::trans_inv(T) * SC().T_wt;
        const Vector6d V = lab::pbvs_velocity(T_ct, s.T_cst, 0.5);
        const Vector6d V2 = V * 2.0;
        CHECK_MAT_NEAR(T * mr::matrix_exp6(mr::vec_to_se3(V2)), s.Tg, 1e-9);
    }
    CHECK_MAT_NEAR(lab::pbvs_velocity(s.T_cst, s.T_cst, 0.5), Vector6d::Zero(), 1e-12);
}

TEST_CASE("test_run_pbvs_and_field_of_view") {
    const Setup s = scene_setup();
    const lab::Scene::Ibvs& ib = SC().ibvs;
    const lab::ServoRun r = lab::run_pbvs(SC().start_poses.corke, s.P, s.xy, s.T_cst, s.ps, K_I(), ib.lam, ib.h,
                                          ib.n_max, ib.tol);
    const int n = r.n_iter;
    CHECK(r.converged);
    CHECK(std::abs(n - EXP().at("pbvs_corke").at("n_iter").get<int>()) <= 1);
    REQUIRE(r.T.size() == static_cast<std::size_t>(n + 1));
    REQUIRE(r.p.size() == r.T.size());
    REQUIRE(r.Z.rows() == n + 1);
    CHECK(r.V.rows() == n);
    REQUIRE(r.err.size() == n + 1);
    CHECK(r.err(n) <= ib.tol);
    write_run_csv("lab11_cpp_pbvs.csv", r);
    const Matrix4d Tf = SC().start_poses.fov;
    const lab::ServoMetrics mi =
        lab::servo_metrics(lab::run_ibvs(Tf, s.P, s.ps, K_I(), ib.lam, ib.h, std::nullopt, ib.n_max, ib.tol),
                           SC().inspection_camera);
    const lab::ServoMetrics mp = lab::servo_metrics(
        lab::run_pbvs(Tf, s.P, s.xy, s.T_cst, s.ps, K_I(), ib.lam, ib.h, ib.n_max, ib.tol), SC().inspection_camera);
    const double ei = EXP().at("fov").at("ibvs").at("min_margin").get<double>();
    const double ep = EXP().at("fov").at("pbvs").at("min_margin").get<double>();
    CHECK_SCALAR_NEAR(mi.min_margin, ei, rel_tol(ei, 1e-4));
    CHECK_SCALAR_NEAR(mp.min_margin, ep, rel_tol(ep, 1e-4));
    CHECK(mi.min_margin > 0);
    CHECK(mp.min_margin < 0);
}

TEST_CASE("test_pbvs_calibration_error") {
    const Setup s = scene_setup();
    const lab::Scene::Ibvs& ib = SC().ibvs;
    const json& E = EXP().at("k_bad").at("pbvs");
    const lab::ServoRun r = lab::run_pbvs(SC().start_poses.corke, s.P, s.xy, s.T_cst, s.ps, K_I(), ib.lam, ib.h,
                                          ib.n_max, ib.tol, as<Matrix3d>(EXP().at("k_bad").at("K_hat")));
    CHECK(!r.converged);
    CHECK(r.n_iter == ib.n_max);
    REQUIRE(!r.T.empty());
    REQUIRE(r.err.size() >= 1);
    CHECK_APPROX(r.err(r.err.size() - 1), E.at("final_err").get<double>());
    const Vector3d p_end = r.T.back().col(3).head(3);
    CHECK_MAT_NEAR(p_end, sbeg::as_vec(E.at("final_pos")), 1e-6);
}

// ------------------------------------------------------------------------------ Part E
TEST_CASE("test_rcm_twist_basis_keeps_port_still") {
    for (const json& c : cases().at("rcm_twist_basis_keeps_port_still")) {
        const double ell = c.at("ell").get<double>();
        const Vector4d xi = as<Vector4d>(c.at("xi"));
        const lab::Matrix64d W = lab::rcm_twist_basis(ell);
        CHECK(Eigen::FullPivLU<MatrixXd>(W).rank() == 4);
        const Vector6d V = W * xi;
        const Vector3d r(0.0, 0.0, -ell);
        const Vector3d v_port = V.tail<3>() + V.head<3>().cross(r);  // velocity of the scope point in the port
        CHECK_MAT_NEAR(v_port, Vector3d(0.0, 0.0, xi(3)), 1e-12);    // along the scope axis only
        CHECK_MAT_NEAR(V.head<3>(), xi.head<3>(), 1e-15);
        CHECK_APPROX(V(5), xi(3));
    }
}

TEST_CASE("test_rcm_ibvs_velocity_law") {
    const MatrixXd p = row2(430, 180), ps = row2(320, 240);
    const json& ref = cases().at("rcm_ibvs_velocity_law");
    int i = 0;
    for (double d_dot : {0.0, 0.01}) {
        const auto xv = lab::rcm_ibvs_velocity(p, ps, VectorXd::Constant(1, 0.08), K_E(), 2.0, 0.1, d_dot);
        CHECK(xv.first(2) == 0.0);
        CHECK_APPROX(xv.first(3), d_dot);
        CHECK_MAT_NEAR(xv.second, lab::rcm_twist_basis(0.1) * xv.first, 1e-15);
        CHECK_MAT_NEAR(lab::interaction_matrix(p, 0.08, K_E()) * xv.second, -2.0 * stack(p - ps), 1e-8);
        CHECK_MAT_NEAR(xv.second, sbeg::as_vec(ref.at(static_cast<std::size_t>(i)).at("V")), 1e-12);
        ++i;
    }
    const auto x0 = lab::rcm_ibvs_velocity(ps, ps, VectorXd::Constant(1, 0.08), K_E(), 2.0, 0.1);
    CHECK_MAT_NEAR(x0.second, Vector6d::Zero(), 1e-15);
}

TEST_CASE("test_joint_rates") {
    const cm::Robot rob = cm::ur5_laparoscope();
    for (const json& c : cases().at("joint_rates")) {
        const VectorXd th = sbeg::as_vec(c.at("theta"));
        const Vector6d V = as<Vector6d>(c.at("V"));
        const MatrixXd Jb = mr::jacobian_body(rob.Blist, th);
        const VectorXd qd = lab::joint_rates(rob.Blist, th, V);
        REQUIRE(qd.size() == 6);
        CHECK_MAT_NEAR(Jb * qd, V, 1e-10);
        CHECK_MAT_NEAR(qd, sbeg::as_vec(c.at("qd")), 1e-9);
        const Vector6d V50 = 50.0 * V;
        const VectorXd qs = lab::joint_rates(rob.Blist, th, V50, 0.5);
        REQUIRE(qs.size() == 6);
        CHECK_APPROX(qs.cwiseAbs().maxCoeff(), 0.5);
        CHECK_MAT_NEAR(qs * (50.0 * qd).cwiseAbs().maxCoeff() / 0.5, 50.0 * qd, 1e-9);
        CHECK_MAT_NEAR(lab::joint_rates(rob.Blist, th, V, 10.0), qd, 1e-12);
    }
}

TEST_CASE("test_rcm_errors") {
    const lab::EndoSetup st = lab::endoscope_setup(SC());
    auto re = lab::rcm_errors(st.T0, st.port);
    CHECK_APPROX(re.first, SC().endoscope.ell0);
    CHECK(re.second.norm() < 1e-12);
    const Matrix4d T = st.T0 * lab::trans(Vector3d(0.002, -0.001, 0.01));
    re = lab::rcm_errors(T, st.port);
    CHECK_APPROX(re.first, 0.11);
    CHECK_MAT_NEAR(re.second, Vector2d(-0.002, 0.001), 1e-12);
}

TEST_CASE("test_endoscope_closed_loop") {
    for (const std::string c : {"default", "delay_unstable"}) {
        INFO("case = " << c);
        lab::EndoOptions opt;
        json E;
        double rel = 1e-6;
        if (c == "default") {
            E = EXP().at("endoscope").at("default");
        } else {
            E = EXP().at("endoscope").at("delay").at(2);
            opt.lam = E.at("lam").get<double>();
            opt.delay = E.at("delay").get<int>();
            rel = 1e-3;
        }
        const lab::EndoSim sim = lab::simulate_endoscope(SC(), opt);
        const lab::EndoMetrics m = lab::endoscope_metrics(sim);
        CHECK_SCALAR_NEAR(m.rms_err, E.at("rms_err").get<double>(), rel_tol(E.at("rms_err"), rel));
        CHECK_SCALAR_NEAR(m.max_err, E.at("max_err").get<double>(), rel_tol(E.at("max_err"), rel));
        CHECK_SCALAR_NEAR(m.max_e_lat, E.at("max_e_lat").get<double>(), rel_tol(E.at("max_e_lat"), rel));
        CHECK_SCALAR_NEAR(m.ell_min, E.at("ell_min").get<double>(), rel_tol(E.at("ell_min"), rel));
        CHECK_SCALAR_NEAR(m.peak_qdot, E.at("peak_qdot").get<double>(), rel_tol(E.at("peak_qdot"), rel));
        if (c == "default") {
            CHECK(m.final_still_err < 1.0);
            CHECK(m.view_frac == 1.0);
            CHECK(m.sat_frac == 0.0);
            MatrixXd D(sim.t.size(), 6);
            D << sim.t, sim.p, sim.err, sim.ell, sim.e_lat;
            sbeg::write_csv("lab11_cpp_endoscope.csv", {"t", "u", "v", "err", "ell", "e_lat"}, D);
        } else {
            CHECK(m.sat_frac > 0.3);
        }
    }
}

TEST_CASE("test_endoscope_rcm_drift_correction") {
    const lab::EndoMetrics m0 = lab::endoscope_metrics(lab::simulate_endoscope(SC()));
    lab::EndoOptions opt;
    opt.k_rcm = 5.0;
    const lab::EndoMetrics m1 = lab::endoscope_metrics(lab::simulate_endoscope(SC(), opt));
    CHECK(m1.max_e_lat < 0.2 * m0.max_e_lat);
    CHECK(m0.max_e_lat < 1e-3);
    CHECK_SCALAR_NEAR(m1.rms_err, m0.rms_err, 0.02 * m0.rms_err);
    const double e = EXP().at("endoscope").at("k_rcm").at("max_e_lat").get<double>();
    CHECK_SCALAR_NEAR(m1.max_e_lat, e, rel_tol(e, 1e-6));
}
