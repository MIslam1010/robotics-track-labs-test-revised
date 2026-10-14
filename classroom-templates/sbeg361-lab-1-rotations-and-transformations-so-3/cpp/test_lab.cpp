// Auto-checks for SBEG 361 Lab 1 (rotations and transformations: SO(3), SE(3), transform trees) -- C++ track.
// Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*elementary*,*axis_angle*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics), the worked examples of MR Chapter 3 (Fig. 3.7, Fig. 3.14,
// Fig. 3.15, Example 3.19) and cpp_cases.json, which holds the numbers of the Python reference (and of
// scipy.spatial.transform), so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using sbeg::as;
using sbeg::as_list;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
const double S2 = 1.0 / std::sqrt(2.0);
const std::set<int> CORRUPTED = {4, 9, 13, 17};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// RANDOM_T of test_lab.py (20 random poses, stored in cpp_cases.json).
const std::vector<Matrix4d>& random_T() {
    static const std::vector<Matrix4d> Ts = as_list<Matrix4d>(cases()["random_T"]);
    return Ts;
}

/// Reference rotation from mrcpp: exp([w_hat] theta).
Matrix3d mr_rot(const Vector3d& axis, double theta) { return mr::matrix_exp3(mr::vec_to_so3(axis * theta)); }

Matrix4d mat4(double a00, double a01, double a02, double a03, double a10, double a11, double a12, double a13,
              double a20, double a21, double a22, double a23) {
    Matrix4d T;
    T << a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, 0.0, 0.0, 0.0, 1.0;
    return T;
}

Matrix3d mat3(double a, double b, double c, double d, double e, double f, double g, double h, double i) {
    Matrix3d R;
    R << a, b, c, d, e, f, g, h, i;
    return R;
}

Matrix4d rp(const Matrix3d& R, const Vector3d& p) { return mr::rp_to_trans(R, p); }

// MR Fig. 3.14 (printed p. 90)
const Matrix4d T_SB_314 = rp(mat3(0, 0, 1, 0, -1, 0, 1, 0, 0), Vector3d(0, -2, 0));
const Matrix4d T_SC_314 = rp(mat3(-1, 0, 0, 0, 0, 1, 0, 1, 0), Vector3d(-1, 1, 0));
const Matrix4d T_BC_314 = rp(mat3(0, 1, 0, 0, 0, -1, -1, 0, 0), Vector3d(0, -3, -1));

// MR Example 3.19 (printed pp. 93-95)
const Matrix4d T_DB = mat4(0, 0, -1, 250, 0, -1, 0, -150, -1, 0, 0, 200);
const Matrix4d T_DE = mat4(0, 0, -1, 300, 0, -1, 0, 100, -1, 0, 0, 120);
const Matrix4d T_AD = mat4(0, 0, -1, 400, 0, -1, 0, 50, -1, 0, 0, 300);
const Matrix4d T_BC = mat4(0, -S2, -S2, 30, 0, S2, -S2, -40, 1, 0, 0, 25);
const Matrix4d T_CE_MR = mat4(0, 0, 1, -75, -S2, S2, 0, -260 * S2, -S2, -S2, 0, 160 * S2);

/// np.isclose(a, b, rtol, atol): |a - b| <= atol + rtol |b|.
double isclose_tol(double b, double rtol = 1e-5, double atol = 1e-8) { return atol + rtol * std::abs(b); }

/// The tracker samples, read with the given helper lab::load_tracker_csv.
std::vector<Matrix3d> tracker_samples() { return lab::load_tracker_csv(sbeg::lab_path("../data/tracker_orientations.csv")); }

/// Built inside each test (not shared) so that the starter fails as a test, not as an error.
lab::TransformTree load_nav() { return lab::load_tree_json(sbeg::lab_path("../data/nav_tree.json")); }

/// The tracked screwdriver (given loader; read once).
const std::pair<Eigen::VectorXd, std::vector<Matrix4d>>& track() {
    static const std::pair<Eigen::VectorXd, std::vector<Matrix4d>> tr =
        lab::load_track_csv(sbeg::lab_path("../data/tool_track.csv"));
    return tr;
}
}  // namespace

// ---------------- Part A: building R and T ----------------
TEST_CASE("test_elementary_rotations_match_mr") {
    for (const auto& c : cases()["elementary_rotations"]) {
        const double theta = c["theta"].get<double>();
        INFO("theta = " << theta);
        CHECK_MAT_NEAR(lab::rot_x(theta), mr_rot(Vector3d::UnitX(), theta), 1e-12);
        CHECK_MAT_NEAR(lab::rot_y(theta), mr_rot(Vector3d::UnitY(), theta), 1e-12);
        CHECK_MAT_NEAR(lab::rot_z(theta), mr_rot(Vector3d::UnitZ(), theta), 1e-12);
        CHECK_MAT_NEAR(lab::rot_y(theta), as<Matrix3d>(c["rot_y"]), 1e-12);
        CHECK_MAT_NEAR(lab::rot_z(theta), as<Matrix3d>(c["rot_z"]), 1e-12);
    }
}

TEST_CASE("test_axis_angle_matches_mr") {
    for (const auto& c : cases()["axis_angle"]) {
        const Vector3d w = as<Vector3d>(c["omega_hat"]);
        const double th = c["theta"].get<double>();
        CHECK_MAT_NEAR(lab::rot_axis_angle(w, th), mr_rot(w, th), 1e-12);
        CHECK_MAT_NEAR(lab::rot_axis_angle(w, th), as<Matrix3d>(c["R"]), 1e-12);
    }
}

TEST_CASE("test_axis_angle_identities") {
    const Vector3d w = Vector3d(1.0, 2.0, 2.0) / 3.0;
    CHECK_MAT_NEAR(lab::rot_axis_angle(w, 0.7), lab::rot_axis_angle(-w, -0.7), 1e-12);  // MR p. 72
    CHECK_MAT_NEAR(lab::rot_axis_angle(w, 0.0), Matrix3d::Identity(), 1e-12);
    CHECK_MAT_NEAR(lab::rot_axis_angle(w, 0.9) * w, w, 1e-12);  // axis is unchanged
}

TEST_CASE("test_axis_angle_rejects_non_unit") {
    CHECK_THROWS_AS(lab::rot_axis_angle(Vector3d(1.0, 1.0, 0.0), 0.5), std::invalid_argument);
}

TEST_CASE("test_rp_to_trans_and_back") {
    for (int i = 0; i < 5; ++i) {
        const Matrix4d& T = random_T()[static_cast<std::size_t>(i)];
        INFO("RANDOM_T[" << i << "]");
        const auto [R, p] = mr::trans_to_rp(T);
        CHECK_MAT_NEAR(lab::rp_to_trans(R, p), T, 1e-12);
        const auto [R2, p2] = lab::trans_to_rp(T);
        CHECK_MAT_NEAR(R2, R, 1e-12);
        CHECK_MAT_NEAR(p2, p, 1e-12);
    }
}

// ---------------- Part B: group property tests, SO(3) and SE(3) ----------------
TEST_CASE("test_so3_errors_values") {
    auto [e, d] = lab::so3_errors(Matrix3d::Identity());
    CHECK_APPROX(e, 0.0);  // Python: e == pytest.approx(0.0)
    CHECK_APPROX(d, 1.0);
    std::tie(e, d) = lab::so3_errors(2.0 * Matrix3d::Identity());
    CHECK_APPROX(e, std::sqrt(27.0));
    CHECK_APPROX(d, 8.0);
    CHECK_THROWS_AS(lab::so3_errors(Matrix2d::Identity()), std::invalid_argument);
    for (const auto& c : cases()["so3_errors"]) {
        const auto [e2, d2] = lab::so3_errors(as<Matrix3d>(c["R"]));
        CHECK_SCALAR_NEAR(e2, c["orth_err"].get<double>(), 1e-12);
        CHECK_SCALAR_NEAR(d2, c["det"].get<double>(), 1e-12);
    }
}

TEST_CASE("test_is_rotation_cases") {
    const Matrix3d Rc = mat3(0, -1, 0, 0, 0, -1, 1, 0, 0);    // MR Fig. 3.7
    const Matrix3d swap = mat3(0, 1, 0, 1, 0, 0, 0, 0, 1);    // det = -1
    const Matrix3d mr_example = mat3(1.0, 0.0, 0.0, 0.0, 0.1, -0.95, 0.0, 1.0, 0.1);  // MR library docstring
    CHECK(lab::is_rotation(Rc));
    CHECK(mr::test_if_so3(Rc));
    CHECK_FALSE(lab::is_rotation(swap));
    CHECK_FALSE(mr::test_if_so3(swap));
    CHECK_FALSE(lab::is_rotation(1.05 * Rc));
    CHECK_FALSE(lab::is_rotation(Matrix2d::Identity()));
    CHECK_FALSE(lab::is_rotation(mr_example));
    CHECK_FALSE(mr::test_if_so3(mr_example));
}

TEST_CASE("test_group_properties") {
    // Closure, inverse = transpose, non-commutativity, length preservation (MR Props. 3.3-3.6).
    for (const auto& c : cases()["group_properties"]) {
        const Matrix3d A = as<Matrix3d>(c["A"]), B = as<Matrix3d>(c["B"]);
        const Vector3d x = as<Vector3d>(c["x"]);
        CHECK(lab::is_rotation(A * B));
        CHECK_MAT_NEAR(lab::rot_inv(A), mr::rot_inv(A), 1e-12);
        CHECK_MAT_NEAR(lab::rot_inv(A), as<Matrix3d>(c["rot_inv_A"]), 1e-12);
        CHECK_MAT_NEAR(lab::rot_inv(A) * A, Matrix3d::Identity(), 1e-12);
        CHECK_APPROX((A * x).norm(), x.norm());
    }
    const Matrix3d X = lab::rot_x(PI / 2), Y = lab::rot_y(PI / 2);
    CHECK_FALSE(sbeg::compare_near(X * Y, Y * X, 1e-8).ok);  // rotations about different axes do not commute
    const Matrix3d Z1 = lab::rot_z(0.4), Z2 = lab::rot_z(1.1);
    CHECK_MAT_NEAR(Z1 * Z2, Z2 * Z1, 1e-12);  // same-axis (planar) rotations commute
}

TEST_CASE("test_project_to_so3") {
    const Matrix3d M = mat3(0.675, 0.150, 0.720, 0.370, 0.771, -0.511, -0.630, 0.619, 0.472);
    CHECK_MAT_NEAR(lab::project_to_so3(M), mr::project_to_so3(M), 1e-8);
    const Matrix3d reflect = Eigen::Vector3d(1.0, 1.0, -1.0).asDiagonal();
    const Matrix3d R = lab::project_to_so3(reflect);
    CHECK(lab::is_rotation(R));
    for (const auto& c : cases()["project_to_so3"])
        CHECK_MAT_NEAR(lab::project_to_so3(as<Matrix3d>(c["M"])), as<Matrix3d>(c["R"]), 1e-9);
}

TEST_CASE("test_is_se3") {
    // The SE(3) twin of test_is_rotation_cases, plus closure (MR Prop. 3.16).
    CHECK(lab::is_se3(random_T()[0]));
    const Matrix4d T01 = random_T()[0] * random_T()[1];
    CHECK(lab::is_se3(T01));  // closure
    Matrix4d bad = random_T()[0];
    bad.topLeftCorner<3, 3>() *= -1.0;  // det R = -1 (a reflection)
    CHECK_FALSE(lab::is_se3(bad));
    Matrix4d bad2 = random_T()[0];
    bad2(3, 0) = 0.1;  // wrong last row
    CHECK_FALSE(lab::is_se3(bad2));
    CHECK_FALSE(lab::is_se3(Matrix3d::Identity()));
    for (const auto& c : cases()["is_se3"]["cases"]) CHECK(lab::is_se3(as<Matrix4d>(c["T"])) == c["expected"].get<bool>());
}

TEST_CASE("test_trans_inv_matches_mr") {
    // The SE(3) twin of rot_inv: T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
    const json& c = cases()["trans_inv_matches_mr"];
    REQUIRE(c.size() == random_T().size());
    for (std::size_t i = 0; i < random_T().size(); ++i) {
        const Matrix4d& T = random_T()[i];
        INFO("RANDOM_T[" << i << "]");
        CHECK_MAT_NEAR(lab::trans_inv(T), mr::trans_inv(T), 1e-12);
        CHECK_MAT_NEAR(lab::trans_inv(T) * T, Matrix4d::Identity(), 1e-12);
        CHECK_MAT_NEAR(lab::trans_inv(T), as<Matrix4d>(c[i]["T_inv"]), 1e-12);
    }
}

// ---------------- Part C: the three uses of R and T ----------------
TEST_CASE("test_mr_figure_3_7") {
    const Matrix3d Rb = mat3(0, -1, 0, 1, 0, 0, 0, 0, 1);
    const Matrix3d Rc = mat3(0, -1, 0, 0, 0, -1, 1, 0, 0);
    const Vector3d pa(1.0, 1.0, 0.0);
    CHECK_MAT_NEAR(lab::change_frame(lab::rot_inv(Rb), pa), Vector3d(1, -1, 0), 1e-12);
    CHECK_MAT_NEAR(lab::change_frame(lab::rot_inv(Rc), pa), Vector3d(0, -1, -1), 1e-12);
    CHECK_MAT_NEAR(Rb, lab::rot_z(PI / 2), 1e-12);
    CHECK_MAT_NEAR(lab::rotate_in_body(Rb, lab::rot_y(-PI / 2)), Rc, 1e-12);
    CHECK_MAT_NEAR(lab::rotate_in_fixed(Rb, lab::rot_x(PI / 2)), Rc, 1e-12);
}

TEST_CASE("test_eod_base_pan_and_wrist") {
    // Defence checkpoint: EOD robot gripper, base yaw (fixed z_s) vs wrist roll (body x_b).
    const Matrix3d R_sb = lab::rot_z(PI / 6) * lab::rot_y(PI / 4);
    const Matrix3d pan = lab::rot_z(PI / 2);
    const Matrix3d roll = lab::rot_x(PI / 3);
    CHECK_MAT_NEAR(lab::rotate_in_fixed(R_sb, pan), pan * R_sb, 1e-12);
    CHECK_MAT_NEAR(lab::rotate_in_body(R_sb, roll), R_sb * roll, 1e-12);
    // roll about the gripper's own x axis leaves that axis (first column) unchanged
    CHECK_MAT_NEAR(lab::rotate_in_body(R_sb, roll).col(0), R_sb.col(0), 1e-12);
    const json& c = cases()["eod_base_pan_and_wrist"];
    CHECK_MAT_NEAR(lab::rotate_in_fixed(R_sb, pan), as<Matrix3d>(c["fixed"]), 1e-12);
    CHECK_MAT_NEAR(lab::rotate_in_body(R_sb, roll), as<Matrix3d>(c["body"]), 1e-12);
}

TEST_CASE("test_mr_fig_3_14_frames_and_point") {
    // T_bc = T_sb^-1 T_sc (MR p. 90) and v_b = (0, 0, 1.5) -> v_s.
    const Matrix4d T_bc = lab::trans_inv(T_SB_314) * T_SC_314;
    CHECK_MAT_NEAR(T_bc, T_BC_314, 1e-12);
    CHECK_MAT_NEAR(lab::transform_point(T_SB_314, Vector3d(0, 0, 1.5)), Vector3d(1.5, -2, 0), 1e-12);
    const json& c = cases()["mr_fig_3_14_frames_and_point"];
    const MatrixXd pts = sbeg::as_mat(c["pts"]);  // 7 x 3, one point per row
    MatrixXd hom(4, pts.rows());
    hom.topRows(3) = pts.transpose();
    hom.row(3).setOnes();
    const MatrixXd expected = (T_SB_314 * hom).topRows(3).transpose();
    CHECK_MAT_NEAR(lab::transform_points(T_SB_314, pts), expected, 1e-12);
    CHECK_MAT_NEAR(lab::transform_points(T_SB_314, pts), sbeg::as_mat(c["expected"]), 1e-12);
}

TEST_CASE("test_mr_fig_3_15_fixed_vs_body_frame") {
    // Pre-multiplication = fixed-frame displacement, post-multiplication = body-frame (MR Eqs. 3.66-3.67).
    const Matrix4d T = lab::rp_to_trans(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, PI / 2))), Vector3d(0, 2, 0));
    const Matrix4d T_sb = mat4(0, 0, 1, 0, 0, -1, 0, -2, 1, 0, 0, 0);
    const Matrix4d fixed = lab::transform_in_fixed(T_sb, T), body = lab::transform_in_body(T_sb, T);
    CHECK_MAT_NEAR(fixed, mat4(0, 1, 0, 2, 0, 0, 1, 2, 1, 0, 0, 0), 1e-12);
    CHECK_MAT_NEAR(body, mat4(0, 0, 1, 0, -1, 0, 0, -4, 0, -1, 0, 0), 1e-12);
    // the rotation block follows the SO(3) rule of MR Eqs. 3.23-3.24
    const Matrix3d R = T.topLeftCorner<3, 3>(), R_sb = T_sb.topLeftCorner<3, 3>();
    const Matrix3d R_fixed = fixed.topLeftCorner<3, 3>(), R_body = body.topLeftCorner<3, 3>();
    CHECK_MAT_NEAR(R_fixed, lab::rotate_in_fixed(R_sb, R), 1e-12);
    CHECK_MAT_NEAR(R_body, lab::rotate_in_body(R_sb, R), 1e-12);
}

TEST_CASE("test_distance_preserved") {
    // MR Prop. 3.18(a): ||Tx - Ty|| = ||x - y||.
    const json& c = cases()["distance_preserved"];
    const Vector3d x = as<Vector3d>(c["x"]), y = as<Vector3d>(c["y"]);
    const double d = (x - y).norm();
    for (int i = 0; i < 5; ++i) {
        const Matrix4d& T = random_T()[static_cast<std::size_t>(i)];
        const Vector3d Tx = lab::transform_point(T, x), Ty = lab::transform_point(T, y);
        CHECK_SCALAR_NEAR((Tx - Ty).norm(), d, isclose_tol(d));
    }
}

TEST_CASE("test_planar_compose") {
    const Matrix2d P = lab::rot_z(PI / 3).topLeftCorner<2, 2>();
    const Matrix2d Q = lab::rot_z(PI / 6).topLeftCorner<2, 2>();
    const auto [R, r] = lab::planar_compose(P, Vector2d(2.0, 1.0), Q, Vector2d(1.0, 0.5));
    Matrix2d R_exp;
    R_exp << 0, -1, 1, 0;
    CHECK_MAT_NEAR(R, R_exp, 1e-12);
    CHECK_MAT_NEAR(r, Vector2d(2.0 + 0.5 - std::sqrt(3.0) / 4, 1.0 + std::sqrt(3.0) / 2 + 0.25), 1e-12);
    const json& c = cases()["planar_compose"];
    CHECK_MAT_NEAR(r, as<Vector2d>(c["r"]), 1e-12);
    // the same composition as a product of 3x3 SE(2) matrices (MR Eq. 3.63)
    const Matrix3d T_sb = lab::planar_to_se2(P, Vector2d(2.0, 1.0)), T_bc = lab::planar_to_se2(Q, Vector2d(1.0, 0.5));
    CHECK_MAT_NEAR(T_sb.row(2), Eigen::RowVector3d(0, 0, 1), 1e-12);
    const Matrix3d T_sc = T_sb * T_bc;
    CHECK_MAT_NEAR(T_sc, lab::planar_to_se2(R, r), 1e-12);
    CHECK_MAT_NEAR(T_sc, as<Matrix3d>(c["T_sc"]), 1e-12);
}

// ---------------- Part D: the transform tree ----------------
TEST_CASE("test_tree_mr_example_3_19") {
    lab::TransformTree tree;
    tree.add("a", "d", T_AD);
    tree.add("d", "b", T_DB);
    tree.add("d", "e", T_DE);
    tree.add("b", "c", T_BC);
    CHECK_MAT_NEAR(tree.lookup("c", "e"), T_CE_MR, 1e-9);
    const Matrix4d T_ae = T_AD * T_DE, T_ac = T_AD * T_DB * T_BC;
    CHECK_MAT_NEAR(tree.lookup("a", "e"), T_ae, 1e-9);
    CHECK_MAT_NEAR(tree.lookup("a", "c"), T_ac, 1e-9);
    CHECK_MAT_NEAR(tree.lookup("e", "c"), mr::trans_inv(T_CE_MR), 1e-9);
    CHECK_MAT_NEAR(tree.lookup("b", "b"), Matrix4d::Identity(), 1e-12);
    CHECK(tree.path("c", "e") == std::vector<std::string>{"c", "b", "d", "e"});
    CHECK_MAT_NEAR(tree.lookup("c", "e"), as<Matrix4d>(cases()["tree_mr_example_3_19"]["T_ce"]), 1e-9);
}

TEST_CASE("test_tree_rejects_bad_edges") {
    const std::vector<Matrix4d>& R = random_T();
    lab::TransformTree tree;
    tree.add("world", "a", R[0]);
    tree.add("a", "b", R[1]);
    tree.add("a", "b", R[2]);  // update is allowed
    CHECK_MAT_NEAR(tree.lookup("a", "b"), R[2], 1e-12);
    CHECK_THROWS_AS(tree.add("world", "b", R[3]), std::invalid_argument);   // second parent
    CHECK_THROWS_AS(tree.add("b", "world", R[3]), std::invalid_argument);   // cycle
    const Matrix4d twice = 2.0 * R[3];
    CHECK_THROWS_AS(tree.add("b", "c", twice), std::invalid_argument);      // not in SE(3)
    CHECK_THROWS_AS(tree.lookup("world", "nowhere"), lab::KeyError);
    tree.add("other_root", "z", R[4]);
    CHECK_THROWS_AS(tree.lookup("world", "z"), lab::LookupError);
}

TEST_CASE("test_quaternion_order_and_json_loader") {
    const lab::TransformTree nav = load_nav();
    for (const auto& c : cases()["quaternion_order_and_json_loader"]) {  // scipy Rotation.random(random_state=s)
        const Vector4d q = as<Vector4d>(c["q_xyzw"]);
        INFO("q_xyzw = " << q.transpose());
        CHECK_MAT_NEAR(lab::quat_xyzw_to_rot(q), as<Matrix3d>(c["R"]), 1e-12);
    }
    const std::set<std::string> frames = nav.frames();
    for (const char* f : {"theatre", "tracker", "patient_ref", "ct_image", "plan_entry", "tool_marker", "tool_tip"}) {
        INFO("frame " << f);
        CHECK(frames.count(f) == 1);
    }
}

TEST_CASE("test_eod_tree_package_in_gripper") {
    const lab::TransformTree eod = lab::load_tree_json(sbeg::lab_path("../data/eod_tree.json"));
    const Matrix4d T_gp = eod.lookup("gripper", "package");
    const Vector3d p_gp = T_gp.topRightCorner<3, 1>();
    const Matrix3d R_gp = T_gp.topLeftCorner<3, 3>();
    CHECK_MAT_NEAR(p_gp, Vector3d(0.1, -0.1, 0.5), 1e-9);
    CHECK_MAT_NEAR(R_gp, mat3(0, 0, -1, 0, 1, 0, 1, 0, 0), 1e-9);
    const Vector3d p_bp = eod.lookup("base_link", "package").topRightCorner<3, 1>();
    CHECK_MAT_NEAR(p_bp, Vector3d(1.4, -0.1, 0.3), 1e-9);
}

// ---------------- Part E: applications ----------------
TEST_CASE("test_tracker_classification") {
    const std::vector<Matrix3d> Rs = tracker_samples();
    REQUIRE(Rs.size() == 20);
    const std::vector<bool> ok = lab::classify_samples(Rs);
    REQUIRE(ok.size() == Rs.size());
    std::set<int> invalid;
    for (std::size_t k = 0; k < ok.size(); ++k)
        if (!ok[k]) invalid.insert(static_cast<int>(k));
    CHECK(invalid == CORRUPTED);
    for (std::size_t k = 0; k < Rs.size(); ++k) {
        INFO("sample " << k);
        CHECK(ok[k] == mr::test_if_so3(Rs[k]));
    }
    CHECK(ok == sbeg::as_bools(cases()["tracker"]["valid"]));
}

TEST_CASE("test_tracker_projection_and_tool") {
    const std::vector<Matrix3d> Rs = tracker_samples();
    const std::vector<bool> ok = lab::classify_samples(Rs);
    REQUIRE(ok.size() == Rs.size());
    std::vector<Matrix3d> valid;
    for (std::size_t k = 0; k < Rs.size(); ++k)
        if (ok[k]) valid.push_back(Rs[k]);
    const json& c = cases()["tracker"];
    const auto projected = as_list<Matrix3d>(c["projected"]);
    REQUIRE(valid.size() == projected.size());
    for (std::size_t k = 0; k < valid.size(); ++k) {
        const Matrix3d Rp = lab::project_to_so3(valid[k]);
        CHECK_MAT_NEAR(Rp, mr::project_to_so3(valid[k]), 1e-9);
        CHECK_MAT_NEAR(Rp, projected[k], 1e-9);
        CHECK(lab::so3_errors(Rp).first < 1e-12);
    }
    const Matrix3d R_mt = lab::rot_x(PI / 6);  // marker-to-tip calibration (illustrative)
    const std::vector<Matrix3d> R_st = lab::tool_orientations(valid, R_mt);
    REQUIRE(R_st.size() == valid.size());
    CHECK_MAT_NEAR(R_st[0], valid[0] * R_mt, 1e-12);
    const auto R_st_ref = as_list<Matrix3d>(c["R_st"]);
    for (std::size_t k = 0; k < R_st.size(); ++k) CHECK_MAT_NEAR(R_st[k], R_st_ref[k], 1e-12);
}

TEST_CASE("test_tool_tip_in_ct_at_start") {
    lab::TransformTree nav = load_nav();
    const std::vector<Matrix4d>& Ts = track().second;
    REQUIRE(Ts.size() > 1);
    nav.add("tracker", "tool_marker", Ts[0]);
    const Matrix4d T = nav.lookup("ct_image", "tool_tip");
    const Matrix4d direct = mr::trans_inv(nav.lookup("tracker", "ct_image")) * Ts[0] * nav.lookup("tool_marker", "tool_tip");
    CHECK_MAT_NEAR(T, direct, 1e-12);
    const Vector3d tip_mm = T.topRightCorner<3, 1>() * 1000.0;
    CHECK_MAT_NEAR(tip_mm, Vector3d(142.0, 84.035, 210.262), 1e-2);  // mm
    const Vector3d tip_plan = nav.lookup("plan_entry", "tool_tip").topRightCorner<3, 1>() * 1000.0;
    CHECK_SCALAR_NEAR(tip_plan.norm(), 1.0, isclose_tol(1.0, 1e-5, 1e-3));  // 1.0 mm entry error
    CHECK(nav.path("ct_image", "tool_tip") ==
          std::vector<std::string>{"ct_image", "patient_ref", "tracker", "tool_marker", "tool_tip"});
}

TEST_CASE("test_tip_depth_along_plan") {
    // Update the tracker -> tool_marker edge for every sample and follow the tip in the plan frame.
    lab::TransformTree nav = load_nav();
    const std::vector<Matrix4d>& Ts = track().second;
    REQUIRE(Ts.size() > 1);
    std::vector<Vector3d> tips;
    for (const Matrix4d& Tk : Ts) {
        nav.add("tracker", "tool_marker", Tk);
        tips.push_back(nav.lookup("plan_entry", "tool_tip").topRightCorner<3, 1>() * 1000.0);  // mm
    }
    REQUIRE(tips.size() == Ts.size());
    CHECK_MAT_NEAR(tips.back(), Vector3d(0.0, 0.826, 4.997), 1e-2);  // t = 8 s
    bool increasing = true;                                          // depth grows monotonically
    for (std::size_t k = 1; k < tips.size(); ++k) increasing = increasing && (tips[k](2) > tips[k - 1](2));
    CHECK(increasing);
    const double lat0 = std::hypot(tips.front()(0), tips.front()(1));
    const double lat1 = std::hypot(tips.back()(0), tips.back()(1));
    CHECK_SCALAR_NEAR(lat0, 1.0, isclose_tol(1.0, 1e-5, 1e-3));
    CHECK_SCALAR_NEAR(lat1, 0.826, isclose_tol(0.826, 1e-5, 1e-3));
    const json& c = cases()["tip_depth_along_plan"];
    CHECK_MAT_NEAR(tips.back(), as<Vector3d>(c["tip_last_mm"]), 1e-9);
    CHECK_MAT_NEAR(tips[tips.size() / 2], as<Vector3d>(c["tip_mid_mm"]), 1e-9);
}
