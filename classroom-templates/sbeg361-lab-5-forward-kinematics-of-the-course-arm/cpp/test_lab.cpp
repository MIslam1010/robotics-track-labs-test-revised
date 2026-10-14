// Auto-checks for SBEG 361 Lab 5 (PoE forward kinematics of the course arm) -- C++ track. Mirrors test_lab.py
// test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*screw_axis*,*screw_exp*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: matrix_exp6, fkin_space, fkin_body, adjoint), the worked
// examples of MR Chapter 4 (Examples 4.1, 4.3-4.7), the UR5 URDF of MR Sec. 4.2, and cpp_cases.json, which holds
// the numbers of the Python reference and of the libraries the Python tests use (scipy Rotation, Pinocchio), with
// the random inputs that test_lab.py draws. Both languages are checked against the same values.
//
// test_compare_log and test_tip_checks_and_joint3_offset also write your results for the plots of Steps 4-5 to
// lab5_cpp_log_errors.csv and lab5_cpp_tip_fits.csv in the folder you run lab_tests from (see plot_results.py).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_list;
using sbeg::json;
using sbeg::Vector6d;

namespace {
constexpr double PI = 3.14159265358979323846;
const Vector3d P_TOOL(0.0, 0.0, 0.06);

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

std::string data_file(const std::string& name) { return sbeg::lab_path("../data/" + name); }

/// Screw list (6 x n) from screw axes given one per line, as the rows of the Python arrays before `.T`.
MatrixXd screws(std::initializer_list<std::array<double, 6>> axes) {
    MatrixXd S(6, static_cast<Eigen::Index>(axes.size()));
    Eigen::Index i = 0;
    for (const auto& a : axes) {
        for (Eigen::Index r = 0; r < 6; ++r) S(r, i) = a[static_cast<std::size_t>(r)];
        ++i;
    }
    return S;
}

Matrix4d mat4(std::initializer_list<double> rows) {
    Matrix4d M;
    Eigen::Index k = 0;
    for (double x : rows) {
        M(k / 4, k % 4) = x;
        ++k;
    }
    return M;
}

struct Robot {
    Matrix4d M;
    MatrixXd S;
};

// ------------------------------------------------------------------ textbook robots (MR Ch. 4)
Robot mr_ex41(double L1 = 1.0, double L2 = 1.0) {  // MR Example 4.1, 3R spatial chain (p. 141-142)
    return {mat4({0, 0, 1, L1, 0, 1, 0, 0, -1, 0, 0, -L2, 0, 0, 0, 1}),
            screws({{0, 0, 1, 0, 0, 0}, {0, -1, 0, 0, 0, -L1}, {1, 0, 0, 0, -L2, 0}})};
}

Robot mr_ex43(double L = 1.0) {  // MR Example 4.3, 6R open chain (pp. 143-144)
    Matrix4d M = Matrix4d::Identity();
    M(1, 3) = 3 * L;
    return {M, screws({{0, 0, 1, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0},
                       {-1, 0, 0, 0, 0, L}, {-1, 0, 0, 0, 0, 2 * L}, {0, 1, 0, 0, 0, 0}})};
}

const MatrixXd& b_ex46() {  // MR Ex. 4.6, L = 1
    static const MatrixXd B = screws({{0, 0, 1, -3, 0, 0}, {0, 1, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, -3},
                                      {-1, 0, 0, 0, 0, -2}, {-1, 0, 0, 0, 0, -1}, {0, 1, 0, 0, 0, 0}});
    return B;
}

Robot mr_ex44(double L1 = 1.0, double L2 = 0.5) {  // MR Example 4.4, RRPRRR (pp. 144-145)
    Matrix4d M = Matrix4d::Identity();
    M(1, 3) = L1 + L2;
    return {M, screws({{0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0},
                       {0, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, -L1}, {0, 1, 0, 0, 0, 0}})};
}

struct UrGeometry {
    double W1, W2, L1, L2, H1, H2;
};
const UrGeometry UR5_PRINTED{0.109, 0.082, 0.425, 0.392, 0.089, 0.095};               // MR Fig. 4.6
const UrGeometry UR5_URDF{0.10915, 0.0823, 0.425, 0.39225, 0.089159, 0.09465};        // MR URDF p. 153
const UrGeometry COURSE_ARM_GEOMETRY{0.0634, 0.0456, 0.110, 0.096, 0.1312, 0.0751};   // course_arm_model.py

Robot ur_type(const UrGeometry& g) {  // MR Example 4.5 screw table (p. 145)
    const double W1 = g.W1, W2 = g.W2, L1 = g.L1, L2 = g.L2, H1 = g.H1, H2 = g.H2;
    return {mat4({-1, 0, 0, L1 + L2, 0, 0, 1, W1 + W2, 0, 1, 0, H1 - H2, 0, 0, 0, 1}),
            screws({{0, 0, 1, 0, 0, 0}, {0, 1, 0, -H1, 0, 0}, {0, 1, 0, -H1, 0, L1},
                    {0, 1, 0, -H1, 0, L1 + L2}, {0, 0, -1, -W1, L1 + L2, 0}, {0, 1, 0, H2 - H1, 0, L1 + L2}})};
}

Robot wam() {  // MR Example 4.7 (pp. 148-150), body screw axes
    const double L1 = 0.550, L2 = 0.300, L3 = 0.060, W1 = 0.045;
    Matrix4d M = Matrix4d::Identity();
    M(2, 3) = L1 + L2 + L3;
    return {M, screws({{0, 0, 1, 0, 0, 0}, {0, 1, 0, L1 + L2 + L3, 0, 0}, {0, 0, 1, 0, 0, 0},
                       {0, 1, 0, L2 + L3, 0, W1}, {0, 0, 1, 0, 0, 0}, {0, 1, 0, L3, 0, 0}, {0, 0, 1, 0, 0, 0}})};
}

Robot rrp() {  // MR Exercise 5.7 RRP robot (Fig. 5.19, p. 204)
    return {mat4({-1, 0, 0, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0, 0, 0, 1}),
            screws({{0, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 2, 0}, {0, 0, 0, 0, 1, 0}})};
}

/// The course-arm reference model of course_arm_model.py: (M, Slist); `tool` moves {b} along z_b.
Robot course_arm_reference(double tool = 0.0) {
    Robot r = ur_type(COURSE_ARM_GEOMETRY);
    Matrix4d Tt = Matrix4d::Identity();
    Tt(2, 3) = tool;
    r.M = (r.M * Tt).eval();
    return r;
}

const std::vector<std::pair<std::string, Robot>>& robots() {
    static const std::vector<std::pair<std::string, Robot>> R = {
        {"ex41", mr_ex41(0.4, 0.3)}, {"ex43", mr_ex43(0.2)}, {"ex44", mr_ex44()}, {"rrp", rrp()},
        {"ur5", ur_type(UR5_URDF)},  {"course_arm", course_arm_reference()}};
    return R;
}

VectorXd vec(std::initializer_list<double> x) {
    VectorXd v(static_cast<Eigen::Index>(x.size()));
    Eigen::Index i = 0;
    for (double e : x) v(i++) = e;
    return v;
}

Vector3d point_of(const Matrix4d& T, const Vector3d& p) { return (T * Vector4d(p(0), p(1), p(2), 1.0)).head<3>(); }

/// data/tip_checks.csv: pose, servo_1..servo_6, x_mm, y_mm, z_mm (the first column is a text label).
std::pair<MatrixXd, MatrixXd> read_tip_checks() {
    std::ifstream in = sbeg::open_input(data_file("tip_checks.csv"));
    std::string line;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        if (header.empty()) {
            header = cells;
        } else {
            rows.push_back(cells);
        }
    }
    auto col = [&](const std::string& name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::runtime_error("tip_checks.csv has no column " + name);
        return static_cast<std::size_t>(it - header.begin());
    };
    const Eigen::Index N = static_cast<Eigen::Index>(rows.size());
    MatrixXd Qs(N, 6), P(N, 3);
    const char* xyz[3] = {"x_mm", "y_mm", "z_mm"};
    for (Eigen::Index k = 0; k < N; ++k) {
        const auto& r = rows[static_cast<std::size_t>(k)];
        for (int i = 0; i < 6; ++i) Qs(k, i) = std::stod(r.at(col("servo_" + std::to_string(i + 1))));
        for (int i = 0; i < 3; ++i) P(k, i) = std::stod(r.at(col(xyz[i])));
    }
    return {Qs, P};
}
}  // namespace

// ------------------------------------------------------------- Checkpoint 1: screw axes and exp
TEST_CASE("test_screw_axis_mr_example_4_1") {
    const double L1 = 0.4, L2 = 0.3;
    const MatrixXd S = mr_ex41(L1, L2).S;
    CHECK_MAT_NEAR(lab::screw_axis(Vector3d(0, 0, 1), Vector3d(0, 0, 0)), S.col(0), 1e-12);
    CHECK_MAT_NEAR(lab::screw_axis(Vector3d(0, -1, 0), Vector3d(L1, 0, 0)), S.col(1), 1e-12);
    CHECK_MAT_NEAR(lab::screw_axis(Vector3d(1, 0, 0), Vector3d(0, 0, -L2)), S.col(2), 1e-12);
    // any point on the axis gives the same screw
    CHECK_MAT_NEAR(lab::screw_axis(Vector3d(1, 0, 0), Vector3d(5.0, 0, -L2)), S.col(2), 1e-12);
}

TEST_CASE("test_screw_axis_prismatic_and_errors") {
    CHECK_MAT_NEAR(lab::screw_axis(std::nullopt, std::nullopt, Vector3d(0, 1, 0)), vec({0, 0, 0, 0, 1, 0}), 1e-12);
    CHECK_THROWS_AS(lab::screw_axis(Vector3d(0, 0, 2), Vector3d(0, 0, 0)), std::invalid_argument);  // not unit
    CHECK_THROWS_AS(lab::screw_axis(std::nullopt, std::nullopt, Vector3d(0, 0, 0.5)), std::invalid_argument);
    CHECK_THROWS_AS(lab::screw_axis(), std::invalid_argument);
}

TEST_CASE("test_screw_exp_matches_mr") {
    for (const auto& c : cases()["screw_exp_matches_mr"]) {
        const Vector6d S = as<Vector6d>(c["S"]);
        const double th = c["theta"].get<double>();
        INFO("S = " << S.transpose() << ", theta = " << th);
        CHECK_MAT_NEAR(lab::screw_exp(S, th), mr::matrix_exp6(mr::vec_to_se3(S * th)), 1e-10);
        CHECK_MAT_NEAR(lab::screw_exp(S, th), as<Matrix4d>(c["T"]), 1e-10);
    }
    Vector6d Sp;
    Sp << 0, 0, 0, 0, 0.6, 0.8;
    CHECK_MAT_NEAR(lab::screw_exp(Sp, 0.25), mr::matrix_exp6(mr::vec_to_se3(Sp * 0.25)), 1e-12);
    Vector6d Sz;
    Sz << 0, 0, 1, 0, 0, 0;
    CHECK_MAT_NEAR(lab::screw_exp(Sz, 0.0), Matrix4d::Identity(), 1e-12);
}

TEST_CASE("test_mr_example_4_5_exponentials") {
    // The two matrix exponentials printed in MR Example 4.5 (p. 146), rounded UR5 values.
    const MatrixXd S = ur_type(UR5_PRINTED).S;
    const Matrix4d E2 = mat4({0, 0, -1, 0.089, 0, 1, 0, 0, 1, 0, 0, 0.089, 0, 0, 0, 1});
    const Matrix4d E5 = mat4({0, 1, 0, 0.708, -1, 0, 0, 0.926, 0, 0, 1, 0, 0, 0, 0, 1});
    CHECK_MAT_NEAR(lab::screw_exp(S.col(1), -PI / 2), E2, 5e-4);
    CHECK_MAT_NEAR(lab::screw_exp(S.col(4), PI / 2), E5, 5e-4);
}

// ------------------------------------------------------------- Checkpoint 2: PoE FK, space and body
TEST_CASE("test_fk_space_matches_mr") {
    // pytest runs this test once per robot (6 items); here one test case loops over the robots.
    for (const auto& entry : robots()) {
        const std::string& name = entry.first;
        const Robot& rob = entry.second;
        INFO("robot = " << name);
        const json& c = cases()["fk_space_matches_mr"][name];
        REQUIRE_MAT_NEAR(rob.M, sbeg::as_mat(c["M"]), 1e-15);  // same robot as in test_lab.py
        REQUIRE_MAT_NEAR(rob.S, sbeg::as_mat(c["Slist"]), 1e-15);
        const auto thetas = sbeg::as_mat(c["thetas"]);
        const auto Ts = as_list<Matrix4d>(c["T"]);
        for (Eigen::Index k = 0; k < thetas.rows(); ++k) {
            const VectorXd th = thetas.row(k).transpose();
            CHECK_MAT_NEAR(lab::fk_space(rob.M, rob.S, th), mr::fkin_space(rob.M, rob.S, th), 1e-10);
            CHECK_MAT_NEAR(lab::fk_space(rob.M, rob.S, th), Ts[static_cast<std::size_t>(k)], 1e-10);
        }
        CHECK_MAT_NEAR(lab::fk_space(rob.M, rob.S, VectorXd::Zero(rob.S.cols())), rob.M, 1e-12);
    }
}

TEST_CASE("test_ur5_mr_example_4_5") {
    const Robot r = ur_type(UR5_PRINTED);
    const Matrix4d T = lab::fk_space(r.M, r.S, vec({0, -PI / 2, 0, 0, PI / 2, 0}));
    const Matrix4d T_mr = mat4({0, -1, 0, 0.095, 1, 0, 0, 0.109, 0, 0, 1, 0.988, 0, 0, 0, 1});
    CHECK_MAT_NEAR(T, T_mr, 5e-4);
}

TEST_CASE("test_rrp_check_value") {
    const Robot r = rrp();
    const Matrix4d T = lab::fk_space(r.M, r.S, vec({PI / 2, PI / 2, 1.0}));
    CHECK_MAT_NEAR(T, mat4({0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 1}), 1e-12);
}

TEST_CASE("test_body_screws_mr_example_4_6") {
    const Robot r = mr_ex43(1.0);
    CHECK_MAT_NEAR(lab::body_screws(r.S, r.M), b_ex46(), 1e-12);
}

TEST_CASE("test_fk_body_equals_space") {
    // pytest runs this test once per robot (6 items); here one test case loops over the robots.
    for (const auto& entry : robots()) {
        const std::string& name = entry.first;
        const Robot& rob = entry.second;
        INFO("robot = " << name);
        const MatrixXd B = mr::adjoint(mr::trans_inv(rob.M)) * rob.S;
        const json& c = cases()["fk_body_equals_space"][name];
        REQUIRE_MAT_NEAR(B, sbeg::as_mat(c["Blist"]), 1e-12);  // same body screws as in the Python reference
        const auto thetas = sbeg::as_mat(c["thetas"]);
        const auto Ts = as_list<Matrix4d>(c["T"]);
        for (Eigen::Index k = 0; k < thetas.rows(); ++k) {
            const VectorXd th = thetas.row(k).transpose();
            CHECK_MAT_NEAR(lab::fk_body(rob.M, B, th), mr::fkin_body(rob.M, B, th), 1e-10);
            CHECK_MAT_NEAR(lab::fk_body(rob.M, B, th), lab::fk_space(rob.M, rob.S, th), 1e-10);
            CHECK_MAT_NEAR(lab::fk_body(rob.M, B, th), Ts[static_cast<std::size_t>(k)], 1e-10);
        }
    }
}

TEST_CASE("test_wam_mr_example_4_7") {
    const Robot r = wam();
    const Matrix4d T = lab::fk_body(r.M, r.S, vec({0, PI / 4, 0, -PI / 4, 0, -PI / 2, 0}));
    const Matrix4d T_mr = mat4({0, 0, -1, 0.3157, 0, 1, 0, 0, 1, 0, 0, 0.6571, 0, 0, 0, 1});
    CHECK_MAT_NEAR(T, T_mr, 5e-5);
}

TEST_CASE("test_course_arm_screws") {
    const UrGeometry& g = COURSE_ARM_GEOMETRY;
    const Robot ref = course_arm_reference();
    const json& c = cases()["course_arm_screws"];
    REQUIRE_MAT_NEAR(ref.S, sbeg::as_mat(c["Slist"]), 1e-15);  // same reference as course_arm_model.py
    REQUIRE_MAT_NEAR(ref.M, as<Matrix4d>(c["M"]), 1e-15);
    const auto [M, S] = lab::course_arm_screws(g.W1, g.W2, g.L1, g.L2, g.H1, g.H2);
    CHECK_MAT_NEAR(S, ref.S, 1e-12);
    CHECK_MAT_NEAR(M, ref.M, 1e-12);
    const Matrix4d Mt = lab::course_arm_screws(g.W1, g.W2, g.L1, g.L2, g.H1, g.H2, 0.06).first;
    CHECK_MAT_NEAR(Mt.col(3).head<3>(), Vector3d(0.206, 0.169, 0.0561), 1e-12);  // S3 needle guide
    CHECK_MAT_NEAR(Mt, as<Matrix4d>(c["M_tool"]), 1e-12);
}

TEST_CASE("test_course_arm_mr_example_4_5_pose") {
    // Course arm at the pose of MR Example 4.5: p = (H2, W1, H1 + L1 + L2 + W2).
    const UrGeometry& g = COURSE_ARM_GEOMETRY;
    const auto [M, S] = lab::course_arm_screws(g.W1, g.W2, g.L1, g.L2, g.H1, g.H2);
    const Matrix4d T = lab::fk_space(M, S, vec({0, -PI / 2, 0, 0, PI / 2, 0}));
    CHECK_MAT_NEAR(T, mat4({0, -1, 0, 0.0751, 1, 0, 0, 0.0634, 0, 0, 1, 0.3828, 0, 0, 0, 1}), 1e-12);
}

// (test_course_arm_copy_is_identical_to_shared is an instructor-side data check of the Python track; the C++
//  track checks its copy of the reference model against cpp_cases.json in test_course_arm_screws instead.)

// ------------------------------------------------------------- Checkpoint 3: URDF -> PoE
TEST_CASE("test_rpy_to_rot_matches_scipy") {
    for (const auto& c : cases()["rpy_to_rot_matches_scipy"]) {
        const Vector3d rpy = as<Vector3d>(c["rpy"]);
        INFO("rpy = " << rpy.transpose());
        CHECK_MAT_NEAR(lab::rpy_to_rot(rpy), as<Matrix3d>(c["R"]), 1e-12);  // scipy Rotation.from_euler("xyz")
    }
    Matrix3d R;
    R << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    CHECK_MAT_NEAR(lab::rpy_to_rot(Vector3d(0, PI / 2, 0)), R, 1e-12);
}

TEST_CASE("test_urdf_course_arm") {
    const auto joints = lab::read_urdf_joints(data_file("course_arm.urdf"));
    const lab::PoeModel m = lab::poe_from_urdf(joints, "tool0");
    const Robot ref = course_arm_reference();
    CHECK_MAT_NEAR(m.Slist, ref.S, 1e-9);
    CHECK_MAT_NEAR(m.M, ref.M, 1e-9);
    CHECK(m.names == std::vector<std::string>{"j1", "j2", "j3", "j4", "j5", "j6"});
    REQUIRE(m.limits.rows() == 6);
    REQUIRE(m.limits.cols() == 2);
    CHECK_MAT_NEAR(m.limits.col(1) * (180.0 / PI), vec({165, 165, 165, 165, 165, 175}), 1e-6);
}

TEST_CASE("test_urdf_ur5_from_mr_text") {
    // The UR5 URDF of MR Sec. 4.2 gives the screw table of MR Example 4.5 (full-precision values).
    const auto joints = lab::read_urdf_joints(data_file("ur5_mr.urdf"));
    const lab::PoeModel m = lab::poe_from_urdf(joints, "ee_link");
    const Robot ref = ur_type(UR5_URDF);
    CHECK_MAT_NEAR(m.Slist, ref.S, 1e-8);
    CHECK_MAT_NEAR(m.M, ref.M, 1e-8);
    CHECK(m.names.size() == 6);
    CHECK(m.limits.size() == 12);
    CHECK(m.limits.array().isInf().all());
}

TEST_CASE("test_urdf_matches_pinocchio") {
    // Python compares with Pinocchio directly; here with Pinocchio's poses stored in cpp_cases.json.
    const lab::PoeModel m = lab::poe_from_urdf(lab::read_urdf_joints(data_file("course_arm.urdf")), "tool0");
    for (const auto& c : cases()["urdf_matches_pinocchio"]) {
        const VectorXd q = sbeg::as_vec(c["q"]);
        INFO("q = " << q.transpose());
        CHECK_MAT_NEAR(lab::fk_space(m.M, m.Slist, q), as<Matrix4d>(c["T"]), 1e-9);
    }
}

// ------------------------------------------------------------- Checkpoint 4: simulation vs hardware
TEST_CASE("test_servo_to_model") {
    const VectorXd off = lab::load_offsets(data_file("servo_offsets.json"));
    CHECK_MAT_NEAR(off, vec({0, 90, 0, 90, 0, 0}), 1e-12);
    const VectorXd zero = VectorXd::Zero(6);
    const VectorXd q = lab::servo_to_model(zero, off);
    CHECK_MAT_NEAR(q, vec({0, -PI / 2, 0, -PI / 2, 0, 0}), 1e-12);
    const MatrixXd zeros = MatrixXd::Zero(3, 6);
    const MatrixXd Q = lab::servo_to_model(zeros, off);
    CHECK(Q.rows() == 3);
    CHECK(Q.cols() == 6);
}

TEST_CASE("test_servo_zero_is_candle_pose") {
    // Servo zero (arm straight up): tool flange above joint 1 at H1 + L1 + L2 + H2 (illustrative calibration).
    const Robot r = course_arm_reference();
    const VectorXd off = lab::load_offsets(data_file("servo_offsets.json"));
    const MatrixXd zeros = MatrixXd::Zero(1, 6);
    const MatrixXd P = lab::tool_points(r.M, r.S, lab::servo_to_model(zeros, off));
    REQUIRE(P.rows() == 1);
    const UrGeometry& g = COURSE_ARM_GEOMETRY;
    CHECK_MAT_NEAR(P.row(0), Vector3d(0, g.W1 + g.W2, g.H1 + g.L1 + g.L2 + g.H2), 1e-12);
}

TEST_CASE("test_tool_points_matches_mr") {
    const Robot r = course_arm_reference();
    const json& c = cases()["tool_points_matches_mr"];
    const MatrixXd Q = sbeg::as_mat(c["Q"]);
    const MatrixXd P = lab::tool_points(r.M, r.S, Q, P_TOOL);
    REQUIRE(P.rows() == Q.rows());
    for (Eigen::Index k = 0; k < Q.rows(); ++k) {
        const VectorXd q = Q.row(k).transpose();
        CHECK_MAT_NEAR(P.row(k), point_of(mr::fkin_space(r.M, r.S, q), P_TOOL), 1e-10);
    }
    CHECK_MAT_NEAR(P, sbeg::as_mat(c["P"]), 1e-10);
}

TEST_CASE("test_compare_log") {
    const lab::JointLog log = lab::load_joint_log(data_file("course_arm_log.csv"));
    const VectorXd off = lab::load_offsets(data_file("servo_offsets.json"));
    const Robot r = course_arm_reference();
    const lab::LogComparison res = lab::compare_log(log, r.M, r.S, off, P_TOOL);
    REQUIRE(log.stop.size() == log.t.size());
    Eigen::Index k0 = 0;
    while (k0 < log.stop.size() && !(log.stop(k0) > 0.5)) ++k0;
    CHECK(res.n_samples == k0);
    CHECK(k0 == 900);
    const MatrixXd e = log.meas.topRows(k0) - log.cmd.topRows(k0);
    const VectorXd rms = (e.array().square().colwise().sum() / static_cast<double>(k0)).sqrt().transpose();
    CHECK_MAT_NEAR(res.joint_rms_deg, rms, 1e-12);
    CHECK_SCALAR_NEAR(res.joint_mean_deg(1), -0.50, 0.05);  // gravity droop of joint 2
    VectorXd err(k0);
    for (Eigen::Index k = 0; k < k0; ++k) {
        const VectorXd qm = (log.meas.row(k) - off.transpose()).transpose() * (PI / 180.0);
        const VectorXd qc = (log.cmd.row(k) - off.transpose()).transpose() * (PI / 180.0);
        err(k) = 1e3 * (point_of(mr::fkin_space(r.M, r.S, qm), P_TOOL) -
                        point_of(mr::fkin_space(r.M, r.S, qc), P_TOOL)).norm();
    }
    CHECK_MAT_NEAR(res.task_err_mm, err, 1e-9);
    CHECK_SCALAR_NEAR(res.task_rms_mm, std::sqrt(err.squaredNorm() / static_cast<double>(k0)), 1e-9);
    CHECK_SCALAR_NEAR(res.task_max_mm, err.maxCoeff(), 1e-9);
    CHECK(3.0 < res.task_rms_mm);
    CHECK(res.task_rms_mm < 4.5);
    CHECK(res.task_max_mm < 10.0);
    CHECK(res.n_samples == cases()["compare_log"]["n_samples"].get<int>());
    CHECK_SCALAR_NEAR(res.task_rms_mm, cases()["compare_log"]["task_rms_mm"].get<double>(), 1e-9);  // Python reference
    CHECK_SCALAR_NEAR(res.task_max_mm, cases()["compare_log"]["task_max_mm"].get<double>(), 1e-9);
    // Results for the plots of Step 4b-c (plot_results.py): t, e_1..e_n (degrees), your task_err_mm.
    if (res.task_err_mm.size() == k0) {
        MatrixXd out(k0, 2 + e.cols());
        out << log.t.head(k0), e, res.task_err_mm;
        std::vector<std::string> header = {"t"};
        for (Eigen::Index i = 0; i < e.cols(); ++i) header.push_back("e_" + std::to_string(i + 1));
        header.push_back("task_err_mm");
        sbeg::write_csv("lab5_cpp_log_errors.csv", header, out);
    }
}

TEST_CASE("test_tip_checks_and_joint3_offset") {
    const auto [Qs, P] = read_tip_checks();
    const VectorXd off = lab::load_offsets(data_file("servo_offsets.json"));
    const Robot r = course_arm_reference();
    const MatrixXd Q = lab::servo_to_model(Qs, off);
    const VectorXd e = lab::tip_errors_mm(r.M, r.S, Q, P, P_TOOL);
    VectorXd e_ref(Q.rows());
    for (Eigen::Index k = 0; k < Q.rows(); ++k) {
        const VectorXd q = (Qs.row(k) - off.transpose()).transpose() * (PI / 180.0);
        e_ref(k) = (1e3 * point_of(mr::fkin_space(r.M, r.S, q), P_TOOL) - P.row(k).transpose()).norm();
    }
    CHECK_MAT_NEAR(e, e_ref, 1e-9);
    CHECK_MAT_NEAR(e, sbeg::as_vec(cases()["tip_checks_and_joint3_offset"]["errors_mm"]), 1e-9);
    const double rms = std::sqrt(e.squaredNorm() / static_cast<double>(e.size()));
    CHECK(4.0 < rms);
    CHECK(rms < 6.0);
    std::vector<std::pair<double, double>> fits;
    for (int j = 0; j < 6; ++j) fits.push_back(lab::fit_joint_offset(r.M, r.S, Q, P, j, P_TOOL));
    std::size_t best = 0;
    for (std::size_t j = 1; j < fits.size(); ++j)
        if (fits[j].second < fits[best].second) best = j;
    CHECK(best == 2);  // joint 3 (0-based index 2)
    CHECK(std::abs(fits[2].first - 1.5) < 0.1);
    CHECK(fits[2].second < 1.0);
    std::vector<double> rms_after;
    for (const auto& f : fits) rms_after.push_back(f.second);
    std::sort(rms_after.begin(), rms_after.end());
    CHECK(rms_after[1] > 2.0);  // clearly better than any other joint
    // Table of Step 5b (plot_results.py): joint (1-based), delta_deg, rms_after_mm.
    MatrixXd out(6, 3);
    for (int j = 0; j < 6; ++j) out.row(j) << j + 1, fits[static_cast<std::size_t>(j)].first,
                                     fits[static_cast<std::size_t>(j)].second;
    sbeg::write_csv("lab5_cpp_tip_fits.csv", {"joint", "delta_deg", "rms_after_mm"}, out);
}
