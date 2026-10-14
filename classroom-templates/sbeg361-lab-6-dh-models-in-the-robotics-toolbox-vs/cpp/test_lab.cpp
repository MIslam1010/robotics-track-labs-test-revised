// Auto-checks for SBEG 361 Lab 6 (DH models vs PoE) -- C++ track. Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*dh_std*,*dh_mod*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mr::fkin_space), Corke's printed examples (Sec. 7.1-7.2,
// pp. 140-142), MR Example C.1 = MR Example 4.1, MR Exercise 5.7 (RRP), and cpp_cases.json, which holds the
// random inputs of test_lab.py and the numbers of the Python reference and of the Robotics Toolbox for Python.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_list;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double D2R = PI / 180.0;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// data/q_samples.csv: 1000 x 6 joint vectors (row = sample).
const MatrixXd& q_samples() {
    static const MatrixXd Q = sbeg::read_csv(sbeg::lab_path("../data/q_samples.csv")).data;
    return Q;
}
VectorXd sample(Eigen::Index k) { return q_samples().row(k).transpose(); }

Matrix4d rz(double t) { return mr::rp_to_trans(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, t))), Vector3d::Zero()); }
Matrix4d rx(double t) { return mr::rp_to_trans(mr::matrix_exp3(mr::vec_to_so3(Vector3d(t, 0, 0))), Vector3d::Zero()); }
Matrix4d tz(double d) { return mr::rp_to_trans(Matrix3d::Identity(), Vector3d(0, 0, d)); }
Matrix4d tx(double a) { return mr::rp_to_trans(Matrix3d::Identity(), Vector3d(a, 0, 0)); }

lab::DHTable table_from_json(const json& t) {
    lab::DHTable tab;
    tab.convention = t["convention"].get<std::string>();
    tab.types = t["types"].get<std::string>();
    tab.d = as_vec(t["d"]);
    tab.a = as_vec(t["a"]);
    tab.alpha = as_vec(t["alpha"]);
    tab.offset = as_vec(t["offset"]);
    tab.base = as<Matrix4d>(t["base"]);
    tab.tool = as<Matrix4d>(t["tool"]);
    return tab;
}

/// Reference FK of a table built from elementary transforms (not from the student's code).
Matrix4d ref_fk(const lab::DHTable& tab, const VectorXd& q) {
    Matrix4d T = tab.base;
    for (std::size_t k = 0; k < tab.types.size(); ++k) {
        const auto i = static_cast<Eigen::Index>(k);
        const double th = tab.offset(i) + (tab.types[k] == 'R' ? q(i) : 0.0);
        const double d = tab.d(i) + (tab.types[k] == 'P' ? q(i) : 0.0);
        const double a = tab.a(i), al = tab.alpha(i);
        const Matrix4d A = tab.convention == "standard" ? Matrix4d(rz(th) * tz(d) * tx(a) * rx(al))
                                                        : Matrix4d(rx(al) * tx(a) * tz(d) * rz(th));
        T = T * A;
    }
    return T * tab.tool;
}

lab::DHTable make_table(const std::string& convention, const std::string& types, const VectorXd& d,
                        const VectorXd& a, const VectorXd& alpha, const VectorXd& offset, const Matrix4d& base,
                        const Matrix4d& tool) {
    lab::DHTable t;
    t.convention = convention;
    t.types = types;
    t.d = d;
    t.a = a;
    t.alpha = alpha;
    t.offset = offset;
    t.base = base;
    t.tool = tool;
    return t;
}

VectorXd vec3(double x, double y, double z) {
    VectorXd v(3);
    v << x, y, z;
    return v;
}

// MR Example 4.1 (Fig. 4.3), L1 = 0.3, L2 = 0.2, and the modified DH table of MR Example C.1 (p. 593)
constexpr double L1E = 0.3, L2E = 0.2;
MatrixXd s_ex41() {
    MatrixXd S(3, 6);
    S << 0, 0, 1, 0, 0, 0,
         0, -1, 0, 0, 0, -L1E,
         1, 0, 0, 0, -L2E, 0;
    return S.transpose();
}
Matrix4d m_ex41() {
    Matrix4d M;
    M << 0, 0, 1, L1E,
         0, 1, 0, 0,
         -1, 0, 0, -L2E,
         0, 0, 0, 1;
    return M;
}
lab::DHTable mdh_exc1() {
    return make_table("modified", "RRR", VectorXd::Zero(3), vec3(0, L1E, L2E), vec3(0, 90, -90) * D2R,
                      vec3(0, -90, 0) * D2R, Matrix4d::Identity(), Matrix4d::Identity());
}
// MR Exercise 5.7 RRP robot (Fig. 5.19) and a standard DH table of it
MatrixXd s_rrp() {
    MatrixXd S(3, 6);
    S << 0, 0, 1, 0, 0, 0,
         1, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 1, 0;
    return S.transpose();
}
Matrix4d m_rrp() {
    Matrix4d M;
    M << -1, 0, 0, 0,
         0, 0, 1, 3,
         0, 1, 0, 2,
         0, 0, 0, 1;
    return M;
}
lab::DHTable dh_rrp() {
    Matrix4d tool;
    tool << 0, 1, 0, 0,
            -1, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    return make_table("standard", "RRP", vec3(2.0, 0, 3.0), VectorXd::Zero(3), vec3(90, 90, 0) * D2R,
                      vec3(90, 90, 0) * D2R, Matrix4d::Identity(), tool);
}

/// PoE model of the RCM holder (the same construction as rcm_poe() in test_lab.py).
void rcm_poe(MatrixXd& S, Matrix4d& M) {
    const Vector3d p = lab::rcm_port();
    const Vector3d z(0, 0, 1), y(0, 1, 0), mz(0, 0, -1);
    S = MatrixXd::Zero(6, 4);
    S.block<3, 1>(0, 0) = z;
    S.block<3, 1>(3, 0) = -z.cross(p);
    S.block<3, 1>(0, 1) = y;
    S.block<3, 1>(3, 1) = -y.cross(p);
    S(5, 2) = -1;
    S.block<3, 1>(0, 3) = mz;
    S.block<3, 1>(3, 3) = -mz.cross(p);
    const Matrix3d R = Vector3d(1.0, -1.0, -1.0).asDiagonal();
    M = mr::rp_to_trans(R, p + Vector3d(0, 0, -lab::RCM_TIP0));
}

/// A writable file in the system temporary folder (UTF-8 path, as sbeg::read_csv expects).
std::string temp_file(const std::string& name) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
#if defined(__cpp_char8_t)
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
#else
    return p.u8string();
#endif
}

/// Write poses in the pose-file format (r11..r33, px, py, pz) with full precision.
void write_poses(const std::string& path, const std::vector<Matrix4d>& Ts) {
    MatrixXd rows(static_cast<Eigen::Index>(Ts.size()), 12);
    for (std::size_t k = 0; k < Ts.size(); ++k) {
        const auto r = static_cast<Eigen::Index>(k);
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 3; ++c) rows(r, 3 * i + c) = Ts[k](i, c);
        for (int i = 0; i < 3; ++i) rows(r, 9 + i) = Ts[k](i, 3);
    }
    sbeg::write_csv(path, {"r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "px", "py", "pz"}, rows);
}
}  // namespace

// ------------------------------------------------------------- provided helpers (library signatures)
TEST_CASE("test_provided_helpers_use_library_signatures") {
    // fk_space(M, Slist, theta) as in Lab 5; screw_exp(S, theta) = MatrixExp6([S] theta).  Given code: this test
    // passes with the untouched starter (as in Python) and protects the helpers if you replace them.
    const MatrixXd S = lab::course_arm_slist();
    const sbeg::Vector6d S1 = S.col(1);
    CHECK_MAT_NEAR(lab::screw_exp(S1, 0.7), mr::matrix_exp6(mr::vec_to_se3(S1 * 0.7)), 1e-12);
    for (Eigen::Index k = 0; k < 20; ++k)
        CHECK_MAT_NEAR(lab::fk_space(lab::course_arm_m(), S, sample(k)),
                       mr::fkin_space(lab::course_arm_m(), S, sample(k)), 1e-12);
}

// ------------------------------------------------------------- Checkpoint 1: link transforms
TEST_CASE("test_dh_std_corke_link_example") {
    // Corke p. 140: Link([0, 0.1, 0.2, pi/2]).A(0.5)
    Matrix4d expected;
    expected << 0.8776, 0, 0.4794, 0.1755,
                0.4794, 0, -0.8776, 0.0959,
                0, 1, 0, 0.1,
                0, 0, 0, 1;
    CHECK_MAT_NEAR(lab::dh_std(0.5, 0.1, 0.2, PI / 2), expected, 1e-4);
    CHECK_MAT_NEAR(lab::dh_std(0.5, 0.1, 0.2, PI / 2), as<Matrix4d>(cases()["dh_std_corke_link_example"]["A"]), 1e-12);
}

TEST_CASE("test_dh_std_is_rz_tz_tx_rx") {
    for (const auto& c : cases()["dh_std_is_rz_tz_tx_rx"]) {
        const double th = c["theta"].get<double>(), d = c["d"].get<double>(), a = c["a"].get<double>(),
                     al = c["alpha"].get<double>();
        INFO("theta, d, a, alpha = " << th << ", " << d << ", " << a << ", " << al);
        CHECK_MAT_NEAR(lab::dh_std(th, d, a, al), rz(th) * tz(d) * tx(a) * rx(al), 1e-12);
        CHECK_MAT_NEAR(lab::dh_std(th, d, a, al), as<Matrix4d>(c["A"]), 1e-12);
    }
}

TEST_CASE("test_dh_mod_is_rx_tx_tz_rz") {
    for (const auto& c : cases()["dh_mod_is_rx_tx_tz_rz"]) {
        const double al = c["alpha"].get<double>(), a = c["a"].get<double>(), d = c["d"].get<double>(),
                     th = c["theta"].get<double>();
        INFO("alpha, a, d, theta = " << al << ", " << a << ", " << d << ", " << th);
        CHECK_MAT_NEAR(lab::dh_mod(al, a, d, th), rx(al) * tx(a) * tz(d) * rz(th), 1e-12);
        CHECK_MAT_NEAR(lab::dh_mod(al, a, d, th), as<Matrix4d>(c["A"]), 1e-12);
    }
}

// ------------------------------------------------------------- Checkpoint 2: FK from a table
TEST_CASE("test_fk_dh_corke_two_link") {
    // Corke Sec. 7.2.1 (p. 142): two-link planar robot, a1 = a2 = 1
    VectorXd ones = VectorXd::Ones(2), zeros = VectorXd::Zero(2);
    const lab::DHTable two = make_table("standard", "RR", zeros, ones, zeros, zeros, Matrix4d::Identity(),
                                        Matrix4d::Identity());
    VectorXd q(2);
    q << PI / 4, -PI / 4;
    CHECK_MAT_NEAR(lab::fk_dh(two, zeros).col(3).head<3>(), Vector3d(2, 0, 0), 1e-12);
    CHECK_MAT_NEAR(lab::fk_dh(two, q).col(3).head<3>(), Vector3d(1.7071, 0.7071, 0), 1e-4);
}

TEST_CASE("test_fk_dh_mr_example_c1_equals_example_4_1") {
    const lab::DHTable tab = mdh_exc1();
    for (const auto& c : cases()["fk_dh_mr_example_c1_equals_example_4_1"]) {
        const VectorXd q = as_vec(c["q"]);
        INFO("q = " << q.transpose());
        CHECK_MAT_NEAR(lab::fk_dh(tab, q), mr::fkin_space(m_ex41(), s_ex41(), q), 1e-12);
    }
}

TEST_CASE("test_fk_dh_prismatic_mr_exercise_5_7") {
    const lab::DHTable tab = dh_rrp();
    Matrix4d expected;
    expected << 0, 1, 0, 0,
                -1, 0, 0, 0,
                0, 0, 1, 6,
                0, 0, 0, 1;
    CHECK_MAT_NEAR(lab::fk_dh(tab, vec3(PI / 2, PI / 2, 1.0)), expected, 1e-9);
    for (const auto& c : cases()["fk_dh_prismatic_mr_exercise_5_7"]) {
        const VectorXd q = as_vec(c["q"]);
        INFO("q = " << q.transpose());
        CHECK_MAT_NEAR(lab::fk_dh(tab, q), mr::fkin_space(m_rrp(), s_rrp(), q), 1e-9);
    }
}

TEST_CASE("test_fk_dh_random_tables_both_conventions") {
    int k = 0;
    for (const auto& c : cases()["fk_dh_random_tables_both_conventions"]) {
        const lab::DHTable tab = table_from_json(c["table"]);
        INFO("random table " << k++ << " (" << tab.convention << ", " << tab.types << ")");
        const MatrixXd qs = as_mat(c["qs"]);
        for (Eigen::Index i = 0; i < qs.rows(); ++i) {
            const VectorXd q = qs.row(i).transpose();
            CHECK_MAT_NEAR(lab::fk_dh(tab, q), ref_fk(tab, q), 1e-12);
        }
        const VectorXd zero = VectorXd::Zero(5);
        const std::vector<Matrix4d> frames = lab::link_frames(tab, zero);
        REQUIRE(frames.size() == 6);
        CHECK_MAT_NEAR(frames[0], tab.base, 1e-12);
        CHECK_MAT_NEAR(frames.back() * tab.tool, lab::fk_dh(tab, zero), 1e-12);
    }
    lab::DHTable bad = table_from_json(cases()["fk_dh_random_tables_both_conventions"][0]["table"]);
    bad.convention = "craig-ish";
    CHECK_THROWS_AS(lab::fk_dh(bad, VectorXd::Zero(5)), std::invalid_argument);
}

// ------------------------------------------------------------- Checkpoint 3: course arm DH tables
TEST_CASE("test_course_arm_dh_matches_poe") {
    const lab::DHTable tab = lab::course_arm_dh();
    CHECK(tab.convention == "standard");
    REQUIRE(tab.types == "RRRRRR");
    const MatrixXd S = lab::course_arm_slist();
    const Matrix4d M = lab::course_arm_m();
    CHECK_MAT_NEAR(lab::fk_dh(tab, VectorXd::Zero(6)), M, 1e-12);
    int bad = 0;
    for (Eigen::Index k = 0; k < q_samples().rows(); ++k) {
        const VectorXd q = sample(k);
        const sbeg::NearResult r = sbeg::compare_near(lab::fk_dh(tab, q), mr::fkin_space(M, S, q), 1e-12);
        if (!r.ok && bad++ < 3) CHECK_MESSAGE(r.ok, "sample " << k << ": " << r.msg);  // report the first 3
    }
    CHECK_MESSAGE(bad == 0, bad << " of 1000 samples differ from the PoE model by more than 1e-12");
}

TEST_CASE("test_course_arm_mdh_matches_poe") {
    const lab::DHTable tab = lab::course_arm_mdh();
    CHECK(tab.convention == "modified");
    REQUIRE(tab.types.size() == 6);
    const MatrixXd S = lab::course_arm_slist();
    const Matrix4d M = lab::course_arm_m();
    int bad = 0;
    for (Eigen::Index k = 0; k < 200; ++k) {
        const VectorXd q = sample(k);
        const sbeg::NearResult r = sbeg::compare_near(lab::fk_dh(tab, q), mr::fkin_space(M, S, q), 1e-12);
        if (!r.ok && bad++ < 3) CHECK_MESSAGE(r.ok, "sample " << k << ": " << r.msg);
    }
    CHECK_MESSAGE(bad == 0, bad << " of 200 samples differ from the PoE model by more than 1e-12");
}

TEST_CASE("test_course_arm_ready_pose_checkpoint") {
    // Value quoted in the handout (Step 3): flange position at q_home
    const Matrix4d T = lab::fk_dh(lab::course_arm_dh(), lab::q_home());
    CHECK_MAT_NEAR(T.col(3).head<3>(), Vector3d(0.23407, 0.06344, 0.08106), 5e-5);
}

// ------------------------------------------------------------- Checkpoint 4: conversions
TEST_CASE("test_poe_from_dh_course_arm_both_conventions") {
    for (const lab::DHTable& tab : {lab::course_arm_dh(), lab::course_arm_mdh()}) {
        INFO("convention " << tab.convention);
        const auto [S, M] = lab::poe_from_dh(tab);
        CHECK_MAT_NEAR(S, lab::course_arm_slist(), 1e-12);
        CHECK_MAT_NEAR(M, lab::course_arm_m(), 1e-12);
    }
}

TEST_CASE("test_poe_from_dh_textbook_chains") {
    {
        const auto [S, M] = lab::poe_from_dh(mdh_exc1());
        CHECK_MAT_NEAR(S, s_ex41(), 1e-12);
        CHECK_MAT_NEAR(M, m_ex41(), 1e-12);
    }
    {
        const auto [S, M] = lab::poe_from_dh(dh_rrp());
        CHECK_MAT_NEAR(S, s_rrp(), 1e-9);
        CHECK_MAT_NEAR(M, m_rrp(), 1e-9);
    }
}

TEST_CASE("test_poe_from_dh_random_tables") {
    int k = 0;
    for (const auto& c : cases()["poe_from_dh_random_tables"]) {
        const lab::DHTable tab = table_from_json(c["table"]);
        INFO("random table " << k++ << " (" << tab.convention << ", " << tab.types << ")");
        const auto [S, M] = lab::poe_from_dh(tab);
        REQUIRE(S.rows() == 6);
        REQUIRE(S.cols() == 6);
        const MatrixXd qs = as_mat(c["qs"]);
        for (Eigen::Index i = 0; i < qs.rows(); ++i) {
            const VectorXd q = qs.row(i).transpose();
            CHECK_MAT_NEAR(mr::fkin_space(M, S, q), ref_fk(tab, q), 1e-10);
        }
    }
}

TEST_CASE("test_std_mod_round_trip") {
    int k = 0;
    for (const auto& c : cases()["std_mod_round_trip"]) {
        const lab::DHTable std_tab = table_from_json(c["table"]);
        INFO("random table " << k++ << " (" << std_tab.types << ")");
        const lab::DHTable mod = lab::std_to_mod(std_tab);
        const lab::DHTable back = lab::mod_to_std(mod);
        CHECK(mod.convention == "modified");
        CHECK(back.convention == "standard");
        CHECK(std_tab.convention == "standard");  // input not modified
        REQUIRE(mod.types == std_tab.types);
        REQUIRE(back.types == std_tab.types);
        const MatrixXd qs = as_mat(c["qs"]);
        for (Eigen::Index i = 0; i < qs.rows(); ++i) {
            const VectorXd q = qs.row(i).transpose();
            const Matrix4d T = ref_fk(std_tab, q);
            CHECK_MAT_NEAR(ref_fk(mod, q), T, 1e-12);
            CHECK_MAT_NEAR(ref_fk(back, q), T, 1e-12);
        }
    }
    const lab::DHTable mod = lab::std_to_mod(lab::course_arm_dh());
    CHECK_MAT_NEAR(mod.a, lab::course_arm_mdh().a, 1e-12);
    CHECK_MAT_NEAR(mod.alpha, lab::course_arm_mdh().alpha, 1e-12);
}

// ------------------------------------------------------------- Checkpoint 5: RCM robot (and toolbox values)
TEST_CASE("test_rcm_dh_matches_poe_and_closed_form") {
    const lab::DHTable tab = lab::rcm_dh();
    REQUIRE(tab.types == "RRPR");
    MatrixXd S;
    Matrix4d M;
    rcm_poe(S, M);
    const Vector3d port = lab::rcm_port();
    const MatrixXd qs = as_mat(cases()["rcm_dh_matches_poe_and_closed_form"]["qs"]);
    for (Eigen::Index i = 0; i < qs.rows(); ++i) {
        const VectorXd q = qs.row(i).transpose();
        INFO("q = " << q.transpose());
        const Matrix4d T = lab::fk_dh(tab, q);
        CHECK_MAT_NEAR(T, mr::fkin_space(M, S, q), 1e-12);
        const Vector3d u(-std::sin(q(1)) * std::cos(q(0)), -std::sin(q(1)) * std::sin(q(0)), -std::cos(q(1)));
        CHECK_MAT_NEAR(T.col(3).head<3>(), port + (lab::RCM_TIP0 + q(2)) * u, 1e-12);
    }
    const auto [S2, M2] = lab::poe_from_dh(tab);
    CHECK_MAT_NEAR(S2, S, 1e-12);
    CHECK_MAT_NEAR(M2, M, 1e-12);
}

TEST_CASE("test_rcm_property_and_distance") {
    const lab::DHTable tab = lab::rcm_dh();
    const MatrixXd qs = as_mat(cases()["rcm_property_and_distance"]["qs"]);
    double dmax = 0.0;
    for (Eigen::Index i = 0; i < qs.rows(); ++i) dmax = std::max(dmax, lab::rcm_distance(lab::fk_dh(tab, qs.row(i).transpose())));
    CHECK(dmax < 1e-12);
    const Matrix4d T = mr::rp_to_trans(Matrix3d::Identity(), lab::rcm_port() + Vector3d(0.003, -0.004, 0.2));  // 5 mm away
    CHECK(std::abs(lab::rcm_distance(T) - 0.005) < 1e-12);
}

TEST_CASE("test_rtb_course_arm_both_conventions") {
    // Python builds the robot in the Robotics Toolbox (rtb_robot). The C++ track compares your fk_dh with the
    // Toolbox's fkine values for the same tables, computed in Python and stored in cpp_cases.json.
    const json& c = cases()["rtb_course_arm_both_conventions"];
    const auto n = c["n"].get<Eigen::Index>();
    const std::vector<Matrix4d> T_dh = as_list<Matrix4d>(c["T_dh"]), T_mdh = as_list<Matrix4d>(c["T_mdh"]);
    const lab::DHTable dh = lab::course_arm_dh(), mdh = lab::course_arm_mdh();
    CHECK(dh.types.size() == 6);
    CHECK(mdh.types.size() == 6);
    CHECK(mdh.convention == "modified");
    for (Eigen::Index k = 0; k < n; ++k) {
        INFO("sample " << k);
        CHECK_MAT_NEAR(lab::fk_dh(dh, sample(k)), T_dh[static_cast<std::size_t>(k)], 1e-10);
        CHECK_MAT_NEAR(lab::fk_dh(mdh, sample(k)), T_mdh[static_cast<std::size_t>(k)], 1e-10);
    }
}

TEST_CASE("test_rtb_prismatic_rcm_and_rrp") {
    const json& c = cases()["rtb_prismatic_rcm_and_rrp"];
    const lab::DHTable rcm = lab::rcm_dh();
    const MatrixXd qs = as_mat(c["rcm_qs"]);
    const std::vector<Matrix4d> T_rcm = as_list<Matrix4d>(c["T_rcm"]);
    for (Eigen::Index i = 0; i < qs.rows(); ++i)
        CHECK_MAT_NEAR(lab::fk_dh(rcm, qs.row(i).transpose()), T_rcm[static_cast<std::size_t>(i)], 1e-10);
    CHECK_MAT_NEAR(lab::fk_dh(dh_rrp(), vec3(PI / 2, PI / 2, 1.0)).col(3).head<3>(), as_vec(c["rrp_p"]), 1e-9);
    const lab::DHTable mrob = lab::std_to_mod(rcm);
    const MatrixXd mq = as_mat(c["mdh_qs"]);
    const std::vector<Matrix4d> T_mdh = as_list<Matrix4d>(c["T_rcm_mdh"]);
    for (Eigen::Index i = 0; i < mq.rows(); ++i)
        CHECK_MAT_NEAR(lab::fk_dh(mrob, mq.row(i).transpose()), T_mdh[static_cast<std::size_t>(i)], 1e-10);
}

// ------------------------------------------------------------- Checkpoint 6: comparison helpers
TEST_CASE("test_read_pose_csv_and_pose_errors") {
    std::vector<Matrix4d> Ts;
    for (Eigen::Index k = 0; k < 20; ++k) Ts.push_back(mr::fkin_space(lab::course_arm_m(), lab::course_arm_slist(), sample(k)));
    const std::string f = temp_file("sbeg361_lab06_poses.csv");
    write_poses(f, Ts);
    const std::vector<Matrix4d> back = lab::read_pose_csv(f);
    REQUIRE(back.size() == 20);
    for (std::size_t k = 0; k < 20; ++k) CHECK_MAT_NEAR(back[k], Ts[k], 1e-15);
    {
        const auto [dp, dr] = lab::pose_errors(Ts, back);
        CHECK(dp < 1e-15);
        CHECK(dr < 1e-7);
    }
    std::vector<Matrix4d> pert = Ts;
    pert[7] = pert[7] * mr::rp_to_trans(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, 0.01))), Vector3d(0.002, 0, 0));
    const auto [dp, dr] = lab::pose_errors(Ts, pert);
    CHECK(std::abs(dp - 0.002) < 1e-12);
    CHECK(std::abs(dr - 0.01) < 1e-9);
    std::error_code ec;
    std::filesystem::remove(std::filesystem::temp_directory_path() / "sbeg361_lab06_poses.csv", ec);
}

TEST_CASE("test_time_fk_returns_seconds_per_call") {
    const MatrixXd S = lab::course_arm_slist();
    const Matrix4d M = lab::course_arm_m();
    const MatrixXd qs = q_samples().topRows(50);
    const double t = lab::time_fk([&](const VectorXd& q) { return mr::fkin_space(M, S, q); }, qs, 2);
    CHECK(t > 0.0);
    CHECK(t < 0.1);
    // Step 6b numbers for your table (printed even when the test passes)
    const MatrixXd q300 = q_samples().topRows(300);
    const lab::DHTable tab = lab::course_arm_dh();
    const double t_dh = lab::time_fk([&](const VectorXd& q) { return lab::fk_dh(tab, q); }, q300, 3);
    const double t_poe = lab::time_fk([&](const VectorXd& q) { return lab::fk_space(M, S, q); }, q300, 3);
    const double t_mr = lab::time_fk([&](const VectorXd& q) { return mr::fkin_space(M, S, q); }, q300, 3);
    MESSAGE("C++ time per FK call (300 samples): fk_dh " << t_dh * 1e6 << " us, fk_space " << t_poe * 1e6
                                                         << " us, mr::fkin_space " << t_mr * 1e6 << " us");
}

TEST_CASE("test_encoder_quantisation_checkpoint") {
    // Handout Step 6c: worst flange error of 1000 samples when every joint is off by one encoder count
    const lab::DHTable tab = lab::course_arm_dh();
    double err = 0.0;
    for (Eigen::Index k = 0; k < q_samples().rows(); ++k) {
        const VectorXd q = sample(k);
        const VectorXd qe = (q.array() + lab::ENC_RES).matrix();
        err = std::max(err, (lab::fk_dh(tab, qe).block<3, 1>(0, 3) - lab::fk_dh(tab, q).block<3, 1>(0, 3)).norm());
    }
    CHECK(std::abs(err - 0.001016) < 2e-6);
}

// ------------------------------------------------------------- C++ FK vs the Python reference
TEST_CASE("test_cpp_fk_matches_python") {
    // Python compiles the Part B program and compares its pose files with modern_robotics. The C++ track does the
    // same with YOUR C++ functions: FK of all 1000 samples by fk_dh (your table) and by the given fk_space, written
    // to pose files, read back with read_pose_csv, compared with mr::fkin_space by pose_errors, and the first
    // samples compared with the Python reference values.
    const MatrixXd S = lab::course_arm_slist();
    const Matrix4d M = lab::course_arm_m();
    const lab::DHTable tab = lab::course_arm_dh();
    std::vector<Matrix4d> ref, Tdh, Tpoe;
    for (Eigen::Index k = 0; k < q_samples().rows(); ++k) {
        ref.push_back(mr::fkin_space(M, S, sample(k)));
        Tdh.push_back(lab::fk_dh(tab, sample(k)));
        Tpoe.push_back(lab::fk_space(M, S, sample(k)));
    }
    const std::vector<Matrix4d> py = as_list<Matrix4d>(cases()["cpp_fk_matches_python"]["T"]);
    for (std::size_t k = 0; k < py.size(); ++k) CHECK_MAT_NEAR(Tdh[k], py[k], 1e-12);
    for (const char* cname : {"poses_dh_cpp.csv", "poses_poe_cpp.csv"}) {
        const std::string name = cname;
        INFO(name);
        const std::string f = temp_file("sbeg361_lab06_" + name);
        write_poses(f, name == "poses_dh_cpp.csv" ? Tdh : Tpoe);
        const std::vector<Matrix4d> back = lab::read_pose_csv(f);
        REQUIRE(back.size() == ref.size());
        const auto [dp, dr] = lab::pose_errors(ref, back);
        CHECK(dp < 1e-12);
        CHECK(dr < 1e-7);
        std::error_code ec;
        std::filesystem::remove(std::filesystem::temp_directory_path() / ("sbeg361_lab06_" + name), ec);
    }
    const MatrixXd q100 = q_samples().topRows(100);
    CHECK(lab::time_fk([&](const VectorXd& q) { return lab::fk_dh(tab, q); }, q100, 3) > 0.0);
    CHECK(lab::time_fk([&](const VectorXd& q) { return lab::fk_space(M, S, q); }, q100, 3) > 0.0);
}
