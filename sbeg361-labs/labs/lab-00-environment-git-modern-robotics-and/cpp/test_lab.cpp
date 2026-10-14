// Auto-checks for SBEG 361 Lab 0 (DOF, C-space, first contact with mrcpp) -- C++ track. Mirrors test_lab.py test
// by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*joint*,*grubler*"      (run only some tests, like pytest -k)
// Results are compared with mrcpp (the C++ port of modern_robotics) and with cpp_cases.json, which holds the
// numbers of the Python reference, so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix4d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::json;
using Strings = std::vector<std::string>;
using Counts = std::map<std::string, int>;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

std::string data_file() { return sbeg::lab_path("../data/mechanisms.json"); }

double radians(double deg) { return deg * PI / 180.0; }

/// Python math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol) with a readable message.
sbeg::NearResult isclose(double a, double b, double rel = 1e-9, double abs_tol = 0.0) {
    return sbeg::compare_near(a, b, std::max(rel * std::max(std::abs(a), std::abs(b)), abs_tol));
}

/// Python np.allclose(a, b) (rtol 1e-5, atol 1e-8 by default): entry by entry |a - b| <= atol + rtol * |b|.
template <typename A, typename B>
sbeg::NearResult allclose(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b, double atol = 1e-8,
                          double rtol = 1e-5) {
    const Eigen::MatrixXd ad = a.template cast<double>(), bd = b.template cast<double>();
    if (ad.rows() != bd.rows() || ad.cols() != bd.cols() || ad.size() == 0)
        return sbeg::compare_near(ad, bd, atol);  // shape message (or empty: ok)
    // message (max error, both values) from compare_near; the verdict is the per-entry NumPy rule
    sbeg::NearResult r = sbeg::compare_near(ad, bd, atol + rtol * bd.cwiseAbs().maxCoeff());
    const Eigen::ArrayXXd tol = atol + rtol * bd.array().abs();
    r.ok = ad.allFinite() && ((ad - bd).array().abs() <= tol).all();
    if (!r.ok) r.msg += "\n  (np.allclose rule: |got - expected| <= atol + rtol * |expected| for every entry)";
    return r;
}

/// Gruebler with the joints given in any of the three JSON forms: {type: count}, [type, ...] or [f_i, ...].
int grubler_json(int m, int N, const json& joints) {
    if (joints.is_object()) return lab::grubler(m, N, joints.get<Counts>());
    if (!joints.empty() && joints[0].is_string()) return lab::grubler(m, N, joints.get<Strings>());
    return lab::grubler(m, N, joints.get<std::vector<int>>());
}
}  // namespace

#define CHECK_NEAR_RESULT(expr)                       \
    do {                                              \
        const ::sbeg::NearResult sbeg_r_ = (expr);    \
        CHECK_MESSAGE(sbeg_r_.ok, #expr "\n  " << sbeg_r_.msg); \
    } while (false)

// ---------------------------------------------------------------- checkpoint 1: environment
TEST_CASE("test_env_report") {
    const std::map<std::string, std::string> rep = lab::env_report();
    for (const char* key : {"cpp_standard", "compiler", "eigen"}) {
        INFO("key = " << key);
        REQUIRE(rep.count(key) == 1);
        CHECK_FALSE(rep.at(key).empty());
    }
    for (const auto& kv : rep) CHECK_FALSE(kv.second.empty());
    CHECK(std::stol(rep.at("cpp_standard")) >= 201703L);  // C++17 or newer
}

// ---------------------------------------------------------------- checkpoint 2: joints and Gruebler
TEST_CASE("test_joint_freedoms") {
    for (const auto& c : cases()["joint_freedoms"]) {
        const std::string j = c["j"].get<std::string>();
        INFO("joint " << j);
        CHECK(lab::joint_freedoms(j) == c["f"].get<int>());
    }
}

TEST_CASE("test_joint_freedoms_unknown") {
    CHECK_THROWS_AS(lab::joint_freedoms("X"), std::invalid_argument);
}

TEST_CASE("test_joint_constraints_table_2_1") {
    for (const auto& c : cases()["joint_constraints_table_2_1"]) {
        const std::string j = c["j"].get<std::string>();
        const int m = c["m"].get<int>();
        INFO("joint " << j << ", m = " << m);
        CHECK(lab::joint_constraints(j, m) == c["c"].get<int>());
    }
}

TEST_CASE("test_joint_constraints_planar_na") {
    for (const auto& jj : cases()["joint_constraints_planar_na"]) {
        const std::string j = jj.get<std::string>();
        INFO("joint " << j);
        CHECK_THROWS_AS(lab::joint_constraints(j, 3), std::invalid_argument);
    }
}

TEST_CASE("test_grubler_mr_examples") {
    // MR Ex 2.3 four-bar and slider-crank, 2.4 five-bar, 2.5, 2.6 parallelogram, 2.7 Delta, 2.8 Stewart-Gough,
    // and a 6-SPS given as integers f_i.
    for (const auto& c : cases()["grubler_mr_examples"]) {
        const int m = c["m"].get<int>(), N = c["N"].get<int>();
        INFO("m = " << m << ", N = " << N << ", joints = " << c["joints"].dump());
        CHECK(grubler_json(m, N, c["joints"]) == c["dof"].get<int>());
    }
    // the same examples written directly with the three overloads
    CHECK(lab::grubler(3, 4, Counts{{"R", 4}}) == 1);
    CHECK(lab::grubler(3, 3, Strings{"R", "R", "RP"}) == 1);
    CHECK(lab::grubler(6, 14, Counts{{"U", 6}, {"P", 6}, {"S", 6}}) == 6);
    std::vector<int> sps(12, 3);
    sps.insert(sps.end(), 6, 1);
    CHECK(lab::grubler(6, 14, sps) == 12);
}

TEST_CASE("test_grubler_kR_chain") {
    for (int k = 1; k < 8; ++k) {
        INFO("k = " << k);
        CHECK(lab::grubler(3, k + 1, Strings(static_cast<std::size_t>(k), "R")) == k);  // MR Ex 2.4
    }
}

TEST_CASE("test_grubler_bad_m") {
    CHECK_THROWS_AS(lab::grubler(4, 3, Strings{"R", "R"}), std::invalid_argument);
}

// ---------------------------------------------------------------- checkpoint 3: mechanism table
TEST_CASE("test_dof_table") {
    const std::map<std::string, int> EXPECTED = {
        {"Four-bar linkage (MR Ex 2.3)", 1},
        {"Slider-crank (MR Ex 2.3)", 1},
        {"Five-bar linkage (MR Ex 2.4)", 2},
        {"Stephenson six-bar (MR Ex 2.4)", 1},
        {"Delta robot (MR Ex 2.7)", 15},
        {"Stewart-Gough 6-UPS (MR Ex 2.8)", 6},
        {"Four-bar (cruciate ligament) knee model", 1},
        {"Linkage-driven prosthetic finger", 1},
        {"Human arm, shoulder to palm (S-R-R-U)", 7},
        {"Hexapod fracture fixator 6-SPS", 12},
        {"Knee exoskeleton with self-aligning RP cuffs", 2},
        {"EOD robot 6R arm grasping a fixed handle", 0},
        {"EOD robot on free chassis, 6R arm grasping a fixed handle", 3},
        {"Casualty-litter scissor lift, two stages", 1},
        {"Dual 7R arms holding one package", 8},
    };
    const std::map<std::string, int> table = lab::dof_table(data_file());
    for (const auto& kv : EXPECTED) {  // one message per wrong entry
        const std::string& name = kv.first;
        INFO("mechanism: " << name);
        REQUIRE(table.count(name) == 1);
        CHECK(table.at(name) == kv.second);
    }
    CHECK(table == EXPECTED);
    CHECK(table == cases()["dof_table"].get<Counts>());
}

// ---------------------------------------------------------------- checkpoint 4: C-space topology
TEST_CASE("test_wrap_to_pi_scalar_and_array") {
    CHECK_NEAR_RESULT(isclose(lab::wrap_to_pi(3 * PI / 2), -PI / 2));
    CHECK_NEAR_RESULT(isclose(lab::wrap_to_pi(PI), -PI));  // [-pi, pi): pi maps to -pi
    const VectorXd x = VectorXd::LinSpaced(401, -20.0, 20.0);
    const VectorXd w = lab::wrap_to_pi(x);
    REQUIRE(w.size() == x.size());
    CHECK(w.minCoeff() >= -PI);
    CHECK(w.maxCoeff() < PI);
    CHECK_NEAR_RESULT(allclose(VectorXd(w.array().cos()), VectorXd(x.array().cos())));
    CHECK_NEAR_RESULT(allclose(VectorXd(w.array().sin()), VectorXd(x.array().sin())));
    const json& c = cases()["wrap_to_pi_scalar_and_array"];
    CHECK_MAT_NEAR(x, sbeg::as_vec(c["x"]), 1e-12);
    CHECK_MAT_NEAR(w, sbeg::as_vec(c["w"]), 1e-12);
}

TEST_CASE("test_torus_distance") {
    // 2R arm: 350 deg and 10 deg are 20 deg apart, not 340 deg
    double d = lab::torus_distance(Vector2d(radians(350), 0.0), Vector2d(radians(10), 0.0));
    CHECK_NEAR_RESULT(isclose(d, radians(20), 1e-9, 1e-12));
    d = lab::torus_distance(Vector3d(0.1, -3.0, 2.0), Vector3d(0.1 + 4 * PI, -3.0, 2.0));
    CHECK_NEAR_RESULT(isclose(d, 0.0, 1e-9, 1e-12));
    d = lab::torus_distance(Vector2d(3.0, 3.0), Vector2d(-3.0, -3.0));
    CHECK_NEAR_RESULT(isclose(d, std::sqrt(2.0) * (2 * PI - 6.0), 1e-9, 1e-12));
    for (const auto& c : cases()["torus_distance"]) {
        const double dc = lab::torus_distance(sbeg::as_vec(c["q1"]), sbeg::as_vec(c["q2"]));
        CHECK_SCALAR_NEAR(dc, c["d"].get<double>(), 1e-12);
    }
}

TEST_CASE("test_cspace_topology") {
    // P-R arm (MR Sec. 2.3.1), 2R arm, door, Cartesian gantry, radiotherapy couch (illustrative), 6R arm
    for (const auto& c : cases()["cspace_topology"]) {
        const Strings joints = c["joints"].get<Strings>();
        INFO("joints = " << c["joints"].dump());
        CHECK(lab::cspace_topology(joints) == c["topo"].get<std::string>());
    }
}

TEST_CASE("test_cspace_topology_rejects") {
    CHECK_THROWS_AS(lab::cspace_topology(Strings{"S"}), std::invalid_argument);
    CHECK_THROWS_AS(lab::cspace_topology(Strings{}), std::invalid_argument);
}

// ---------------------------------------------------------------- checkpoint 5: representation
TEST_CASE("test_latlong_roundtrip") {
    // 200 random (lat, lon) pairs: the Python test draws them with np.random.default_rng(0); they are stored in
    // cpp_cases.json because C++ cannot reproduce NumPy's random numbers.
    const json& all = cases()["latlong_roundtrip"];
    REQUIRE(all.size() == 200);
    for (const auto& c : all) {
        const double lat = c["lat"].get<double>(), lon = c["lon"].get<double>();
        INFO("lat = " << lat << ", lon = " << lon);
        const Vector3d p = lab::latlong_to_xyz(lat, lon);
        CHECK_NEAR_RESULT(isclose(p.norm(), 1.0, 1e-9, 1e-12));
        CHECK_MAT_NEAR(p, as<Vector3d>(c["p"]), 1e-12);
        const auto [la, lo] = lab::xyz_to_latlong(Vector3d(3.0 * p));
        CHECK_NEAR_RESULT(isclose(la, lat, 1e-9, 1e-9));
        CHECK_NEAR_RESULT(isclose(lab::wrap_to_pi(lo - lon), 0.0, 1e-9, 1e-9));
    }
}

TEST_CASE("test_longitude_rate_singularity") {
    const double w = 0.2;
    CHECK_NEAR_RESULT(isclose(lab::longitude_rate(0.0, w), 0.2));
    CHECK_NEAR_RESULT(isclose(lab::longitude_rate(radians(60), w), 0.4, 1e-12));
    CHECK_NEAR_RESULT(isclose(lab::longitude_rate(radians(89), w), 11.4597, 1e-4));
    // the Cartesian speed is the same everywhere: compare with finite differences
    for (const double deg : {0.0, 45.0, 80.0, 89.0, 89.9}) {
        INFO("latitude = " << deg << " deg");
        const double lat = radians(deg);
        const double lr = lab::longitude_rate(lat, w);
        const double h = 1e-7;
        const Vector3d v = (lab::latlong_to_xyz(lat, h * lr) - lab::latlong_to_xyz(lat, 0.0)) / h;
        CHECK_NEAR_RESULT(isclose(v.norm(), w, 1e-5));
    }
    for (const auto& c : cases()["longitude_rate_singularity"]) {
        const double lat = c["lat"].get<double>();
        INFO("lat = " << lat);
        CHECK_NEAR_RESULT(isclose(lab::longitude_rate(lat, c["speed"].get<double>()), c["rate"].get<double>(), 1e-12));
    }
}

// ---------------------------------------------------------------- checkpoint 6: mrcpp (modern_robotics)
TEST_CASE("test_fk_2r_against_modern_robotics") {
    const json& all = cases()["fk_2r_against_modern_robotics"];
    const double L1 = 0.6, L2 = 0.4;
    Eigen::MatrixXd Slist(6, 2);
    Slist.col(0) << 0, 0, 1, 0, 0, 0;
    Slist.col(1) << 0, 0, 1, 0, -L1, 0;
    Matrix4d M = Matrix4d::Identity();
    M(0, 3) = L1 + L2;
    // 50 random configurations (np.random.default_rng(1) in Python), stored in cpp_cases.json
    REQUIRE(all["cases"].size() == 50);
    for (const auto& c : all["cases"]) {
        const Vector2d th = as<Vector2d>(c["theta"]);
        INFO("theta = (" << th(0) << ", " << th(1) << ")");
        const Matrix4d T = lab::fk_2r_mr(th, L1, L2);
        CHECK_NEAR_RESULT(allclose(T, mr::fkin_space(M, Slist, th), 1e-12));
        CHECK_NEAR_RESULT(allclose(T, as<Matrix4d>(c["T"]), 1e-12));
        CHECK_NEAR_RESULT(allclose(Vector2d(T.block<2, 1>(0, 3)), lab::fk_2r_trig(th, L1, L2), 1e-12));
        CHECK_NEAR_RESULT(allclose(lab::fk_2r_trig(th, L1, L2), as<Vector2d>(c["xy"]), 1e-12));
        CHECK_NEAR_RESULT(isclose(T(2, 3), 0.0, 1e-9, 1e-8));  // np.isclose(T[2, 3], 0.0)
        // orientation is a rotation by theta1 + theta2 about z
        const double phi = th(0) + th(1);
        Matrix2d Rz;
        Rz << std::cos(phi), -std::sin(phi), std::sin(phi), std::cos(phi);
        CHECK_NEAR_RESULT(allclose(Matrix2d(T.topLeftCorner<2, 2>()), Rz));
    }
}

TEST_CASE("test_fk_2r_known_values") {
    CHECK_NEAR_RESULT(allclose(lab::fk_2r_trig(Vector2d(0.0, 0.0)), Vector2d(1.0, 0.0)));
    CHECK_NEAR_RESULT(allclose(lab::fk_2r_trig(Vector2d(PI / 2, -PI / 2)), Vector2d(0.4, 0.6)));
    CHECK_NEAR_RESULT(
        allclose(lab::fk_2r_trig(Vector2d(PI / 6, PI / 3)), Vector2d(0.6 * std::cos(PI / 6), 0.3 + 0.4)));
    for (const auto& c : cases()["fk_2r_known_values"])
        CHECK_MAT_NEAR(lab::fk_2r_trig(as<Vector2d>(c["theta"])), as<Vector2d>(c["xy"]), 1e-12);
}
