// Auto-checks for SBEG 361 Lab 4 (exponential coordinates on SE(3), wrenches, F/T sensing) -- C++ track.
// Mirrors test_lab.py test by test (20 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*exp6*,*log6*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: matrix_exp6, matrix_log6, adjoint, trans_inv),
// MR Example 3.26 (pp. 105-106) and Example 3.28 (pp. 108-109), and the noise-free data model of make_data.py.
// cpp_cases.json holds the random inputs of test_lab.py and the numbers of the Python reference, so both
// languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::json;
using sbeg::Matrix6d;
using sbeg::Vector6d;

namespace {
constexpr double PI = 3.14159265358979323846;
const double S2 = 1.0 / std::sqrt(2.0);

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

std::string lab_file(const std::string& name) { return sbeg::lab_path("../" + name); }

Vector6d v6(double a, double b, double c, double d, double e, double f) {
    Vector6d V;
    V << a, b, c, d, e, f;
    return V;
}

Matrix4d mr_rp(const Matrix3d& R, const Vector3d& p) { return mr::rp_to_trans(R, p); }

/// make_data.press_profile: axial tissue force (N), three sin^2 presses of 2 s starting at 1, 4, 7 s; peak 2 N.
VectorXd press_profile(const VectorXd& t) {
    VectorXd F = VectorXd::Zero(t.size());
    for (const double t0 : {1.0, 4.0, 7.0})
        for (Eigen::Index k = 0; k < t.size(); ++k)
            if (t(k) >= t0 && t(k) <= t0 + 2.0) {
                const double s = std::sin(PI * (t(k) - t0) / 2.0);
                F(k) = 2.0 * s * s;
            }
    return F;
}

/// make_data.gravity(mass, com, R): gravity wrench in {f} (noise-free data model).
Vector6d md_gravity(double mass, const Vector3d& com, const Matrix3d& R) {
    const Vector3d f = mass * R.transpose() * Vector3d(0.0, 0.0, -9.81);
    Vector6d F;
    F << com.cross(f), f;
    return F;
}

std::vector<Matrix3d> rotations_of(const std::vector<Matrix4d>& Ts) {
    std::vector<Matrix3d> Rs;
    for (const Matrix4d& T : Ts) Rs.push_back(T.topLeftCorner<3, 3>());
    return Rs;
}

/// The laparoscope calibration of test_lab.py (lap_calibration()).
lab::PayloadEstimate lap_calibration() {
    const lab::PoseWrenchData d = lab::load_pose_wrench_csv(lab_file("lap_calib.csv"));
    return lab::estimate_payload(rotations_of(d.Ts), d.Fs);
}

/// T_TIP of test_lab.py: T_ft = (I, (0, 0, 0.30)), the tool tip {t} seen from the sensor {f}.
Matrix4d t_tip() { return as<Matrix4d>(cases()["tip_force_during_palpation"]["T_ft"]); }

struct EodEstimates {
    json d;
    lab::PayloadEstimate g, t;  // empty gripper, gripper + package
};

/// eod_estimates() of test_lab.py.
EodEstimates eod_estimates() {
    EodEstimates e;
    e.d = lab::load_json(lab_file("eod_lift.json"));
    std::vector<Matrix3d> Rs;
    for (const auto& x : e.d["empty"]) Rs.push_back(as<Matrix4d>(x["T_sf"]).topLeftCorner<3, 3>());
    MatrixXd Fe(static_cast<Eigen::Index>(e.d["empty"].size()), 6), Fl(static_cast<Eigen::Index>(e.d["loaded"].size()), 6);
    for (Eigen::Index k = 0; k < Fe.rows(); ++k) Fe.row(k) = as<Vector6d>(e.d["empty"][static_cast<std::size_t>(k)]["F"]).transpose();
    for (Eigen::Index k = 0; k < Fl.rows(); ++k) Fl.row(k) = as<Vector6d>(e.d["loaded"][static_cast<std::size_t>(k)]["F"]).transpose();
    e.g = lab::estimate_payload(Rs, Fe);
    e.t = lab::estimate_payload(Rs, Fl);
    return e;
}
}  // namespace

// ===================================================== Checkpoint 1: exp and log on SE(3)
TEST_CASE("test_matrix_exp6_matches_mr") {
    for (const auto& c : cases()["matrix_exp6_matches_mr"]) {
        const Matrix4d se3 = as<Matrix4d>(c["se3"]);
        CHECK_MAT_NEAR(lab::matrix_exp6(se3), mr::matrix_exp6(se3), 1e-9);
        CHECK_MAT_NEAR(lab::matrix_exp6(se3), as<Matrix4d>(c["T"]), 1e-9);
    }
    // pure translation, MR Eq. 3.89
    const Matrix4d se3 = mr::vec_to_se3(v6(0, 0, 0, 0.03, 0.04, 0));
    CHECK_MAT_NEAR(lab::matrix_exp6(se3), mr_rp(Matrix3d::Identity(), Vector3d(0.03, 0.04, 0)), 1e-12);
}

TEST_CASE("test_exp6_screw_through_a_point") {
    // Screw through a point (Week 4 revision): axis z through q = (1, 0, 0), theta = 90 deg, pitch 0 and 0.1 m/rad.
    for (const double h : {0.0, 0.1}) {
        INFO("pitch h = " << h);
        const double pz = h * PI / 2;
        const Vector3d w(0, 0, 1);
        Vector6d S;
        S << w, -w.cross(Vector3d(1, 0, 0)) + h * w;
        const Matrix4d T = lab::matrix_exp6(mr::vec_to_se3(S) * (PI / 2));
        Matrix4d T_exp;
        T_exp << 0, -1, 0, 1, 1, 0, 0, -1, 0, 0, 1, pz, 0, 0, 0, 1;
        CHECK_MAT_NEAR(T, T_exp, 1e-12);
    }
}

TEST_CASE("test_matrix_log6_matches_mr_and_inverts_exp") {
    for (const auto& c : cases()["matrix_log6_matches_mr_and_inverts_exp"]) {
        const Matrix4d T = as<Matrix4d>(c["T"]);
        const Matrix4d L = lab::matrix_log6(T);
        CHECK_MAT_NEAR(L.row(3), Eigen::RowVector4d::Zero(), 1e-8);  // np.allclose(L[3], 0)
        CHECK_MAT_NEAR(L, mr::matrix_log6(T), 1e-8);
        CHECK_MAT_NEAR(L, as<Matrix4d>(c["L"]), 1e-8);
        CHECK_MAT_NEAR(mr::matrix_exp6(L), T, 1e-9);
    }
}

TEST_CASE("test_matrix_log6_special_cases") {
    CHECK_MAT_NEAR(lab::matrix_log6(Matrix4d::Identity()), Matrix4d::Zero(), 1e-12);
    const Matrix4d T = mr_rp(Matrix3d::Identity(), Vector3d(0.4, 0, 0.3));  // case (a): R = I
    CHECK_MAT_NEAR(lab::se3_to_vec(lab::matrix_log6(T)), v6(0, 0, 0, 0.4, 0, 0.3), 1e-12);
    const Vector3d diags[3] = {Vector3d(1, -1, -1), Vector3d(-1, 1, -1), Vector3d(-1, -1, 1)};  // theta = pi
    for (const Vector3d& dg : diags) {
        INFO("R = diag(" << dg.transpose() << ")");
        const Matrix3d R = dg.asDiagonal();
        const Matrix4d Tp = mr_rp(R, Vector3d(0.1, 0.2, -0.3));
        const Matrix4d L = lab::matrix_log6(Tp);
        const Matrix3d so3 = L.topLeftCorner<3, 3>();
        CHECK_SCALAR_NEAR(lab::so3_to_vec(so3).norm(), PI, 1e-9);
        CHECK_MAT_NEAR(mr::matrix_exp6(L), Tp, 1e-9);
    }
}

TEST_CASE("test_log6_mr_example_3_26") {
    // MR Example 3.26 (pp. 104-106): S = (0, 0, 1, 3.37, -3.37, 0), theta = pi/6.
    const double c30 = std::cos(PI / 6), s30 = 0.5, c60 = 0.5, s60 = std::sin(PI / 3);
    Matrix4d Tsb, Tsc;
    Tsb << c30, -s30, 0, 1, s30, c30, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1;
    Tsc << c60, -s60, 0, 2, s60, c60, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
    const auto [S, theta] = lab::screw_between(Tsb, Tsc, "space");
    CHECK_SCALAR_NEAR(theta, PI / 6, 1e-9);
    CHECK_MAT_NEAR(S, v6(0, 0, 1, 3.3660, -3.3660, 0), 1e-4);
    const json& c = cases()["log6_mr_example_3_26"];
    CHECK_MAT_NEAR(S, as<Vector6d>(c["S"]), 1e-9);
}

TEST_CASE("test_screw_between_space_and_body") {
    for (const auto& c : cases()["screw_between_space_and_body"]) {
        const Matrix4d T1 = as<Matrix4d>(c["T1"]), T2 = as<Matrix4d>(c["T2"]);
        const auto [Ss, ts] = lab::screw_between(T1, T2, "space");
        const auto [Sb, tb] = lab::screw_between(T1, T2, "body");
        CHECK_SCALAR_NEAR(Ss.head<3>().norm(), 1.0, 1e-9);
        CHECK_SCALAR_NEAR(ts, tb, 1e-9);
        CHECK_MAT_NEAR(Matrix4d(mr::matrix_exp6(mr::vec_to_se3(Ss) * ts) * T1), T2, 1e-9);
        CHECK_MAT_NEAR(Matrix4d(T1 * mr::matrix_exp6(mr::vec_to_se3(Sb) * tb)), T2, 1e-9);
        CHECK_MAT_NEAR(Sb, Vector6d(mr::adjoint(mr::trans_inv(T1)) * Ss), 1e-9);  // S_b = [Ad_{T1^-1}] S_s
        CHECK_MAT_NEAR(Ss, as<Vector6d>(c["S_s"]), 1e-9);
        CHECK_SCALAR_NEAR(ts, c["theta_s"].get<double>(), 1e-9);
    }
    const auto [S, th] = lab::screw_between(Matrix4d::Identity(), mr_rp(Matrix3d::Identity(), Vector3d(0, 0.3, 0.4)));
    CHECK_MAT_NEAR(S, v6(0, 0, 0, 0, 0.6, 0.8), 1e-12);
    CHECK_SCALAR_NEAR(th, 0.5, 1e-12);
    CHECK_THROWS_AS(lab::screw_between(Matrix4d::Identity(), Matrix4d::Identity(), "world"), std::invalid_argument);
}

TEST_CASE("test_screw_path_needle_guide") {
    // Needle-guide screw path (Lab 3 / Week 4 revision): half-way along the screw from T1 to T2.
    const Matrix4d T1 = mr_rp(Matrix3d::Identity(), Vector3d(0.3, 0, 0.1));
    const Matrix4d T2 = mr_rp(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, PI / 2))), Vector3d(0.2, 0.3, 0.15));
    CHECK_MAT_NEAR(lab::screw_path(T1, T2, 0.0), T1, 1e-12);
    CHECK_MAT_NEAR(lab::screw_path(T1, T2, 1.0), T2, 1e-12);
    const Matrix4d Tm = lab::screw_path(T1, T2, 0.5);
    CHECK_MAT_NEAR(Vector3d(Tm.topRightCorner<3, 1>()), Vector3d(0.1 + 0.3 * S2, 0.1 + 0.1 * S2, 0.125), 1e-9);
    CHECK_MAT_NEAR(Tm, as<Matrix4d>(cases()["screw_path_needle_guide"]["T_half"]), 1e-9);
}

// ===================================================== Checkpoint 2: wrenches and the adjoint transpose
TEST_CASE("test_point_force_wrench_and_power") {
    const Vector6d F = lab::point_force_wrench(Vector3d(1, 0, -2), Vector3d(0, 0, 0.3));
    CHECK_MAT_NEAR(F, v6(0, 0.3, 0, 1, 0, -2), 1e-12);
    // Python also checks that power returns a float: in C++ the return type is double by declaration.
    CHECK(lab::power(v6(1, 2, 3, 4, 5, 6), v6(1, 0, 0, 0, 0, 1)) == 7.0);
}

TEST_CASE("test_change_wrench_frame_mr_example_3_28") {
    // MR Example 3.28 (pp. 108-109): F_f = [Ad_Thf]^T F_h + [Ad_Taf]^T F_a = (0, 0, -0.75, 0, -6, 0).
    Matrix4d Thf, Taf;
    Thf << 1, 0, 0, -0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    Taf << 1, 0, 0, -0.25, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    const Vector6d Ff = lab::change_wrench_frame(v6(0, 0, 0, 0, -5, 0), Thf) +
                        lab::change_wrench_frame(v6(0, 0, 0, 0, 0, 1), Taf);
    CHECK_MAT_NEAR(Ff, v6(0, 0, -0.75, 0, -6, 0), 1e-12);
}

TEST_CASE("test_power_is_frame_independent") {
    for (const auto& c : cases()["power_is_frame_independent"]) {
        const Matrix4d T_ab = as<Matrix4d>(c["T_ab"]);
        const Vector6d F_a = as<Vector6d>(c["F_a"]), V_b = as<Vector6d>(c["V_b"]);
        const Vector6d F_b = lab::change_wrench_frame(F_a, T_ab);
        CHECK_MAT_NEAR(F_b, Vector6d(mr::adjoint(T_ab).transpose() * F_a), 1e-12);
        CHECK_MAT_NEAR(F_b, as<Vector6d>(c["F_b"]), 1e-12);
        const Vector6d V_a = mr::adjoint(T_ab) * V_b;
        CHECK_SCALAR_NEAR(lab::power(V_b, F_b), lab::power(V_a, F_a), 1e-9);  // MR Eq. 3.94
        CHECK_MAT_NEAR(lab::change_wrench_frame(F_b, mr::trans_inv(T_ab)), F_a, 1e-9);
    }
}

TEST_CASE("test_wrench_to_screw") {
    lab::WrenchScrew w = lab::wrench_to_screw(v6(0, 5.886, 0, 0, 0, -19.62));  // Workshop 4b
    CHECK(std::abs(w.h) < 1e-12);
    CHECK_SCALAR_NEAR(w.magnitude, 19.62, 1e-12);
    REQUIRE(w.q.has_value());
    CHECK_MAT_NEAR(*w.q, Vector3d(0.3, 0, 0), 1e-12);
    CHECK_MAT_NEAR(w.s_hat, Vector3d(0, 0, -1), 1e-12);
    for (const auto& c : cases()["wrench_to_screw"]) {  // recompose and invariance
        const Vector6d F = as<Vector6d>(c["F"]);
        const lab::WrenchScrew s = lab::wrench_to_screw(F);
        REQUIRE(s.q.has_value());
        const Vector3d q = *s.q;
        Vector6d Fr;
        Fr << q.cross(s.magnitude * s.s_hat) + s.h * s.magnitude * s.s_hat, s.magnitude * s.s_hat;
        CHECK_MAT_NEAR(Fr, F, 1e-9);
        CHECK_SCALAR_NEAR(s.h, c["h"].get<double>(), 1e-9);
        const double h2 = lab::wrench_to_screw(lab::change_wrench_frame(F, as<Matrix4d>(c["T"]))).h;
        CHECK_SCALAR_NEAR(s.h, h2, 1e-9);  // pitch is frame invariant
    }
    w = lab::wrench_to_screw(v6(0, 0, 2, 0, 0, 0));
    CHECK(w.h == std::numeric_limits<double>::infinity());
    CHECK_FALSE(w.q.has_value());
    CHECK(w.magnitude == 2.0);
    CHECK_THROWS_AS(lab::wrench_to_screw(Vector6d::Zero()), std::invalid_argument);
}

TEST_CASE("test_gravity_wrench_workshop_3") {
    // Workshop 3a: 0.6 kg tool, CoM (0, 0, 0.10) in {f}, R_sf = Rot(x, 120 deg).
    const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(Vector3d(2 * PI / 3, 0, 0)));
    const Vector6d F = lab::gravity_wrench(0.6, Vector3d(0, 0, 0.10), R);
    const double s = std::sin(2 * PI / 3);
    CHECK_MAT_NEAR(F, v6(0.6 * 9.81 * s * 0.1, 0, 0, 0, -0.6 * 9.81 * s, 0.6 * 9.81 * 0.5), 1e-12);
    CHECK_MAT_NEAR(F, as<Vector6d>(cases()["gravity_wrench_workshop_3"]["F"]), 1e-12);
}

// ===================================================== Checkpoint 3: laparoscope -- calibration and tip force
TEST_CASE("test_estimate_payload_exact_data") {
    const json& c = cases()["estimate_payload_exact_data"];
    const double m = c["mass"].get<double>();
    const Vector3d cm = as<Vector3d>(c["c"]);
    const Vector6d b = as<Vector6d>(c["bias"]);
    const std::vector<Matrix3d> Rs = sbeg::as_list<Matrix3d>(c["R_list"]);
    MatrixXd Fs(static_cast<Eigen::Index>(Rs.size()), 6);
    for (std::size_t k = 0; k < Rs.size(); ++k) {
        const Vector3d g = m * Rs[k].transpose() * lab::G_S;
        Vector6d F;
        F << cm.cross(g), g;
        Fs.row(static_cast<Eigen::Index>(k)) = (b + F).transpose();
    }
    const auto [m_e, c_e, b_e] = lab::estimate_payload(Rs, Fs);
    CHECK_SCALAR_NEAR(m_e, m, 1e-9);
    CHECK_MAT_NEAR(c_e, cm, 1e-9);
    CHECK_MAT_NEAR(b_e, b, 1e-9);
}

TEST_CASE("test_laparoscope_calibration") {
    const auto [m, r, b] = lap_calibration();
    const json& c = cases()["laparoscope_calibration"];
    CHECK_SCALAR_NEAR(m, c["LAP_MASS"].get<double>(), 0.005);      // 0.600 kg
    CHECK_MAT_NEAR(r, as<Vector3d>(c["LAP_COM"]), 0.002);          // (0, 0, 0.10) m
    CHECK_MAT_NEAR(b, as<Vector6d>(c["LAP_BIAS"]), 0.01);
    CHECK_SCALAR_NEAR(m, c["mass"].get<double>(), 1e-6);           // Python reference on the same data
    CHECK_MAT_NEAR(r, as<Vector3d>(c["r_com"]), 1e-6);
    CHECK_MAT_NEAR(b, as<Vector6d>(c["bias"]), 1e-6);
}

TEST_CASE("test_compensate_removes_gravity_and_bias") {
    const auto [m, r, b] = lap_calibration();
    const lab::PoseWrenchData d = lab::load_pose_wrench_csv(lab_file("lap_calib.csv"));
    REQUIRE(d.Ts.size() == static_cast<std::size_t>(d.Fs.rows()));
    for (std::size_t k = 0; k < d.Ts.size(); ++k) {
        INFO("reading " << k);
        const Vector6d F = d.Fs.row(static_cast<Eigen::Index>(k)).transpose();
        const Vector6d Fc = lab::compensate(F, m, r, d.Ts[k].topLeftCorner<3, 3>(), b);
        CHECK(Fc.tail<3>().cwiseAbs().maxCoeff() < 0.08);
        CHECK(Fc.head<3>().cwiseAbs().maxCoeff() < 0.006);
    }
}

TEST_CASE("test_tip_force_during_palpation") {
    const auto [m, r, b] = lap_calibration();
    const lab::PoseWrenchData d = lab::load_pose_wrench_csv(lab_file("lap_palpation.csv"));
    const MatrixXd Ft = lab::tip_forces(d.Ts, d.Fs, m, r, b, t_tip());
    REQUIRE(Ft.rows() == d.t.size());
    REQUIRE(Ft.cols() == 6);
    // the C++ copy of make_data.press_profile agrees with Python
    const json& pc = cases()["tip_force_during_palpation"]["press_check"];
    CHECK_MAT_NEAR(press_profile(sbeg::as_vec(pc["t"])), sbeg::as_vec(pc["F"]), 1e-12);
    const VectorXd F = press_profile(d.t);  // true tip force f_t = (0.15 F, -0.05 F, -F) in {t}
    double sum = 0.0, peak = 0.0;
    int n = 0;
    for (Eigen::Index k = 0; k < d.t.size(); ++k) {
        if (!(d.t(k) < 7.0)) continue;  // presses 1 and 2 (no port force)
        const Vector3d f_true(0.15 * F(k), -0.05 * F(k), -F(k));
        const Vector3d f = Ft.row(k).tail<3>().transpose();
        sum += (f - f_true).squaredNorm();
        peak = std::max(peak, f.norm());
        ++n;
    }
    REQUIRE(n > 0);
    CHECK(std::sqrt(sum / n) < 0.06);
    CHECK_SCALAR_NEAR(peak, Vector3d(0.3, -0.1, -2.0).norm(), 0.1);  // 2.025 N
}

TEST_CASE("test_port_force_detected_by_tip_moment") {
    // A single point contact at the tip produces m_t = 0; the third press also has a port force.
    const auto [m, r, b] = lap_calibration();
    const lab::PoseWrenchData d = lab::load_pose_wrench_csv(lab_file("lap_palpation.csv"));
    const MatrixXd Ft = lab::tip_forces(d.Ts, d.Fs, m, r, b, t_tip());
    REQUIRE(Ft.rows() == d.t.size());
    REQUIRE(Ft.cols() == 6);
    const VectorXd F = press_profile(d.t);
    double s12 = 0.0, s3 = 0.0;
    int n12 = 0, n3 = 0;
    for (Eigen::Index k = 0; k < d.t.size(); ++k) {
        if (!(F(k) > 1.5)) continue;
        const double mt = Ft.row(k).head<3>().norm();
        if (d.t(k) < 7) {
            s12 += mt;
            ++n12;
        } else {
            s3 += mt;
            ++n3;
        }
    }
    REQUIRE(n12 > 0);
    REQUIRE(n3 > 0);
    CHECK(s12 / n12 < 0.015);
    CHECK(s3 / n3 > 0.04);
}

// ===================================================== Checkpoint 4: EOD gripper and package
TEST_CASE("test_eod_gripper_and_package_identification") {
    const EodEstimates e = eod_estimates();
    const json& c = cases()["eod_gripper_and_package_identification"];
    CHECK_SCALAR_NEAR(e.g.mass, c["GRIPPER_MASS"].get<double>(), 0.02);
    CHECK_MAT_NEAR(e.g.r_com, as<Vector3d>(c["GRIPPER_COM"]), 0.003);
    const double mp = e.t.mass - e.g.mass;
    const Vector3d rp = (e.t.mass * e.t.r_com - e.g.mass * e.g.r_com) / mp;  // composite centre of mass
    CHECK_SCALAR_NEAR(mp, c["PACKAGE_MASS"].get<double>(), 0.02);
    CHECK_MAT_NEAR(rp, as<Vector3d>(c["PACKAGE_COM"]), 0.005);
}

TEST_CASE("test_eod_wrench_at_turret_and_front_edge") {
    const EodEstimates e = eod_estimates();
    const Matrix4d T_sf = as<Matrix4d>(e.d["lift"]["T_sf"]);
    const Vector6d Fp = lab::compensate(as<Vector6d>(e.d["lift"]["F"]), e.g.mass, e.g.r_com,
                                        T_sf.topLeftCorner<3, 3>(), e.t.bias);  // package wrench in {f}
    const Vector6d F_s = lab::change_wrench_frame(Fp, mr::trans_inv(T_sf));       // F_s = [Ad_{T_fs}]^T F_f
    const json& c = cases()["eod_gripper_and_package_identification"];
    const Vector6d exact = mr::adjoint(mr::trans_inv(T_sf)).transpose() *
                           md_gravity(c["PACKAGE_MASS"].get<double>(), as<Vector3d>(c["PACKAGE_COM"]),
                                      T_sf.topLeftCorner<3, 3>());
    CHECK_MAT_NEAR(exact, as<Vector6d>(cases()["eod_wrench_at_turret_and_front_edge"]["exact_F_s"]), 1e-9);
    CHECK_MAT_NEAR(F_s, exact, 0.4);
    CHECK_SCALAR_NEAR(F_s(1), 37.1, 0.5);  // pitching moment at the turret base
    const Vector6d F_e = lab::change_wrench_frame(F_s, as<Matrix4d>(e.d["front_edge"]["T_se"]));
    CHECK_SCALAR_NEAR(F_e(1), 26.3, 0.5);  // about the front track edge
}

TEST_CASE("test_eod_power_duality") {
    const EodEstimates e = eod_estimates();
    const Matrix4d T_sf = as<Matrix4d>(e.d["lift"]["T_sf"]);
    const Vector6d Fp = lab::compensate(as<Vector6d>(e.d["lift"]["F"]), e.g.mass, e.g.r_com,
                                        T_sf.topLeftCorner<3, 3>(), e.t.bias);
    const Vector6d V_f = as<Vector6d>(e.d["lift"]["V_f"]);
    const double P_f = lab::power(V_f, Fp);
    const double P_s = lab::power(mr::adjoint(T_sf) * V_f, lab::change_wrench_frame(Fp, mr::trans_inv(T_sf)));
    CHECK_SCALAR_NEAR(P_f, P_s, 1e-9);
    CHECK(P_f > -7.3);  // gravity absorbs about 7 W while lifting
    CHECK(P_f < -6.8);
}
