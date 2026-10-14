// Self-test of the modern_robotics C++ port: every function against mrcpp_cases.json
// (inputs and outputs computed by the Python library with make_mrcpp_cases.py).
//
// Tolerance: 1e-9 (absolute, entry by entry) everywhere, except
//   * ikin_body / ikin_space when the Python IK FAILED (unreachable target): after 20 diverging Newton steps the
//     iterates are not reproducible to round-off, so only `success == false` is checked;
//   * forward_dynamics_trajectory / simulate_control: 1e-8 (round-off grows through the Euler integration and the
//     inverse of the mass matrix; observed differences are ~1e-13).
#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <string>

using sbeg::as;
using sbeg::as_list;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;
using namespace mr;

namespace {
constexpr double TOL = 1e-9;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("mrcpp_cases.json"));
    return C;
}

const json& of(const std::string& fn) {
    const json& C = cases();
    REQUIRE_MESSAGE(C.contains(fn), "no cases for " << fn << " in mrcpp_cases.json");
    REQUIRE(!C.at(fn).empty());
    return C.at(fn);
}

struct Chain {
    TransformList Mlist;
    InertiaList Glist;
    MatrixXd Slist;
};
Chain chain(const json& in) {
    return {as_list<Matrix4d>(in.at("Mlist")), as_list<Matrix6d>(in.at("Glist")), as_mat(in.at("Slist"))};
}
}  // namespace

TEST_CASE("every Python function has cases") {
    const json& C = cases();
    const json& names = C.at("_meta").at("python_functions");
    CHECK(names.size() == 47);
    // 47 Python functions + the pinv helper
    CHECK(C.size() - 1 == 48);
}

// ------------------------------------------------------------------ helpers
TEST_CASE("near_zero, normalize, pinv") {
    for (const auto& c : of("near_zero")) CHECK(near_zero(c["in"]["z"].get<double>()) == c["out"].get<bool>());
    for (const auto& c : of("normalize")) CHECK_MAT_NEAR(normalize(as_vec(c["in"]["V"])), as_vec(c["out"]), TOL);
    for (const auto& c : of("pinv")) CHECK_MAT_NEAR(pinv(as_mat(c["in"]["A"])), as_mat(c["out"]), TOL);
}

// ------------------------------------------------------------------ chapter 3
TEST_CASE("rot_inv, vec_to_so3, so3_to_vec, axis_ang3") {
    for (const auto& c : of("rot_inv")) CHECK_MAT_NEAR(rot_inv(as<Matrix3d>(c["in"]["R"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("vec_to_so3"))
        CHECK_MAT_NEAR(vec_to_so3(as<Vector3d>(c["in"]["omg"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("so3_to_vec"))
        CHECK_MAT_NEAR(so3_to_vec(as<Matrix3d>(c["in"]["so3mat"])), as_vec(c["out"]), TOL);
    for (const auto& c : of("axis_ang3")) {
        const auto [w, th] = axis_ang3(as<Vector3d>(c["in"]["expc3"]));
        CHECK_MAT_NEAR(w, as_vec(c["out"]["omghat"]), TOL);
        CHECK_SCALAR_NEAR(th, c["out"]["theta"].get<double>(), TOL);
    }
}

TEST_CASE("matrix_exp3, matrix_log3") {
    for (const auto& c : of("matrix_exp3"))
        CHECK_MAT_NEAR(matrix_exp3(as<Matrix3d>(c["in"]["so3mat"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("matrix_log3"))
        CHECK_MAT_NEAR(matrix_log3(as<Matrix3d>(c["in"]["R"])), as_mat(c["out"]), TOL);
}

TEST_CASE("rp_to_trans, trans_to_rp, trans_inv, vec_to_se3, se3_to_vec, adjoint, screw_to_axis") {
    for (const auto& c : of("rp_to_trans"))
        CHECK_MAT_NEAR(rp_to_trans(as<Matrix3d>(c["in"]["R"]), as<Vector3d>(c["in"]["p"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("trans_to_rp")) {
        const auto [R, p] = trans_to_rp(as<Matrix4d>(c["in"]["T"]));
        CHECK_MAT_NEAR(R, as_mat(c["out"]["R"]), TOL);
        CHECK_MAT_NEAR(p, as_vec(c["out"]["p"]), TOL);
    }
    for (const auto& c : of("trans_inv"))
        CHECK_MAT_NEAR(trans_inv(as<Matrix4d>(c["in"]["T"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("vec_to_se3"))
        CHECK_MAT_NEAR(vec_to_se3(as<Vector6d>(c["in"]["V"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("se3_to_vec"))
        CHECK_MAT_NEAR(se3_to_vec(as<Matrix4d>(c["in"]["se3mat"])), as_vec(c["out"]), TOL);
    for (const auto& c : of("adjoint")) CHECK_MAT_NEAR(adjoint(as<Matrix4d>(c["in"]["T"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("screw_to_axis"))
        CHECK_MAT_NEAR(screw_to_axis(as<Vector3d>(c["in"]["q"]), as<Vector3d>(c["in"]["s"]),
                                     c["in"]["h"].get<double>()),
                       as_vec(c["out"]), TOL);
}

TEST_CASE("axis_ang6, matrix_exp6, matrix_log6") {
    for (const auto& c : of("axis_ang6")) {
        const auto [S, th] = axis_ang6(as<Vector6d>(c["in"]["expc6"]));
        CHECK_MAT_NEAR(S, as_vec(c["out"]["S"]), TOL);
        CHECK_SCALAR_NEAR(th, c["out"]["theta"].get<double>(), TOL);
    }
    for (const auto& c : of("matrix_exp6"))
        CHECK_MAT_NEAR(matrix_exp6(as<Matrix4d>(c["in"]["se3mat"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("matrix_log6"))
        CHECK_MAT_NEAR(matrix_log6(as<Matrix4d>(c["in"]["T"])), as_mat(c["out"]), TOL);
}

TEST_CASE("project_to_so3/se3, distance_to_so3/se3, test_if_so3/se3") {
    for (const auto& c : of("project_to_so3"))
        CHECK_MAT_NEAR(project_to_so3(as<Matrix3d>(c["in"]["mat"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("distance_to_so3"))
        CHECK_SCALAR_NEAR(distance_to_so3(as<Matrix3d>(c["in"]["mat"])), c["out"].get<double>(), TOL);
    for (const auto& c : of("test_if_so3"))
        CHECK(test_if_so3(as<Matrix3d>(c["in"]["mat"])) == c["out"].get<bool>());
    for (const auto& c : of("project_to_se3"))
        CHECK_MAT_NEAR(project_to_se3(as<Matrix4d>(c["in"]["mat"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("distance_to_se3"))
        CHECK_SCALAR_NEAR(distance_to_se3(as<Matrix4d>(c["in"]["mat"])), c["out"].get<double>(), TOL);
    for (const auto& c : of("test_if_se3"))
        CHECK(test_if_se3(as<Matrix4d>(c["in"]["mat"])) == c["out"].get<bool>());
}

// ------------------------------------------------------------------ chapters 4-6
TEST_CASE("fkin_space, fkin_body, jacobian_space, jacobian_body") {
    for (const auto& c : of("fkin_space"))
        CHECK_MAT_NEAR(fkin_space(as<Matrix4d>(c["in"]["M"]), as_mat(c["in"]["Slist"]), as_vec(c["in"]["thetalist"])),
                       as_mat(c["out"]), TOL);
    for (const auto& c : of("fkin_body"))
        CHECK_MAT_NEAR(fkin_body(as<Matrix4d>(c["in"]["M"]), as_mat(c["in"]["Blist"]), as_vec(c["in"]["thetalist"])),
                       as_mat(c["out"]), TOL);
    for (const auto& c : of("jacobian_space"))
        CHECK_MAT_NEAR(jacobian_space(as_mat(c["in"]["Slist"]), as_vec(c["in"]["thetalist"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("jacobian_body"))
        CHECK_MAT_NEAR(jacobian_body(as_mat(c["in"]["Blist"]), as_vec(c["in"]["thetalist"])), as_mat(c["out"]), TOL);
}

TEST_CASE("ikin_body, ikin_space") {
    for (const char* fn : {"ikin_body", "ikin_space"}) {
        for (const auto& c : of(fn)) {
            const json& in = c["in"];
            const bool body = std::string(fn) == "ikin_body";
            const MatrixXd list = as_mat(in[body ? "Blist" : "Slist"]);
            const IKResult r = (body ? ikin_body : ikin_space)(list, as<Matrix4d>(in["M"]), as<Matrix4d>(in["T"]),
                                                             as_vec(in["thetalist0"]), in["eomg"].get<double>(),
                                                             in["ev"].get<double>());
            const bool ok = c["out"]["success"].get<bool>();
            INFO(fn);
            CHECK(r.success == ok);
            if (ok) CHECK_MAT_NEAR(r.thetalist, as_vec(c["out"]["thetalist"]), TOL);
        }
    }
}

// ------------------------------------------------------------------ chapter 8
TEST_CASE("ad, inverse_dynamics, mass_matrix, vel_quadratic_forces, gravity_forces, end_effector_forces") {
    for (const auto& c : of("ad")) CHECK_MAT_NEAR(ad(as<Vector6d>(c["in"]["V"])), as_mat(c["out"]), TOL);
    for (const auto& c : of("inverse_dynamics")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        CHECK_MAT_NEAR(inverse_dynamics(as_vec(in["thetalist"]), as_vec(in["dthetalist"]), as_vec(in["ddthetalist"]),
                                        as<Vector3d>(in["g"]), as<Vector6d>(in["Ftip"]), k.Mlist, k.Glist, k.Slist),
                       as_vec(c["out"]), TOL);
    }
    for (const auto& c : of("mass_matrix")) {
        const Chain k = chain(c["in"]);
        CHECK_MAT_NEAR(mass_matrix(as_vec(c["in"]["thetalist"]), k.Mlist, k.Glist, k.Slist), as_mat(c["out"]), TOL);
    }
    for (const auto& c : of("vel_quadratic_forces")) {
        const Chain k = chain(c["in"]);
        CHECK_MAT_NEAR(vel_quadratic_forces(as_vec(c["in"]["thetalist"]), as_vec(c["in"]["dthetalist"]), k.Mlist,
                                            k.Glist, k.Slist),
                       as_vec(c["out"]), TOL);
    }
    for (const auto& c : of("gravity_forces")) {
        const Chain k = chain(c["in"]);
        CHECK_MAT_NEAR(gravity_forces(as_vec(c["in"]["thetalist"]), as<Vector3d>(c["in"]["g"]), k.Mlist, k.Glist,
                                      k.Slist),
                       as_vec(c["out"]), TOL);
    }
    for (const auto& c : of("end_effector_forces")) {
        const Chain k = chain(c["in"]);
        CHECK_MAT_NEAR(end_effector_forces(as_vec(c["in"]["thetalist"]), as<Vector6d>(c["in"]["Ftip"]), k.Mlist,
                                           k.Glist, k.Slist),
                       as_vec(c["out"]), TOL);
    }
}

TEST_CASE("forward_dynamics, euler_step, inverse_dynamics_trajectory, forward_dynamics_trajectory") {
    for (const auto& c : of("forward_dynamics")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        CHECK_MAT_NEAR(forward_dynamics(as_vec(in["thetalist"]), as_vec(in["dthetalist"]), as_vec(in["taulist"]),
                                        as<Vector3d>(in["g"]), as<Vector6d>(in["Ftip"]), k.Mlist, k.Glist, k.Slist),
                       as_vec(c["out"]), TOL);
    }
    for (const auto& c : of("euler_step")) {
        const json& in = c["in"];
        const auto [t1, d1] = euler_step(as_vec(in["thetalist"]), as_vec(in["dthetalist"]),
                                         as_vec(in["ddthetalist"]), in["dt"].get<double>());
        CHECK_MAT_NEAR(t1, as_vec(c["out"]["thetalist"]), TOL);
        CHECK_MAT_NEAR(d1, as_vec(c["out"]["dthetalist"]), TOL);
    }
    for (const auto& c : of("inverse_dynamics_trajectory")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        CHECK_MAT_NEAR(inverse_dynamics_trajectory(as_mat(in["thetamat"]), as_mat(in["dthetamat"]),
                                                   as_mat(in["ddthetamat"]), as<Vector3d>(in["g"]),
                                                   as_mat(in["Ftipmat"]), k.Mlist, k.Glist, k.Slist),
                       as_mat(c["out"]), TOL);
    }
    for (const auto& c : of("forward_dynamics_trajectory")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        const auto [thm, dthm] = forward_dynamics_trajectory(
            as_vec(in["thetalist"]), as_vec(in["dthetalist"]), as_mat(in["taumat"]), as<Vector3d>(in["g"]),
            as_mat(in["Ftipmat"]), k.Mlist, k.Glist, k.Slist, in["dt"].get<double>(), in["intRes"].get<int>());
        CHECK_MAT_NEAR(thm, as_mat(c["out"]["thetamat"]), 1e-8);
        CHECK_MAT_NEAR(dthm, as_mat(c["out"]["dthetamat"]), 1e-8);
    }
}

// ------------------------------------------------------------------ chapter 9
TEST_CASE("time scalings and trajectories") {
    for (const auto& c : of("cubic_time_scaling"))
        CHECK_SCALAR_NEAR(cubic_time_scaling(c["in"]["Tf"].get<double>(), c["in"]["t"].get<double>()),
                          c["out"].get<double>(), TOL);
    for (const auto& c : of("quintic_time_scaling"))
        CHECK_SCALAR_NEAR(quintic_time_scaling(c["in"]["Tf"].get<double>(), c["in"]["t"].get<double>()),
                          c["out"].get<double>(), TOL);
    for (const auto& c : of("joint_trajectory")) {
        const json& in = c["in"];
        CHECK_MAT_NEAR(joint_trajectory(as_vec(in["thetastart"]), as_vec(in["thetaend"]), in["Tf"].get<double>(),
                                        in["N"].get<int>(), in["method"].get<int>()),
                       as_mat(c["out"]), TOL);
    }
    for (const char* fn : {"screw_trajectory", "cartesian_trajectory"}) {
        for (const auto& c : of(fn)) {
            const json& in = c["in"];
            const bool screw = std::string(fn) == "screw_trajectory";
            const TransformList traj =
                (screw ? screw_trajectory : cartesian_trajectory)(as<Matrix4d>(in["Xstart"]), as<Matrix4d>(in["Xend"]),
                                                                  in["Tf"].get<double>(), in["N"].get<int>(),
                                                                  in["method"].get<int>());
            const auto expected = as_list<Matrix4d>(c["out"]);
            INFO(fn);
            REQUIRE(traj.size() == expected.size());
            for (std::size_t i = 0; i < traj.size(); ++i) CHECK_MAT_NEAR(traj[i], expected[i], TOL);
        }
    }
}

// ------------------------------------------------------------------ chapter 11
TEST_CASE("computed_torque, simulate_control") {
    for (const auto& c : of("computed_torque")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        CHECK_MAT_NEAR(computed_torque(as_vec(in["thetalist"]), as_vec(in["dthetalist"]), as_vec(in["eint"]),
                                       as<Vector3d>(in["g"]), k.Mlist, k.Glist, k.Slist, as_vec(in["thetalistd"]),
                                       as_vec(in["dthetalistd"]), as_vec(in["ddthetalistd"]), in["Kp"].get<double>(),
                                       in["Ki"].get<double>(), in["Kd"].get<double>()),
                       as_vec(c["out"]), TOL);
    }
    for (const auto& c : of("simulate_control")) {
        const json& in = c["in"];
        const Chain k = chain(in);
        const auto [tau, th] = simulate_control(
            as_vec(in["thetalist"]), as_vec(in["dthetalist"]), as<Vector3d>(in["g"]), as_mat(in["Ftipmat"]), k.Mlist,
            k.Glist, k.Slist, as_mat(in["thetamatd"]), as_mat(in["dthetamatd"]), as_mat(in["ddthetamatd"]),
            as<Vector3d>(in["gtilde"]), as_list<Matrix4d>(in["Mtildelist"]), as_list<Matrix6d>(in["Gtildelist"]),
            in["Kp"].get<double>(), in["Ki"].get<double>(), in["Kd"].get<double>(), in["dt"].get<double>(),
            in["intRes"].get<int>());
        CHECK_MAT_NEAR(tau, as_mat(c["out"]["taumat"]), 1e-8);
        CHECK_MAT_NEAR(th, as_mat(c["out"]["thetamat"]), 1e-8);
    }
}

// ------------------------------------------------------------------ support library behaviour
TEST_CASE("support: wrong sizes throw instead of aborting") {
    const MatrixXd A = MatrixXd::Ones(3, 3);
    const VectorXd v = VectorXd::Ones(6);
    CHECK_THROWS_AS((void)(A * v).eval(), sbeg::EigenError);
    CHECK_THROWS_AS((void)A(5, 0), sbeg::EigenError);
    CHECK_THROWS_AS((void)fkin_space(Matrix4d::Identity(), MatrixXd::Zero(6, 1), VectorXd::Zero(2)),
                    std::invalid_argument);
    CHECK_THROWS_AS((void)as<Matrix3d>(json::parse("[[1,2],[3,4]]")), std::runtime_error);
    CHECK_THROWS_AS((void)sbeg::load_json(sbeg::lab_path("no_such_file.json")), std::runtime_error);
    // compare_near reports shape mismatches and NaN instead of crashing
    CHECK_FALSE(sbeg::compare_near(Vector3d::Zero(), Vector6d::Zero(), 1.0).ok);
    CHECK_FALSE(sbeg::compare_near(Vector3d(std::nan(""), 0, 0), Vector3d::Zero(), 1.0).ok);
    CHECK(sbeg::compare_near(Eigen::RowVector3d(1, 2, 3), Vector3d(1, 2, 3), 0.0).ok);
    CHECK(sbeg::approx_tol(0.0) == 1e-12);
    CHECK(sbeg::approx_tol(-8.0) == 8e-6);
}
