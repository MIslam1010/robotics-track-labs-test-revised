// Auto-checks for SBEG 461 Lab 1 (Lagrangian dynamics vs modern_robotics) -- C++ track. Mirrors test_lab.py.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*time_derivative*,*euler_lagrange*"      (run only some tests, like pytest -k)
// Reference values: MR Eq. 8.6 (particle), MR Eq. 8.9-8.10 (2R point masses), MR Fig. 8.3 (M at two poses),
// MR Proposition 8.1 (Mdot - 2C skew), mrcpp (the C++ port of modern_robotics: mass_matrix, vel_quadratic_forces,
// gravity_forces, inverse_dynamics) and cpp_cases.json, which holds the numbers of the Python (SymPy) reference,
// so both languages are checked against the same values.
// test_make_symbols has no C++ counterpart (there are no symbols in the C++ track).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as_mat;
using sbeg::as_mats;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
const std::vector<std::string> ROBOTS = {"mr_2r_point", "arm_exo_2r", "eod_3r"};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// One row of data/states.csv (theta3 / dtheta3 dropped for the 2R robots, as in test_lab.py).
struct State {
    std::string name;
    VectorXd th, dth;
};

const std::vector<State>& states() {
    static const std::vector<State> S = [] {
        std::ifstream in = sbeg::open_input(sbeg::lab_path("../data/states.csv"));
        std::vector<State> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string cell;
            std::vector<std::string> cells;
            while (std::getline(ss, cell, ',')) cells.push_back(cell);
            if (cells.size() != 7) throw std::runtime_error("states.csv: expected 7 columns: " + line);
            const Eigen::Index n = lab::load_params(cells[0]).L.size();
            State s{cells[0], VectorXd(n), VectorXd(n)};
            for (Eigen::Index i = 0; i < n; ++i) {
                s.th(i) = std::stod(cells[static_cast<std::size_t>(1 + i)]);
                s.dth(i) = std::stod(cells[static_cast<std::size_t>(4 + i)]);
            }
            rows.push_back(s);
        }
        return rows;
    }();
    return S;
}

/// numpy.allclose(got, expected, atol=atol) with numpy's default rtol = 1e-5:  |got - exp| <= atol + rtol |exp|.
sbeg::NearResult allclose(const MatrixXd& got, const MatrixXd& expected, double atol) {
    sbeg::NearResult r = sbeg::compare_near(got, expected, atol);  // shape check and message
    if (r.ok || got.rows() != expected.rows() || got.cols() != expected.cols() || !got.allFinite()) return r;
    r.ok = ((got - expected).cwiseAbs().array() <= atol + 1e-5 * expected.cwiseAbs().array()).all();
    r.msg += "\n  (numpy.allclose: tolerance atol + 1e-5 |expected|)";
    return r;
}

#define CHECK_ALLCLOSE(got, expected, atol)                                                          \
    do {                                                                                             \
        const ::sbeg::NearResult allclose_ = allclose((got), (expected), (atol));                    \
        CHECK_MESSAGE(allclose_.ok, "CHECK_ALLCLOSE(" #got ", " #expected ")\n  " << allclose_.msg); \
    } while (false)

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

lab::StateFn kinetic(const lab::ChainParams& p) {
    return [p](const VectorXd& q, const VectorXd& dq) { return lab::planar_chain_energy(p, q, dq).K; };
}

lab::ConfigFn potential(const lab::ChainParams& p) {
    return [p](const VectorXd& q) { return lab::planar_chain_energy(p, q, VectorXd::Zero(q.size())).P; };
}

lab::MatrixFn mass_fn(const lab::ChainParams& p) {
    const lab::StateFn K = kinetic(p);
    return [K](const VectorXd& q) { return lab::mass_matrix(K, q); };
}

lab::NumericModel model(const std::string& name) { return lab::numeric_model(lab::load_params(name)); }

struct ClosedForm {
    Matrix2d M;
    Vector2d c, g, tau;
};

/// MR Eq. 8.9 generalised to links with CoM at r_i and inertia I_i (reduces to MR for r = L, I = 0).
ClosedForm closed_form_2r(const lab::ChainParams& p, const VectorXd& th, const VectorXd& dth, const VectorXd& ddth) {
    const double L1 = p.L(0), m1 = p.m(0), m2 = p.m(1), r1 = p.r(0), r2 = p.r(1), I1 = p.I(0), I2 = p.I(1), g = p.g;
    const double c2 = std::cos(th(1)), s2 = std::sin(th(1)), h = m2 * L1 * r2;
    ClosedForm f;
    f.M << I1 + I2 + m1 * r1 * r1 + m2 * (L1 * L1 + r2 * r2 + 2 * L1 * r2 * c2), I2 + m2 * (r2 * r2 + L1 * r2 * c2),
        I2 + m2 * (r2 * r2 + L1 * r2 * c2), I2 + m2 * r2 * r2;
    f.c << -h * s2 * (2 * dth(0) * dth(1) + dth(1) * dth(1)), h * s2 * dth(0) * dth(0);
    f.g << (m1 * r1 + m2 * L1) * g * std::cos(th(0)) + m2 * r2 * g * std::cos(th(0) + th(1)),
        m2 * r2 * g * std::cos(th(0) + th(1));
    f.tau = f.M * ddth + f.c + f.g;
    return f;
}

/// Same as np.linspace(a, b, n).
VectorXd linspace(double a, double b, Eigen::Index n) { return VectorXd::LinSpaced(n, a, b); }
}  // namespace

// ---------------------------------------------------------------- checkpoint 1: Euler-Lagrange machinery
TEST_CASE("test_time_derivative_chain_rule") {
    const lab::StateFn f = [](const VectorXd& q, const VectorXd& dq) { return std::sin(q(0)) * dq(1) * dq(1); };
    for (const auto& c : cases()["time_derivative_chain_rule"]) {
        const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), ddq = as_vec(c["ddq"]);
        const double want = std::cos(q(0)) * dq(0) * dq(1) * dq(1) + 2 * std::sin(q(0)) * dq(1) * ddq(1);
        const double got = lab::time_derivative(f, q, dq, ddq);
        CHECK_SCALAR_NEAR(got, want, 1e-9);
        CHECK_SCALAR_NEAR(got, c["value"].get<double>(), 1e-9);
    }
}

TEST_CASE("test_euler_lagrange_particle_mr_eq_8_6") {
    const json& C = cases()["euler_lagrange_particle_mr_eq_8_6"];
    const double m = C["m"].get<double>(), g = C["g"].get<double>();
    const lab::StateFn Lag = [m, g](const VectorXd& x, const VectorXd& dx) { return 0.5 * m * dx(0) * dx(0) - m * g * x(0); };
    for (const auto& c : C["cases"]) {
        const double ddx = c["ddx"].get<double>();
        const VectorXd f = lab::euler_lagrange(Lag, vec({c["x"].get<double>()}), vec({c["dx"].get<double>()}), vec({ddx}));
        // Python (exact SymPy): == 0. Differences of differences of L (|L| up to ~60 here): round-off ~2e-8, so 1e-6.
        CHECK_MAT_NEAR(f, vec({m * ddx + m * g}), 1e-6);
        CHECK_MAT_NEAR(f, vec({c["f"].get<double>()}), 1e-6);
    }
}

TEST_CASE("test_euler_lagrange_pendulum") {
    const json& C = cases()["euler_lagrange_pendulum"];
    const double m = C["m"].get<double>(), l = C["l"].get<double>(), g = C["g"].get<double>();
    const lab::StateFn Lag = [m, l, g](const VectorXd& th, const VectorXd& dth) {
        return 0.5 * m * l * l * dth(0) * dth(0) - m * g * l * std::sin(th(0));
    };
    for (const auto& c : C["cases"]) {
        const double th = c["th"].get<double>(), ddth = c["ddth"].get<double>();
        const VectorXd f = lab::euler_lagrange(Lag, vec({th}), vec({c["dth"].get<double>()}), vec({ddth}));
        CHECK_MAT_NEAR(f, vec({m * l * l * ddth + m * g * l * std::cos(th)}), 1e-6);  // as above
        CHECK_MAT_NEAR(f, vec({c["f"].get<double>()}), 1e-6);
    }
}

// ---------------------------------------------------------------- checkpoint 2: energies, M, g, Christoffel, C
TEST_CASE("test_energy_mr_2r_point") {
    const lab::ChainParams p = lab::load_params("mr_2r_point");
    const lab::Energy E = lab::planar_chain_energy(p, vec({0.4, -1.1}), vec({0.7, -0.3}));
    // MR p. 272: K1, K2, P1, P2 with L = m = 1
    const double c2 = std::cos(-1.1);
    const double K_mr = 0.5 * 0.7 * 0.7 + 0.5 * ((2 + 2 * c2) * 0.7 * 0.7 + 2 * (1 + c2) * 0.7 * -0.3 + 0.3 * 0.3);
    const double P_mr = 9.81 * std::sin(0.4) + 9.81 * (std::sin(0.4) + std::sin(0.4 - 1.1));
    CHECK_SCALAR_NEAR(E.K, K_mr, 1e-12);
    CHECK_SCALAR_NEAR(E.P, P_mr, 1e-12);
    const json& c = cases()["energy_mr_2r_point"];
    CHECK_SCALAR_NEAR(E.K, c["K"].get<double>(), 1e-12);
    CHECK_SCALAR_NEAR(E.P, c["P"].get<double>(), 1e-12);
}

TEST_CASE("test_mass_matrix_symbolic_mr_eq_8_10") {
    const lab::StateFn K = kinetic(lab::load_params("mr_2r_point"));
    for (const auto& c : cases()["mass_matrix_symbolic_mr_eq_8_10"]) {
        const VectorXd th = as_vec(c["theta"]);
        INFO("theta = " << th.transpose());
        const MatrixXd M = lab::mass_matrix(K, th);
        Matrix2d want;
        want << 3 + 2 * std::cos(th(1)), 1 + std::cos(th(1)), 1 + std::cos(th(1)), 1;
        CHECK_MAT_NEAR(M, want, 1e-12);
        CHECK_MAT_NEAR(M, as_mat(c["M"]), 1e-12);
    }
}

TEST_CASE("test_gravity_symbolic") {
    const lab::ConfigFn P = potential(lab::load_params("mr_2r_point"));
    const VectorXd th = vec({0.37, 1.9});
    const VectorXd gv = lab::gravity_vector(P, th);
    const Vector2d want(2 * 9.81 * std::cos(th(0)) + 9.81 * std::cos(th(0) + th(1)), 9.81 * std::cos(th(0) + th(1)));
    CHECK_MAT_NEAR(gv, want, 1e-9);  // Python (exact SymPy): 1e-12; finite differences: 1e-9
    CHECK_MAT_NEAR(gv, as_vec(cases()["gravity_symbolic"]["g"]), 1e-9);
}

TEST_CASE("test_christoffel_2r") {
    const lab::ChainParams p = lab::load_params("arm_exo_2r");
    const double t2 = 0.8, h = p.m(1) * p.L(0) * p.r(1), s = std::sin(t2);
    const std::vector<MatrixXd> G = lab::christoffel(mass_fn(p), vec({0.3, t2}));
    REQUIRE(G.size() == 2);
    MatrixXd want0(2, 2), want1(2, 2);  // want0(j, k) = Gamma_1jk, want1(j, k) = Gamma_2jk
    want0 << 0, -h * s, -h * s, -h * s;
    want1 << h * s, 0, 0, 0;
    CHECK_MAT_NEAR(G[0], want0, 1e-9);  // Python (exact SymPy): 1e-12; finite differences: 1e-9
    CHECK_MAT_NEAR(G[1], want1, 1e-9);
    const std::vector<MatrixXd> G_ref = as_mats(cases()["christoffel_2r"]["G"]);
    for (std::size_t i = 0; i < 2; ++i) CHECK_MAT_NEAR(G[i], G_ref[i], 1e-9);
}

TEST_CASE("test_christoffel_symmetric_in_jk_3r") {
    const lab::ChainParams p = lab::load_params("eod_3r");
    const json& c = cases()["christoffel_symmetric_in_jk_3r"];
    const std::vector<MatrixXd> G = lab::christoffel(mass_fn(p), as_vec(c["theta"]));
    REQUIRE(G.size() == 3);
    const std::vector<MatrixXd> dM = as_mats(c["dM"]), G_ref = as_mats(c["G"]);  // dM[k] = dM/dtheta_k (SymPy)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                INFO("i, j, k = " << i << ", " << j << ", " << k);
                const auto I = static_cast<std::size_t>(i), J = static_cast<std::size_t>(j);
                CHECK_SCALAR_NEAR(G[I](j, k) - G[I](k, j), 0.0, 1e-12);  // symmetric in j, k
                CHECK_SCALAR_NEAR(G[I](j, k) + G[J](i, k), dM[static_cast<std::size_t>(k)](i, j), 1e-9);
            }
    for (std::size_t i = 0; i < 3; ++i) CHECK_MAT_NEAR(G[i], G_ref[i], 1e-9);
}

TEST_CASE("test_coriolis_matrix_times_dtheta_mr_eq_8_10") {
    const lab::MatrixFn M = mass_fn(lab::load_params("mr_2r_point"));
    for (const auto& c : cases()["coriolis_matrix_times_dtheta_mr_eq_8_10"]) {
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]);
        INFO("theta = " << th.transpose() << ", dtheta = " << dth.transpose());
        const MatrixXd C = lab::coriolis_matrix(M, th, dth);
        const Vector2d want(-std::sin(th(1)) * (2 * dth(0) * dth(1) + dth(1) * dth(1)),
                            std::sin(th(1)) * dth(0) * dth(0));
        CHECK_MAT_NEAR(C * dth, want, 1e-9);  // Python (exact SymPy): == 0; finite differences: 1e-9
        CHECK_MAT_NEAR(C, as_mat(c["C"]), 1e-9);
    }
}

// ---------------------------------------------------------------- checkpoint 3: numeric model
TEST_CASE("test_mass_matrix_mr_fig_8_3") {
    const lab::NumericModel mdl = model("mr_2r_point");
    for (const auto& c : cases()["mass_matrix_mr_fig_8_3"]) {
        const double th2 = c["th2"].get<double>();
        INFO("th2 = " << th2);
        const MatrixXd M = mdl.M(vec({0.0, th2}));
        CHECK_ALLCLOSE(M, as_mat(c["M_fig"]), 0.006);  // MR Fig. 8.3 prints 2 decimals
        CHECK_MAT_NEAR(M, as_mat(c["M"]), 1e-12);
    }
}

TEST_CASE("test_numeric_model_vs_closed_form") {
    const json& C = cases()["numeric_model_vs_closed_form"];
    const VectorXd ddth = vec({0.4, -1.3});
    for (const std::string name : {"mr_2r_point", "arm_exo_2r"}) {
        const lab::ChainParams p = lab::load_params(name);
        const lab::NumericModel mdl = lab::numeric_model(p);
        for (const State& s : states()) {
            if (s.name != name) continue;
            INFO(name << ", theta = " << s.th.transpose() << ", dtheta = " << s.dth.transpose());
            const ClosedForm f = closed_form_2r(p, s.th, s.dth, ddth);
            CHECK_ALLCLOSE(mdl.M(s.th), f.M, 1e-10);
            CHECK_ALLCLOSE(mdl.c(s.th, s.dth), f.c, 1e-10);
            CHECK_ALLCLOSE(mdl.g(s.th), f.g, 1e-10);
            CHECK_ALLCLOSE(mdl.tau(s.th, s.dth, ddth), f.tau, 1e-10);
        }
    }
    for (const auto& c : C["cases"]) {  // the same states, against the SymPy model of the Python reference
        const lab::NumericModel mdl = model(c["name"].get<std::string>());
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]);
        CHECK_ALLCLOSE(mdl.c(th, dth), as_vec(c["c"]), 1e-10);
        CHECK_ALLCLOSE(mdl.tau(th, dth, ddth), as_vec(c["tau"]), 1e-10);
    }
}

TEST_CASE("test_properties_M_pd_and_skew") {
    const json& C = cases()["properties_M_pd_and_skew"];
    REQUIRE(C.size() == states().size());
    for (std::size_t k = 0; k < states().size(); ++k) {
        const State& s = states()[k];
        INFO(s.name << ", theta = " << s.th.transpose() << ", dtheta = " << s.dth.transpose());
        const lab::NumericModel mdl = model(s.name);
        const MatrixXd M = mdl.M(s.th);
        CHECK_ALLCLOSE(M, M.transpose(), 1e-12);
        const Eigen::SelfAdjointEigenSolver<MatrixXd> eig(0.5 * (M + M.transpose()));
        CHECK(eig.eigenvalues().minCoeff() > 0);
        CHECK(lab::skew_residual(mdl, s.th, s.dth) < 1e-10);
        CHECK_ALLCLOSE(mdl.C(s.th, s.dth) * s.dth, mdl.c(s.th, s.dth), 1e-10);
        CHECK_ALLCLOSE(mdl.C(s.th, s.dth), as_mat(C[k]["C"]), 1e-10);
        CHECK_ALLCLOSE(mdl.Mdot(s.th, s.dth), as_mat(C[k]["Mdot"]), 1e-10);
    }
}

TEST_CASE("test_power_balance") {
    const json& C = cases()["power_balance"];
    REQUIRE(C.size() == states().size());
    for (std::size_t k = 0; k < states().size(); ++k) {
        const State& s = states()[k];
        INFO(s.name << ", theta = " << s.th.transpose() << ", dtheta = " << s.dth.transpose());
        const lab::NumericModel mdl = model(s.name);
        const VectorXd ddth = linspace(-1.0, 1.5, s.th.size());
        CHECK(lab::power_residual(mdl, s.th, s.dth, ddth) < 1e-6);  // Python (exact SymPy): 1e-9; nested differences
        CHECK_SCALAR_NEAR(mdl.K(s.th, s.dth), 0.5 * s.dth.dot(mdl.M(s.th) * s.dth), 1e-12);
        CHECK_SCALAR_NEAR(mdl.K(s.th, s.dth), C[k]["K"].get<double>(), 1e-12);
    }
}

// ---------------------------------------------------------------- checkpoint 4: modern_robotics model
TEST_CASE("test_mr_chain_structure") {
    for (const std::string& name : ROBOTS) {
        INFO("robot " << name);
        const lab::ChainParams p = lab::load_params(name);
        const int n = p.n();
        const lab::MRChain ch = lab::mr_chain(p);
        REQUIRE(ch.Mlist.size() == static_cast<std::size_t>(n + 1));
        REQUIRE(ch.Glist.size() == static_cast<std::size_t>(n));
        REQUIRE(ch.Slist.rows() == 6);
        REQUIRE(ch.Slist.cols() == n);
        CHECK_ALLCLOSE(ch.gvec, Vector3d(0, -p.g, 0), 1e-8);
        // home pose of the tip = product of the Mlist entries = Trans(sum L, 0, 0)
        Matrix4d T = Matrix4d::Identity();
        for (const Matrix4d& Mi : ch.Mlist) T = T * Mi;
        CHECK_ALLCLOSE((T.block<3, 1>(0, 3)), Vector3d(p.L.sum(), 0, 0), 1e-8);
        // FK at a test pose: tip x-y matches planar geometry (bridge to 361)
        const VectorXd th = vec({0.3, -0.6, 0.9}).head(n);
        const Matrix4d Tsb = mr::fkin_space(T, ch.Slist, th);
        double x = 0, y = 0, phi = 0;
        for (int i = 0; i < n; ++i) {
            phi += th(i);
            x += p.L(i) * std::cos(phi);
            y += p.L(i) * std::sin(phi);
        }
        CHECK_ALLCLOSE((Tsb.block<2, 1>(0, 3)), Vector2d(x, y), 1e-8);
        const json& c = cases()["mr_chain_structure"][name];
        CHECK_ALLCLOSE(ch.Slist, as_mat(c["Slist"]), 1e-12);
        const std::vector<MatrixXd> Mref = as_mats(c["Mlist"]), Gref = as_mats(c["Glist"]);
        for (std::size_t i = 0; i < ch.Mlist.size(); ++i) CHECK_ALLCLOSE(ch.Mlist[i], Mref[i], 1e-12);
        for (std::size_t i = 0; i < ch.Glist.size(); ++i) CHECK_ALLCLOSE(ch.Glist[i], Gref[i], 1e-12);
    }
}

TEST_CASE("test_mr_terms_match_library") {
    for (const State& s : states()) {
        INFO(s.name << ", theta = " << s.th.transpose() << ", dtheta = " << s.dth.transpose());
        const lab::ChainParams p = lab::load_params(s.name);
        const lab::MRChain ch = lab::mr_chain(p);
        const lab::MRTerms t = lab::mr_terms(p, s.th, s.dth);
        CHECK_ALLCLOSE(t.M, mr::mass_matrix(s.th, ch.Mlist, ch.Glist, ch.Slist), 1e-12);
        CHECK_ALLCLOSE(t.c, mr::vel_quadratic_forces(s.th, s.dth, ch.Mlist, ch.Glist, ch.Slist), 1e-12);
        CHECK_ALLCLOSE(t.g, mr::gravity_forces(s.th, ch.gvec, ch.Mlist, ch.Glist, ch.Slist), 1e-12);
    }
    const json& C = cases()["mr_terms_match_library"];  // the same states, modern_robotics (Python) values
    REQUIRE(C.size() == states().size());
    for (std::size_t k = 0; k < states().size(); ++k) {
        const State& s = states()[k];
        CHECK_ALLCLOSE(as_vec(C[k]["theta"]), s.th, 1e-15);  // states.csv was read correctly
        const lab::MRTerms t = lab::mr_terms(lab::load_params(s.name), s.th, s.dth);
        CHECK_ALLCLOSE(t.M, as_mat(C[k]["M"]), 1e-12);
        CHECK_ALLCLOSE(t.c, as_vec(C[k]["c"]), 1e-12);
        CHECK_ALLCLOSE(t.g, as_vec(C[k]["g"]), 1e-12);
    }
}

TEST_CASE("test_mass_matrix_from_jacobians") {
    const json& C = cases()["mass_matrix_from_jacobians"];
    REQUIRE(C.size() == states().size());
    for (std::size_t k = 0; k < states().size(); ++k) {
        const State& s = states()[k];
        INFO(s.name << ", theta = " << s.th.transpose());
        const MatrixXd Mj = lab::mass_matrix_from_jacobians(lab::load_params(s.name), s.th);
        CHECK_ALLCLOSE(Mj, model(s.name).M(s.th), 1e-10);
        CHECK_ALLCLOSE(Mj, as_mat(C[k]["M"]), 1e-10);
    }
}

TEST_CASE("test_361_library_matches_modern_robotics") {
    // The kinematics backend fk_space / jacobian_space (mrcpp in the given code, or your own SBEG 361 C++ functions)
    // against mr::fkin_space / mr::jacobian_space, with the screw axes of your mr_chain.
    for (const std::string& name : ROBOTS) {
        INFO("robot " << name);
        const lab::ChainParams p = lab::load_params(name);
        const int n = p.n();
        MatrixXd Slist = MatrixXd::Zero(6, n);
        double joint_x = 0.0;
        for (int i = 0; i < n; ++i) {
            Slist(2, i) = 1.0;
            Slist(4, i) = -joint_x;
            joint_x += p.L(i);
        }
        CHECK_ALLCLOSE(lab::mr_chain(p).Slist, Slist, 1e-12);
        Matrix4d Mtip = Matrix4d::Identity();
        Mtip(0, 3) = joint_x;
        for (const State& s : states()) {
            if (s.name != name) continue;
            CHECK_ALLCLOSE(lab::fk_space(Mtip, Slist, s.th), mr::fkin_space(Mtip, Slist, s.th), 1e-12);
            CHECK_ALLCLOSE(lab::jacobian_space(Slist, s.th), mr::jacobian_space(Slist, s.th), 1e-12);
        }
    }
}

// ---------------------------------------------------------------- checkpoint 5: numeric model vs modern_robotics
TEST_CASE("test_compare_sympy_vs_modern_robotics") {
    for (const std::string& name : ROBOTS) {
        INFO("robot " << name);
        lab::StateList st;
        for (const State& s : states())
            if (s.name == name) st.emplace_back(s.th, s.dth);
        const std::map<std::string, double> err = lab::compare(lab::load_params(name), st);
        double worst = 0.0;
        for (const char* key : {"M", "c", "g", "M_jac"}) {
            INFO("key " << key);
            REQUIRE(err.count(key) == 1);
        }
        for (const auto& [key, value] : err) worst = std::max(worst, value);
        CHECK(worst < 1e-9);
    }
}

TEST_CASE("test_inverse_dynamics_vs_modern_robotics") {
    for (const State& s : states()) {
        INFO(s.name << ", theta = " << s.th.transpose() << ", dtheta = " << s.dth.transpose());
        const lab::ChainParams p = lab::load_params(s.name);
        const lab::MRChain ch = lab::mr_chain(p);
        const VectorXd ddth = linspace(0.5, -0.8, s.th.size());
        const VectorXd tau_mr =
            mr::inverse_dynamics(s.th, s.dth, ddth, ch.gvec, mr::Vector6d::Zero(), ch.Mlist, ch.Glist, ch.Slist);
        CHECK_ALLCLOSE(model(s.name).tau(s.th, s.dth, ddth), tau_mr, 1e-9);
    }
    const json& C = cases()["inverse_dynamics_vs_modern_robotics"];  // modern_robotics (Python) values
    for (const auto& c : C) {
        const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]), ddth = as_vec(c["ddtheta"]);
        CHECK_ALLCLOSE(model(c["name"].get<std::string>()).tau(th, dth, ddth), as_vec(c["tau"]), 1e-9);
    }
}

// ---------------------------------------------------------------- independent reference values (Workshop Problem 4)
TEST_CASE("test_eod_3r_gravity_and_mass_matrix_workshop_p4") {
    // Hard-coded numbers from Workshop Problem 4b/4d (recomputed independently), so that a consistent but wrong pair
    // of models (numeric Lagrangian and Mlist/Glist with the same mistake) cannot pass.
    const lab::NumericModel mdl = model("eod_3r");
    const VectorXd th0 = VectorXd::Zero(3);
    const VectorXd g0 = vec({106.4385, 32.8635, 3.4335});
    CHECK_ALLCLOSE(mdl.g(th0), g0, 1e-4);
    MatrixXd M0(3, 3);
    M0 << 9.7205, 3.6905, 0.4280, 3.6905, 1.6805, 0.2180, 0.4280, 0.2180, 0.0430;
    CHECK_ALLCLOSE(mdl.M(th0), M0, 1e-4);
    const VectorXd th1 = vec({45 * PI / 180, -60 * PI / 180, -30 * PI / 180});
    const VectorXd g1 = vec({82.880, 30.855, 2.428});
    CHECK_ALLCLOSE(mdl.g(th1), g1, 1e-3);
}
