// Auto-checks for SBEG 461 Lab 11 (RRT for the course arm; path smoothing + time scaling) -- C++ track.
// Mirrors test_lab.py test by test (a parametrised Python test is ONE test case here that loops over the
// parameters).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*link_points*,*clearance*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: fkin_space for the skeleton points, inverse_dynamics),
// an independent loop implementation of the capsule clearance written in this file, the numbers of the Python
// reference in cpp_cases.json, and the defining properties of RRT / RRT* trees (MR Algorithm 10.3, p. 381;
// RRT* p. 385), shortcutting (MR Sec. 10.8) and trapezoidal time scaling (MR Sec. 9.2.2). Planning tests use
// fixed seeds of lab::Rng but only check properties, so any correct implementation passes whatever order it
// draws its random numbers in.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
const std::vector<std::string> SCENES = {"bench_or", "eod_rig"};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Arm& ARM() {
    static const lab::Arm A = lab::load_course_arm();
    return A;
}

lab::Scene scene(const std::string& name) { return lab::load_scene(name); }

VectorXd row(const MatrixXd& M, Eigen::Index k) { return M.row(k).transpose(); }

/// List of configurations in JSON -> N x 6 matrix (row = configuration).
MatrixXd qlist(const json& j) {
    MatrixXd Q(static_cast<Eigen::Index>(j.size()), 6);
    for (std::size_t k = 0; k < j.size(); ++k) Q.row(static_cast<Eigen::Index>(k)) = as_vec(j[k]).transpose();
    return Q;
}

bool same(const VectorXd& a, const VectorXd& b) { return a.size() == b.size() && (a.array() == b.array()).all(); }

// ------------------------------------------------------------------ independent model (as in test_lab.py)
/// Skeleton points by mrcpp: point k is at FKinSpace(I, S[:, :l], q[:l]) p_k, l = its link.
MatrixXd ref_points(const VectorXd& q) {
    const lab::Arm& A = ARM();
    MatrixXd P(7, 3);
    for (Eigen::Index k = 0; k < 7; ++k) {
        const int l = A.skeleton_link.at(static_cast<std::size_t>(k));
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        if (l > 0) T = mr::fkin_space(Eigen::Matrix4d::Identity(), A.Slist.leftCols(l), q.head(l));
        const Vector3d p = A.skeleton.row(k).transpose();
        P.row(k) = (T.topLeftCorner<3, 3>() * p + T.block<3, 1>(0, 3)).transpose();
    }
    return P;
}

double ref_clearance(const VectorXd& q, const lab::Scene& sc) {
    const MatrixXd P = ref_points(q);
    const lab::Arm& A = ARM();
    double best = std::numeric_limits<double>::infinity();
    for (Eigen::Index k = 1; k < 6; ++k) {
        const Vector3d a = P.row(k).transpose(), b = P.row(k + 1).transpose();
        const double r = A.link_radius(k - 1);
        for (Eigen::Index m = 0; m < sc.spheres.rows(); ++m) {
            const Vector3d c = sc.spheres.block<1, 3>(m, 0).transpose();
            const double t = std::clamp((c - a).dot(b - a) / (b - a).dot(b - a), 0.0, 1.0);
            best = std::min(best, (a + t * (b - a) - c).norm() - r - sc.spheres(m, 3));
        }
        for (Eigen::Index m = 0; m < sc.planes.rows(); ++m) {
            const Vector3d n = sc.planes.block<1, 3>(m, 0).transpose();
            best = std::min(best, std::min(n.dot(a), n.dot(b)) - sc.planes(m, 3) - r);
        }
    }
    return best;
}

/// Re-check a via-point path more finely than the planner (sampled checks may miss < tol).
bool path_is_free(const MatrixXd& path, const lab::Scene& sc, double step = 0.005, double tol = 0.003) {
    const lab::Arm& A = ARM();
    if (path.cols() != 6) return false;
    for (Eigen::Index s = 0; s + 1 < path.rows(); ++s) {
        const VectorXd qa = row(path, s), qb = row(path, s + 1);
        const int n = std::max(1, static_cast<int>(std::ceil((qb - qa).cwiseAbs().maxCoeff() / step)));
        for (int i = 0; i <= n; ++i) {
            const VectorXd q = qa + (static_cast<double>(i) / n) * (qb - qa);
            for (Eigen::Index j = 0; j < 6; ++j)
                if (q(j) < A.limits(j, 0) - 1e-9 || q(j) > A.limits(j, 1) + 1e-9) return false;
            if (ref_clearance(q, sc) < -tol) return false;
        }
    }
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

// ------------------------------------------------------------------------------------------ Part A
TEST_CASE("test_link_points_tool_point_is_fk") {
    for (const auto& c : cases()["link_points_tool_point_is_fk"]) {
        const VectorXd q = as_vec(c["q"]);
        const MatrixXd P = lab::link_points(ARM(), q);
        REQUIRE(P.rows() == 7);
        REQUIRE(P.cols() == 3);
        const Eigen::Matrix4d T = mr::fkin_space(ARM().M, ARM().Slist, q);
        const Vector3d tool = T.block<3, 1>(0, 3);
        CHECK_MAT_NEAR(P.row(6), tool, 1e-12);
        CHECK_MAT_NEAR(P.row(6), as_vec(c["tool"]), 1e-12);
    }
}

TEST_CASE("test_link_points_match_truncated_chains_and_batch") {
    const json& c = cases()["link_points_match_truncated_chains_and_batch"];
    const MatrixXd Q = as_mat(c["Q"]);
    const std::vector<MatrixXd> PB = lab::link_points_batch(ARM(), Q);
    REQUIRE(PB.size() == 6);
    const std::vector<MatrixXd> P_ref = sbeg::as_mats(c["P"]);
    for (Eigen::Index k = 0; k < Q.rows(); ++k) {
        INFO("configuration " << k);
        const std::size_t u = static_cast<std::size_t>(k);
        CHECK_MAT_NEAR(PB[u], ref_points(row(Q, k)), 1e-12);
        CHECK_MAT_NEAR(PB[u], P_ref[u], 1e-12);
    }
}

TEST_CASE("test_link_points_rigid_segments") {
    const json& c = cases()["link_points_rigid_segments"];
    const VectorXd L0 = as_vec(c["L0"]);
    const MatrixXd qs = qlist(c["qs"]);
    for (Eigen::Index k = 0; k < qs.rows(); ++k) {
        const MatrixXd P = lab::link_points(ARM(), row(qs, k));
        REQUIRE(P.rows() == 7);
        VectorXd L(6);
        for (Eigen::Index i = 0; i < 6; ++i) L(i) = (P.row(i + 1) - P.row(i)).norm();
        CHECK_MAT_NEAR(L, L0, 1e-12);
    }
}

TEST_CASE("test_clearance_matches_independent_model") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        const json& rows = cases()["clearance_matches_independent_model"][name];
        const MatrixXd Q = [&] {
            MatrixXd M(static_cast<Eigen::Index>(rows.size()), 6);
            for (std::size_t k = 0; k < rows.size(); ++k)
                M.row(static_cast<Eigen::Index>(k)) = as_vec(rows[k]["q"]).transpose();
            return M;
        }();
        VectorXd ref(Q.rows());
        for (Eigen::Index k = 0; k < Q.rows(); ++k) {
            ref(k) = ref_clearance(row(Q, k), sc);
            CHECK_SCALAR_NEAR(lab::clearance(ARM(), sc, row(Q, k)), ref(k), 1e-10);
            CHECK_SCALAR_NEAR(lab::clearance(ARM(), sc, row(Q, k)),
                              rows[static_cast<std::size_t>(k)]["clearance"].get<double>(), 1e-10);
        }
        const VectorXd got = lab::clearance_batch(ARM(), sc, Q);
        REQUIRE(got.size() == Q.rows());
        CHECK_MAT_NEAR(got, ref, 1e-10);
    }
}

TEST_CASE("test_start_goal_free_and_straight_line_blocked") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        const VectorXd mid = 0.5 * (sc.q_start + sc.q_goal);
        CHECK(lab::config_free(ARM(), sc, sc.q_start));
        CHECK(lab::config_free(ARM(), sc, sc.q_goal));
        CHECK_FALSE(lab::config_free(ARM(), sc, mid));
        CHECK_FALSE(lab::edge_free(ARM(), sc, sc.q_start, sc.q_goal));
        const json& c = cases()["start_goal_free_and_straight_line_blocked"][name];
        CHECK_SCALAR_NEAR(lab::clearance(ARM(), sc, sc.q_start), c["clearance_start"].get<double>(), 1e-10);
        CHECK_SCALAR_NEAR(lab::clearance(ARM(), sc, sc.q_goal), c["clearance_goal"].get<double>(), 1e-10);
        CHECK_SCALAR_NEAR(lab::clearance(ARM(), sc, mid), c["clearance_mid"].get<double>(), 1e-10);
    }
}

TEST_CASE("test_config_free_respects_joint_limits") {
    const lab::Scene sc = scene("bench_or");
    VectorXd q = sc.q_start;
    q(0) = ARM().limits(0, 1) + 0.01;
    CHECK_FALSE(lab::config_free(ARM(), sc, q));
    CHECK(lab::config_free(ARM(), sc, sc.q_start, 0.0));
    CHECK_FALSE(lab::config_free(ARM(), sc, sc.q_start, 1.0));
}

TEST_CASE("test_edge_free_short_moves_and_certified") {
    const lab::Scene sc = scene("bench_or");
    const json& c = cases()["edge_free_short_moves_and_certified"];
    const VectorXd qa = sc.q_start;
    VectorXd dq(6);
    dq << 0.05, -0.05, 0.0, 0.0, 0.0, 0.05;
    const VectorXd qb = qa + dq;
    CHECK_MAT_NEAR(qb, as_vec(c["qb"]), 1e-15);
    CHECK(lab::edge_free(ARM(), sc, qa, qb));
    CHECK(lab::edge_free(ARM(), sc, qa, qb, 0.02, 0.0, true));
    // an edge that leaves the joint limits is not free
    VectorXd qc = qa;
    qc(4) = ARM().limits(4, 0) - 0.1;
    CHECK_FALSE(lab::edge_free(ARM(), sc, qa, qc));
    // certified implies sampled (the certified check is conservative)
    const MatrixXd qx = qlist(c["qx"]);
    for (Eigen::Index k = 0; k < qx.rows(); ++k) {
        INFO("random edge " << k);
        if (lab::edge_free(ARM(), sc, qa, row(qx, k), 0.02, 0.0, true)) CHECK(lab::edge_free(ARM(), sc, qa, row(qx, k)));
    }
}

TEST_CASE("test_edge_free_catches_a_thin_collision") {
    // A motion whose endpoints are free but whose middle collides must be rejected.
    const lab::Scene sc = scene("eod_rig");
    const VectorXd qs = sc.q_start, qg = sc.q_goal;
    const int S = 401;
    MatrixXd Q(S, 6);
    for (int i = 0; i < S; ++i) Q.row(i) = (qs + (static_cast<double>(i) / (S - 1)) * (qg - qs)).transpose();
    const VectorXd c = lab::clearance_batch(ARM(), sc, Q);
    REQUIRE(c.size() == S);
    Eigen::Index k = 0;
    c.minCoeff(&k);
    REQUIRE(c(k) < 0);
    // free points on either side of the collision
    Eigen::Index left = -1, right = -1;
    for (Eigen::Index i = 0; i < k; ++i)
        if (c(i) > 0) left = i;
    for (Eigen::Index i = S - 1; i >= k; --i)
        if (c(i) > 0) right = i;
    REQUIRE(left >= 0);
    REQUIRE(right >= 0);
    const json& ref = cases()["edge_free_catches_a_thin_collision"];
    CHECK(k == ref["k"].get<int>());
    CHECK_SCALAR_NEAR(c(k), ref["c_min"].get<double>(), 1e-10);
    CHECK_FALSE(lab::edge_free(ARM(), sc, row(Q, left), row(Q, right)));
}

// ------------------------------------------------------------------------------------------ Part B
TEST_CASE("test_sample_config_box_and_goal_bias") {
    lab::Rng rng(1);
    const MatrixXd box = scene("bench_or").sample_limits;
    const VectorXd qg = VectorXd::Constant(6, 9.0);
    const int n = 3000;
    int n_goal = 0;
    VectorXd sum = VectorXd::Zero(6);
    bool in_box = true;
    for (int k = 0; k < n; ++k) {
        const VectorXd x = lab::sample_config(rng, box, qg, 0.2);
        REQUIRE(x.size() == 6);
        if (same(x, qg)) {
            ++n_goal;
            continue;
        }
        sum += x;
        in_box = in_box && (x.array() >= box.col(0).array()).all() && (x.array() <= box.col(1).array()).all();
    }
    const double frac = static_cast<double>(n_goal) / n;
    INFO("goal fraction = " << frac);
    CHECK(frac > 0.17);
    CHECK(frac < 0.23);
    CHECK(in_box);
    REQUIRE(n_goal < n);
    CHECK_MAT_NEAR(sum / (n - n_goal), 0.5 * (box.col(0) + box.col(1)), 0.12);
    bool all_goal = true, any_goal = false;
    for (int k = 0; k < 10; ++k) all_goal = all_goal && same(lab::sample_config(rng, box, qg, 1.0), qg);
    for (int k = 0; k < 50; ++k) any_goal = any_goal || same(lab::sample_config(rng, box, qg, 0.0), qg);
    CHECK(all_goal);
    CHECK_FALSE(any_goal);
}

TEST_CASE("test_nearest_and_steer") {
    const json& c = cases()["nearest_and_steer"];
    const MatrixXd nodes = as_mat(c["nodes"]);
    const MatrixXd queries = qlist(c["queries"]);
    for (Eigen::Index k = 0; k < queries.rows(); ++k) {
        Eigen::Index best = 0;
        (nodes.rowwise() - queries.row(k)).rowwise().norm().minCoeff(&best);
        const int got = lab::nearest(nodes, row(queries, k));
        CHECK(got == static_cast<int>(best));
        CHECK(got == c["nearest"][static_cast<std::size_t>(k)].get<int>());
    }
    MatrixXd nodes2(3, 6);
    nodes2 << VectorXd::Zero(6).transpose(), VectorXd::Ones(6).transpose(), VectorXd::Zero(6).transpose();
    CHECK(lab::nearest(nodes2, 0.1 * VectorXd::Ones(6)) == 0);  // tie -> smallest index
    VectorXd a = VectorXd::Zero(6), b(6), e(6);
    b << 3.0, 4.0, 0, 0, 0, 0;
    e << 0.6, 0.8, 0, 0, 0, 0;
    CHECK_MAT_NEAR(lab::steer(a, b, 1.0), e, 1e-8);
    CHECK_MAT_NEAR(lab::steer(a, b, 6.0), b, 1e-8);
    CHECK_MAT_NEAR(lab::steer(a, b, 5.0), b, 1e-8);
}

TEST_CASE("test_tree_path_and_path_length") {
    MatrixXd nodes(4, 6);
    nodes << 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 0, 0,
             0, 0, 2, 0, 0, 0;
    const std::vector<int> parent = {-1, 0, 1, 0};
    const MatrixXd p = lab::tree_path(nodes, parent, 2);
    CHECK_MAT_NEAR(p, nodes.topRows(3), 1e-8);
    CHECK_MAT_NEAR(p, as_mat(cases()["tree_path_and_path_length"]["path_2"]), 1e-12);
    CHECK_APPROX(lab::path_length(p), 2.0);
    CHECK_APPROX(lab::path_length(lab::tree_path(nodes, parent, 3)), 2.0);
}

TEST_CASE("test_rrt_finds_a_valid_path") {
    for (const auto& name : SCENES) {
        for (unsigned seed : {0u, 1u, 2u}) {
            INFO("scene " << name << ", seed " << seed);
            const lab::Scene sc = scene(name);
            const double d = 0.3;
            lab::Rng rng(seed);
            const lab::RRTResult res =
                lab::rrt(ARM(), sc, sc.q_start, sc.q_goal, rng, d, 0.1, 3000, 0.02, sc.sample_limits);
            REQUIRE(res.success);
            const MatrixXd& path = res.path;
            const MatrixXd& nodes = res.nodes;
            const std::vector<int>& parent = res.parent;
            REQUIRE(path.rows() >= 2);
            REQUIRE(path.cols() == 6);
            REQUIRE(nodes.cols() == 6);
            REQUIRE(parent.size() == static_cast<std::size_t>(nodes.rows()));
            CHECK_MAT_NEAR(path.row(0), sc.q_start, 1e-8);
            CHECK_MAT_NEAR(path.row(path.rows() - 1), sc.q_goal, 1e-8);
            CHECK(parent[0] == -1);
            bool parents_ok = true, edges_ok = true;
            for (std::size_t i = 1; i < parent.size(); ++i) {
                const int p = parent[i];
                parents_ok = parents_ok && p >= 0 && p < static_cast<int>(i);
                if (p < 0 || p >= static_cast<int>(i)) continue;
                const double edge = (nodes.row(static_cast<Eigen::Index>(i)) - nodes.row(p)).norm();
                edges_ok = edges_ok && edge <= d + 1e-9 && edge > 0;
            }
            CHECK(parents_ok);
            CHECK(edges_ok);
            CHECK(path_is_free(path, sc));
        }
    }
}

TEST_CASE("test_rrt_reports_failure") {
    const lab::Scene sc = scene("bench_or");
    const VectorXd bad = 0.5 * (sc.q_start + sc.q_goal);  // in collision
    lab::Rng rng0(0);
    lab::RRTResult res = lab::rrt(ARM(), sc, sc.q_start, bad, rng0, 0.3, 0.1, 200, 0.02, sc.sample_limits);
    CHECK_FALSE(res.success);
    CHECK(res.path.rows() == 0);
    lab::Rng rng1(0);
    res = lab::rrt(ARM(), sc, sc.q_start, sc.q_goal, rng1, 0.01, 0.1, 30, 0.02, sc.sample_limits);
    CHECK_FALSE(res.success);
    CHECK(res.nodes.rows() <= 30);
}

TEST_CASE("test_rrt_star_tree_is_consistent") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        lab::Rng rng(3);
        const lab::RRTStarResult res =
            lab::rrt_star(ARM(), sc, sc.q_start, sc.q_goal, rng, 0.3, 0.1, 400, 1.0, 0.02, sc.sample_limits);
        REQUIRE(res.success);
        const MatrixXd& nodes = res.nodes;
        const std::vector<int>& parent = res.parent;
        const VectorXd& cost = res.cost;
        CHECK(nodes.rows() == 400);  // RRT* keeps going after the first solution
        REQUIRE(parent.size() == static_cast<std::size_t>(nodes.rows()));
        REQUIRE(cost.size() == nodes.rows());
        CHECK(cost(0) == 0.0);
        CHECK(parent[0] == -1);
        int bad = 0;
        for (Eigen::Index i = 1; i < nodes.rows(); ++i) {
            const int p = parent[static_cast<std::size_t>(i)];
            if (p < 0 || p >= nodes.rows()) {
                ++bad;
                continue;
            }
            if (std::abs(cost(i) - (cost(p) + (nodes.row(i) - nodes.row(p)).norm())) > 1e-9) ++bad;
        }
        CHECK_MESSAGE(bad == 0, bad << " nodes with cost != cost(parent) + edge length");
        const MatrixXd& path = res.path;
        REQUIRE(path.rows() >= 2);
        CHECK_MAT_NEAR(path.row(path.rows() - 1), sc.q_goal, 1e-8);
        REQUIRE(res.goal_index >= 0);
        REQUIRE(res.goal_index < cost.size());
        CHECK_SCALAR_NEAR(lab::path_length(path), cost(res.goal_index), 1e-9);
        const auto& hist = res.best_cost_history;
        for (std::size_t k = 1; k < hist.size(); ++k) CHECK(hist[k].second <= hist[k - 1].second + 1e-12);
        CHECK(path_is_free(path, sc));
    }
}

TEST_CASE("test_rrt_star_beats_rrt_on_average") {
    const lab::Scene sc = scene("bench_or");
    std::vector<double> star, plain;
    for (unsigned seed = 0; seed < 4; ++seed) {
        INFO("seed " << seed);
        lab::Rng r1(seed), r2(seed);
        const lab::RRTStarResult rs =
            lab::rrt_star(ARM(), sc, sc.q_start, sc.q_goal, r1, 0.3, 0.1, 700, 1.0, 0.02, sc.sample_limits);
        const lab::RRTResult rp = lab::rrt(ARM(), sc, sc.q_start, sc.q_goal, r2, 0.3, 0.1, 3000, 0.02, sc.sample_limits);
        REQUIRE(rs.success);
        REQUIRE(rp.success);
        star.push_back(lab::path_length(rs.path));
        plain.push_back(lab::path_length(rp.path));
    }
    INFO("median RRT* = " << median(star) << ", median RRT = " << median(plain));
    CHECK(median(star) < median(plain));
}

// ------------------------------------------------------------------------------------------ Part C
TEST_CASE("test_shortcut_greedy_straight_when_free") {
    lab::Scene sc = scene("bench_or");
    sc.spheres = MatrixXd(0, 4);  // remove the clutter
    const VectorXd qs = sc.q_start, qg = sc.q_goal;
    VectorXd d1(6), d2(6);
    d1 << 0.3, -0.3, 0.2, 0, 0, 0;
    d2 << 0, -0.3, 0, 0.2, 0, 0;
    MatrixXd detour(4, 6);
    detour << qs.transpose(), (qs + d1).transpose(), (qg + d2).transpose(), qg.transpose();
    const MatrixXd out = lab::shortcut_greedy(ARM(), sc, detour);
    REQUIRE(out.rows() == 2);
    REQUIRE(out.cols() == 6);
    CHECK_MAT_NEAR(out.row(0), qs, 1e-8);
    CHECK_MAT_NEAR(out.row(1), qg, 1e-8);
}

TEST_CASE("test_shortcut_greedy_on_rrt_path") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        lab::Rng rng(5);
        const MatrixXd path =
            lab::rrt(ARM(), sc, sc.q_start, sc.q_goal, rng, 0.3, 0.1, 3000, 0.02, sc.sample_limits).path;
        REQUIRE(path.rows() >= 2);
        const MatrixXd out = lab::shortcut_greedy(ARM(), sc, path);
        REQUIRE(out.rows() >= 1);
        REQUIRE(out.cols() == 6);
        CHECK_MAT_NEAR(out.row(0), path.row(0), 1e-8);
        CHECK_MAT_NEAR(out.row(out.rows() - 1), path.row(path.rows() - 1), 1e-8);
        CHECK(lab::path_length(out) <= lab::path_length(path) + 1e-12);
        CHECK(out.rows() >= 3);  // straight line is blocked
        bool only_via = true;  // keeps original via points only
        for (Eigen::Index k = 0; k < out.rows(); ++k) {
            bool found = false;
            for (Eigen::Index i = 0; i < path.rows() && !found; ++i)
                found = sbeg::compare_near(out.row(k), path.row(i), 1e-8).ok;
            only_via = only_via && found;
        }
        CHECK(only_via);
        // greedy property: from each kept point the next-but-one kept point is not reachable directly
        for (Eigen::Index k = 0; k + 2 < out.rows(); ++k) CHECK_FALSE(lab::edge_free(ARM(), sc, row(out, k), row(out, k + 2)));
        CHECK(path_is_free(out, sc));
    }
}

TEST_CASE("test_point_on_path") {
    MatrixXd path(3, 6);
    path << 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 2, 0, 0, 0, 0;
    const auto [k, p] = lab::point_on_path(path, 1.5);
    VectorXd e(6);
    e << 1, 0.5, 0, 0, 0, 0;
    CHECK(k == 1);
    CHECK_MAT_NEAR(p, e, 1e-8);
}

TEST_CASE("test_shortcut_random") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        lab::Rng rng_plan(7), rng_smooth(7);
        const MatrixXd path =
            lab::rrt(ARM(), sc, sc.q_start, sc.q_goal, rng_plan, 0.3, 0.1, 3000, 0.02, sc.sample_limits).path;
        REQUIRE(path.rows() >= 2);
        const MatrixXd out = lab::shortcut_random(ARM(), sc, path, rng_smooth, 60);
        REQUIRE(out.rows() >= 1);
        REQUIRE(out.cols() == 6);
        CHECK_MAT_NEAR(out.row(0), path.row(0), 1e-8);
        CHECK_MAT_NEAR(out.row(out.rows() - 1), path.row(path.rows() - 1), 1e-8);
        CHECK(lab::path_length(out) <= lab::path_length(path) + 1e-9);
        CHECK(path_is_free(out, sc));
    }
}

TEST_CASE("test_time_scale_path") {
    const lab::Scene sc = scene("bench_or");
    VectorXd d1(6);
    d1 << 0.4, -0.4, 0.3, 0, 0, 0;
    MatrixXd path(3, 6);
    path << sc.q_start.transpose(), (sc.q_start + d1).transpose(), sc.q_goal.transpose();
    const double dt = 0.01;
    const lab::ScaledPath r = lab::time_scale_path(path, ARM().qdot_max, ARM().qddot_max, dt);
    const VectorXd& t = r.t;
    const MatrixXd &Q = r.Q, &Qd = r.Qd, &Qdd = r.Qdd;
    const Eigen::Index K = t.size();
    REQUIRE(K >= 3);
    REQUIRE(Q.rows() == K);
    REQUIRE(Qd.rows() == K);
    REQUIRE(Qdd.rows() == K);
    REQUIRE(Q.cols() == 6);
    REQUIRE(Qd.cols() == 6);
    REQUIRE(Qdd.cols() == 6);
    VectorXd T_ref(2);
    for (Eigen::Index s = 0; s < 2; ++s)
        T_ref(s) = lab::trapezoid_from_limits(row(path, s), row(path, s + 1), ARM().qdot_max, ARM().qddot_max).T;
    const json& c = cases()["time_scale_path"];
    CHECK_MAT_NEAR(r.seg_T, T_ref, 1e-8);
    CHECK_MAT_NEAR(r.seg_T, as_vec(c["seg_T"]), 1e-12);
    CHECK_SCALAR_NEAR(t(K - 1), T_ref.sum(), 1e-8);
    CHECK_SCALAR_NEAR(t(K - 1), c["t_end"].get<double>(), 1e-12);
    const VectorXd dts = t.tail(K - 1) - t.head(K - 1);
    CHECK(dts.minCoeff() > 0);
    CHECK(dts.maxCoeff() <= dt + 1e-12);
    CHECK(t(0) == 0.0);
    CHECK_MAT_NEAR(Q.row(0), path.row(0), 1e-8);
    CHECK_MAT_NEAR(Q.row(K - 1), path.row(2), 1e-8);
    Eigen::Index k = 0;
    (t.array() - T_ref(0)).abs().minCoeff(&k);
    CHECK_SCALAR_NEAR(t(k), T_ref(0), 1e-8);
    CHECK_MAT_NEAR(Q.row(k), path.row(1), 1e-8);
    CHECK_MAT_NEAR(Qd.row(k), VectorXd::Zero(6), 1e-8);
    CHECK_MAT_NEAR(Qd.row(0), VectorXd::Zero(6), 1e-8);
    CHECK_MAT_NEAR(Qd.row(K - 1), VectorXd::Zero(6), 1e-8);
    bool vel_ok = true, acc_ok = true;
    for (Eigen::Index i = 0; i < K; ++i)
        for (Eigen::Index jj = 0; jj < 6; ++jj) {
            vel_ok = vel_ok && std::abs(Qd(i, jj)) <= ARM().qdot_max(jj) + 1e-9;
            acc_ok = acc_ok && std::abs(Qdd(i, jj)) <= ARM().qddot_max(jj) + 1e-9;
        }
    CHECK(vel_ok);
    CHECK(acc_ok);
    // velocity is the time derivative of position (central differences)
    double worst = 0.0;
    for (Eigen::Index i = 1; i + 1 < K; ++i) {
        const VectorXd v_fd = (row(Q, i + 1) - row(Q, i - 1)) / (t(i + 1) - t(i - 1));
        worst = std::max(worst, (v_fd - row(Qd, i)).cwiseAbs().maxCoeff());
    }
    INFO("max |finite-difference velocity - Qd| = " << worst);
    CHECK(worst < 0.05);
    // every sample lies on the straight segments
    for (Eigen::Index i = 0; i < K; i += 7) {
        const VectorXd q = row(Q, i);
        double best = std::numeric_limits<double>::infinity();
        for (Eigen::Index s = 0; s < 2; ++s) {
            const VectorXd a = row(path, s), b = row(path, s + 1);
            const double u = std::clamp((q - a).dot(b - a) / (b - a).dot(b - a), 0.0, 1.0);
            best = std::min(best, (a + u * (b - a) - q).norm());
        }
        INFO("sample " << i);
        CHECK(best < 1e-9);
    }
}

TEST_CASE("test_torque_profile_matches_mr") {
    const lab::Scene sc = scene("eod_rig");
    VectorXd d1(6);
    d1 << -0.4, -0.3, 0.2, 0.1, 0, 0;
    MatrixXd path(2, 6);
    path << sc.q_start.transpose(), (sc.q_start + d1).transpose();
    const lab::ScaledPath r = lab::time_scale_path(path, ARM().qdot_max, ARM().qddot_max, 0.02);
    REQUIRE(r.Q.rows() == r.t.size());
    REQUIRE(r.Qd.rows() == r.t.size());
    REQUIRE(r.Qdd.rows() == r.t.size());
    CHECK_MAT_NEAR(r.seg_T, as_vec(cases()["torque_profile_matches_mr"]["seg_T"]), 1e-12);
    const MatrixXd tau = lab::torque_profile(ARM(), r.Q, r.Qd, r.Qdd);
    REQUIRE(tau.rows() == r.Q.rows());
    REQUIRE(tau.cols() == r.Q.cols());
    for (Eigen::Index k = 0; k < r.t.size(); k += 5) {
        INFO("sample " << k);
        const VectorXd ref = mr::inverse_dynamics(row(r.Q, k), row(r.Qd, k), row(r.Qdd, k), ARM().g,
                                                  sbeg::Vector6d::Zero(), ARM().Mlist, ARM().Glist, ARM().Slist);
        CHECK_MAT_NEAR(tau.row(k), ref, 1e-10);
    }
}

TEST_CASE("test_pipeline_feasible_on_course_arm") {
    for (const auto& name : SCENES) {
        INFO("scene " << name);
        const lab::Scene sc = scene(name);
        const lab::PipelineResult out = lab::plan_and_scale(ARM(), sc, 0, "rrt", "greedy");
        REQUIRE(out.success);
        CHECK(out.limits.ok);
        REQUIRE(out.traj.t.size() > 0);
        CHECK(out.traj.t(out.traj.t.size() - 1) < 3.0);  // smoothed motion is short
    }
}
