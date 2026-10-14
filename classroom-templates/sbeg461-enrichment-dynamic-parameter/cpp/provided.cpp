// SBEG 461 enrichment lab "parameter identification" -- the provided functions of starter.py (C++ track).
// GIVEN code: do not edit.  load_meta, load_robot, n_params, param_names, inertia_matrix, spatial_inertia,
// nominal_parameters, link_frames_motion, stack_regressor, joint_index, stack_torque, build_system,
// pi_to_log_cholesky, random_states (declared and documented in lab.hpp).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

sbeg::json load_meta() { return sbeg::load_json(data_path("meta.json")); }

std::string data_path(const std::string& file) { return sbeg::lab_path("../data/" + file); }

Robot load_robot(const std::string& key) { return get_robot(key); }

int n_params(int n) { return 13 * n; }

std::vector<std::string> param_names(int n) {
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i)
        for (const auto& s : LINK_NAMES) out.push_back(s + std::to_string(i + 1));
    for (const char* s : {"Ia", "bv", "bc"})
        for (int j = 0; j < n; ++j) out.push_back(std::string(s) + std::to_string(j + 1));
    return out;
}

Eigen::Matrix3d inertia_matrix(const Eigen::VectorXd& I6) {
    if (I6.size() != 6) throw std::invalid_argument("inertia_matrix: need 6 entries");
    Eigen::Matrix3d I;
    I << I6(0), I6(1), I6(2),
         I6(1), I6(3), I6(4),
         I6(2), I6(4), I6(5);
    return I;
}

Matrix6d spatial_inertia(const Eigen::VectorXd& pi10) {
    if (pi10.size() != NI) throw std::invalid_argument("spatial_inertia: need 10 entries");
    const double m = pi10(0);
    const Eigen::Vector3d h = pi10.segment<3>(1);
    Matrix6d G = Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = inertia_matrix(pi10.tail(6));
    G.topRightCorner<3, 3>() = mr::vec_to_so3(h);
    G.bottomLeftCorner<3, 3>() = mr::vec_to_so3(h).transpose();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

Eigen::VectorXd nominal_parameters(const Robot& robot, const Eigen::VectorXd& armature,
                                   const Eigen::VectorXd& b_viscous, const Eigen::VectorXd& b_coulomb) {
    const int n = robot.n;
    Eigen::VectorXd pi = Eigen::VectorXd::Zero(13 * n);
    for (int i = 0; i < n; ++i)
        pi.segment(NI * i, NI) = inertial_parameters(robot.Glist[static_cast<std::size_t>(i)]);
    const Eigen::VectorXd* extra[3] = {&armature, &b_viscous, &b_coulomb};
    for (int k = 0; k < 3; ++k) {
        if (extra[k]->size() == 0) continue;
        if (extra[k]->size() != n) throw std::invalid_argument("nominal_parameters: need n entries");
        pi.segment(NI * n + k * n, n) = *extra[k];
    }
    return pi;
}

LinkFrames link_frames_motion(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                              const Eigen::VectorXd& ddtheta) {
    const int n = robot.n;
    if (theta.size() != n || dtheta.size() != n || ddtheta.size() != n)
        throw std::invalid_argument("link_frames_motion: theta, dtheta, ddtheta need n entries");
    LinkFrames F;
    F.A = Eigen::MatrixXd::Zero(6, n);
    Eigen::Matrix4d Mi = Eigen::Matrix4d::Identity();
    for (int i = 0; i < n; ++i) {
        Mi = Mi * robot.Mlist[static_cast<std::size_t>(i)];
        F.A.col(i) = mr::adjoint(mr::trans_inv(Mi)) * Vector6d(robot.Slist.col(i));
    }
    Vector6d Vp = Vector6d::Zero();
    Vector6d dVp;
    dVp << 0.0, 0.0, 0.0, -robot.g;
    for (int i = 0; i < n; ++i) {
        const Vector6d Ai = F.A.col(i);
        const Eigen::Matrix4d Ti =
            mr::matrix_exp6(mr::vec_to_se3(Ai * -theta(i))) * mr::trans_inv(robot.Mlist[static_cast<std::size_t>(i)]);
        const Matrix6d AdT = mr::adjoint(Ti);
        const Vector6d Vi = AdT * Vp + Ai * dtheta(i);
        const Vector6d dVi = AdT * dVp + mr::ad(Vi) * Ai * dtheta(i) + Ai * ddtheta(i);
        F.T.push_back(Ti);
        F.V.push_back(Vi);
        F.dV.push_back(dVi);
        Vp = Vi;
        dVp = dVi;
    }
    return F;
}

Eigen::MatrixXd stack_regressor(const Robot& robot, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& dQ,
                                const Eigen::MatrixXd& ddQ) {
    const int n = robot.n;
    const Eigen::Index N = Q.rows();
    if (Q.cols() != n || dQ.rows() != N || dQ.cols() != n || ddQ.rows() != N || ddQ.cols() != n)
        throw std::invalid_argument("stack_regressor: Q, dQ, ddQ must be N x n");
    Eigen::MatrixXd W(N * n, 13 * n);
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::MatrixXd Yk =
            regressor(robot, Q.row(k).transpose(), dQ.row(k).transpose(), ddQ.row(k).transpose());
        if (Yk.rows() != n || Yk.cols() != 13 * n) throw std::runtime_error("stack_regressor: regressor must be n x 13n");
        W.middleRows(k * n, n) = Yk;
    }
    return W;
}

Eigen::VectorXi joint_index(int n_samples, int n) {
    Eigen::VectorXi j(static_cast<Eigen::Index>(n_samples) * n);
    for (Eigen::Index k = 0; k < j.size(); ++k) j(k) = static_cast<int>(k % n);
    return j;
}

Eigen::VectorXd stack_torque(const Eigen::MatrixXd& tau) {
    Eigen::VectorXd v(tau.size());
    for (Eigen::Index k = 0; k < tau.rows(); ++k)
        for (Eigen::Index j = 0; j < tau.cols(); ++j) v(k * tau.cols() + j) = tau(k, j);
    return v;
}

LinearSystem build_system(const Robot& robot, const ProcessedLog& data, const std::vector<int>& idx, double v_min) {
    const Eigen::MatrixXd Yall = stack_regressor(robot, data.q, data.dq, data.ddq);
    const Eigen::VectorXd tall = stack_torque(data.tau);
    const Eigen::VectorXi jall = joint_index(static_cast<int>(data.q.rows()), robot.n);
    const std::vector<bool> keep = moving_rows(data.dq, v_min);
    if (static_cast<Eigen::Index>(keep.size()) != Yall.rows())
        throw std::runtime_error("build_system: moving_rows must return one entry per row (N n)");
    std::vector<Eigen::Index> rows, cols;
    for (std::size_t r = 0; r < keep.size(); ++r)
        if (keep[r]) rows.push_back(static_cast<Eigen::Index>(r));
    if (idx.empty()) {
        for (Eigen::Index c = 0; c < Yall.cols(); ++c) cols.push_back(c);
    } else {
        for (int c : idx) {
            if (c < 0 || c >= Yall.cols()) throw std::invalid_argument("build_system: column index out of range");
            cols.push_back(c);
        }
    }
    LinearSystem S;
    S.Y.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(cols.size()));
    S.tau.resize(static_cast<Eigen::Index>(rows.size()));
    S.joint.resize(static_cast<Eigen::Index>(rows.size()));
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto rr = static_cast<Eigen::Index>(r);
        for (std::size_t c = 0; c < cols.size(); ++c) S.Y(rr, static_cast<Eigen::Index>(c)) = Yall(rows[r], cols[c]);
        S.tau(rr) = tall(rows[r]);
        S.joint(rr) = jall(rows[r]);
    }
    return S;
}

Eigen::VectorXd pi_to_log_cholesky(const Eigen::VectorXd& pi10) {
    const Eigen::Matrix4d J = pseudo_inertia(pi10);
    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();  // reversal permutation
    for (int i = 0; i < 4; ++i) P(i, 3 - i) = 1.0;
    const Eigen::LLT<Eigen::Matrix4d> llt(P * J * P);
    if (llt.info() != Eigen::Success) throw std::invalid_argument("pi_to_log_cholesky: pseudo-inertia not positive definite");
    const Eigen::Matrix4d L = llt.matrixL();
    Eigen::Matrix4d U = P * L * P;  // upper triangular, J = U U^T
    const double alpha = std::log(U(3, 3));
    U /= U(3, 3);
    Eigen::VectorXd out(10);
    out << alpha, std::log(U(0, 0)), std::log(U(1, 1)), std::log(U(2, 2)), U(0, 1), U(1, 2), U(0, 2), U(0, 3),
        U(1, 3), U(2, 3);
    return out;
}

Trajectory random_states(const Robot& robot, int n_samples, unsigned seed) {
    std::mt19937_64 gen(seed);  // fully specified by the C++ standard: the same numbers on every platform
    auto uniform = [&gen](double lo, double hi) {
        const double u = static_cast<double>(gen() >> 11) * (1.0 / 9007199254740992.0);  // [0, 1), 53 bits
        return lo + (hi - lo) * u;
    };
    const int n = robot.n;
    Trajectory s;
    s.q.resize(n_samples, n);
    s.dq.resize(n_samples, n);
    s.ddq.resize(n_samples, n);
    for (int k = 0; k < n_samples; ++k)
        for (int j = 0; j < n; ++j) s.q(k, j) = uniform(robot.limits(j, 0), robot.limits(j, 1));
    for (int k = 0; k < n_samples; ++k)
        for (int j = 0; j < n; ++j) s.dq(k, j) = uniform(-2.0, 2.0);
    for (int k = 0; k < n_samples; ++k)
        for (int j = 0; j < n; ++j) s.ddq(k, j) = uniform(-5.0, 5.0);
    return s;
}

}  // namespace lab
