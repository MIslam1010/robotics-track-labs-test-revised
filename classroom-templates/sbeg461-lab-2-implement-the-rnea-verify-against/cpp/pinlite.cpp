// SBEG 461 Lab 2 -- C++ track: GIVEN code (do not edit).  pinlite = Pinocchio's model + rnea conventions (see
// pinlite.hpp).  pin_rnea follows Pinocchio's own RNEA (Featherstone, Rigid Body Dynamics Algorithms, Table 5.1) in
// joint frames with LINEAR-FIRST spatial vectors; it is checked against the real Pinocchio through cpp_cases.json.
#include "pinlite.hpp"

#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <stdexcept>
#include <string>

namespace lab {

namespace {
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::Vector6d;

// Linear-first spatial algebra (Pinocchio's MotionTpl / ForceTpl / InertiaTpl operations).
Vector6d motion_act_inv(const Matrix4d& M, const Vector6d& m) {  // SE3::actInv(motion)
    const Matrix3d R = M.topLeftCorner<3, 3>();
    const Vector3d p = M.topRightCorner<3, 1>();
    const Vector3d v = m.head<3>(), w = m.tail<3>();
    Vector6d out;
    out.head<3>() = R.transpose() * (v - p.cross(w));
    out.tail<3>() = R.transpose() * w;
    return out;
}

Vector6d force_act(const Matrix4d& M, const Vector6d& f) {  // SE3::act(force)
    const Matrix3d R = M.topLeftCorner<3, 3>();
    const Vector3d p = M.topRightCorner<3, 1>();
    const Vector3d Rf = R * f.head<3>();
    Vector6d out;
    out.head<3>() = Rf;
    out.tail<3>() = R * f.tail<3>() + p.cross(Rf);
    return out;
}

Vector6d motion_cross_motion(const Vector6d& a, const Vector6d& b) {  // a ^ b (motion x motion)
    Vector6d out;
    out.head<3>() = a.tail<3>().cross(b.head<3>()) + a.head<3>().cross(b.tail<3>());
    out.tail<3>() = a.tail<3>().cross(b.tail<3>());
    return out;
}

Vector6d motion_cross_force(const Vector6d& a, const Vector6d& f) {  // a ^ f (motion x* force)
    Vector6d out;
    out.head<3>() = a.tail<3>().cross(f.head<3>());
    out.tail<3>() = a.tail<3>().cross(f.tail<3>()) + a.head<3>().cross(f.head<3>());
    return out;
}

Vector6d inertia_times(const PinJoint& J, const Vector6d& m) {  // Inertia * motion -> force
    const Vector3d v = m.head<3>(), w = m.tail<3>();
    const Vector3d f = J.mass * (v - J.lever.cross(w));
    Vector6d out;
    out.head<3>() = f;
    out.tail<3>() = J.inertia * w + J.lever.cross(f);
    return out;
}

Vector6d motion_subspace(const PinJoint& J) {  // S (6x1), linear first
    Vector6d S = Vector6d::Zero();
    if (J.revolute) {
        S.tail<3>() = J.axis;
    } else {
        S.head<3>() = J.axis;
    }
    return S;
}

Matrix4d joint_transform(const PinJoint& J, double q) {  // placement * M_J(q)
    Matrix4d MJ = Matrix4d::Identity();
    if (J.revolute) {
        MJ.topLeftCorner<3, 3>() = mr::matrix_exp3(mr::vec_to_so3(J.axis * q));
    } else {
        MJ.topRightCorner<3, 1>() = J.axis * q;
    }
    return J.placement * MJ;
}
}  // namespace

std::pair<PinModel, PinInfo> pinocchio_model(const Robot& robot) {
    PinModel model;
    model.gravity = robot.g;
    Matrix4d Mi = Matrix4d::Identity(), J_prev = Matrix4d::Identity();
    int parent = 0;
    for (int i = 0; i < robot.n; ++i) {
        Mi = (Mi * robot.Mlist[static_cast<std::size_t>(i)]).eval();
        const auto [R, p] = mr::trans_to_rp(Mi);
        const Vector3d w = robot.Slist.block<3, 1>(0, i), v = robot.Slist.block<3, 1>(3, i);
        PinJoint jt;
        Vector3d q;
        if (w.norm() > 1e-12) {
            q = w.cross(v);  // point on the axis closest to {0} (|w| = 1)
            jt.revolute = true;
            jt.axis = R.transpose() * w;
        } else {
            q = p;
            jt.revolute = false;
            jt.axis = R.transpose() * v;
        }
        const Matrix4d J = mr::rp_to_trans(R, q);
        jt.placement = mr::trans_inv(J_prev) * J;
        jt.parent = parent;
        const sbeg::Matrix6d& G = robot.Glist[static_cast<std::size_t>(i)];
        jt.mass = G(3, 3);
        jt.lever = R.transpose() * (p - q);
        jt.inertia = G.topLeftCorner<3, 3>();
        model.joints.push_back(jt);
        model.njoints += 1;
        J_prev = J;
        parent = model.njoints - 1;
    }
    const Matrix4d M_ee = Mi * robot.Mlist[static_cast<std::size_t>(robot.n)];
    PinInfo info;
    info.T_jn_ee = mr::trans_inv(J_prev) * M_ee;
    return {model, info};
}

Eigen::VectorXd pin_rnea(const PinModel& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                         const Eigen::VectorXd& a) {
    return pin_rnea(model, q, v, a, std::vector<Vector6d>(static_cast<std::size_t>(model.njoints), Vector6d::Zero()));
}

Eigen::VectorXd pin_rnea(const PinModel& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                         const Eigen::VectorXd& a, const std::vector<Vector6d>& fext) {
    const int n = model.njoints - 1;
    if (q.size() != n || v.size() != n || a.size() != n)
        throw std::invalid_argument("pin_rnea: q, v, a must have model.njoints - 1 = " + std::to_string(n) +
                                    " entries");
    if (fext.size() != static_cast<std::size_t>(model.njoints))
        throw std::invalid_argument("pin_rnea: fext must have model.njoints = " + std::to_string(model.njoints) +
                                    " entries (fext[0] = universe), got " + std::to_string(fext.size()));
    const std::size_t N = static_cast<std::size_t>(model.njoints);
    std::vector<Matrix4d> liMi(N, Matrix4d::Identity());
    std::vector<Vector6d> vel(N, Vector6d::Zero()), acc(N, Vector6d::Zero()), f(N, Vector6d::Zero());
    acc[0].head<3>() = -model.gravity;  // a_gf[0] = -gravity (linear part)
    for (int j = 1; j <= n; ++j) {
        const std::size_t ju = static_cast<std::size_t>(j);
        const PinJoint& J = model.joints[ju];
        const std::size_t par = static_cast<std::size_t>(J.parent);
        const Eigen::Index k = j - 1;
        const Vector6d S = motion_subspace(J);
        liMi[ju] = joint_transform(J, q(k));
        const Vector6d vJ = S * v(k);
        vel[ju] = motion_act_inv(liMi[ju], vel[par]) + vJ;
        acc[ju] = motion_act_inv(liMi[ju], acc[par]) + S * a(k) + motion_cross_motion(vel[ju], vJ);
        f[ju] = inertia_times(J, acc[ju]) + motion_cross_force(vel[ju], inertia_times(J, vel[ju])) - fext[ju];
    }
    VectorXd tau(n);
    for (int j = n; j >= 1; --j) {
        const std::size_t ju = static_cast<std::size_t>(j);
        const PinJoint& J = model.joints[ju];
        tau(j - 1) = motion_subspace(J).dot(f[ju]);
        if (J.parent > 0) f[static_cast<std::size_t>(J.parent)] += force_act(liMi[ju], f[ju]);
    }
    return tau;
}

}  // namespace lab
