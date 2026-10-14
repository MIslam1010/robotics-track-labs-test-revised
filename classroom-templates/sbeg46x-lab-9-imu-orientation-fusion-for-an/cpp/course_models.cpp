// SBEG 46X Lab 9 -- robot model (C++ track).  GIVEN code: do not edit.
//
// C++ copy of course_models.py, numbers identical:
//   gait_exo_2r()  Hip-knee gait exoskeleton 2R (SBEG 461 project scenario S2): swing leg on a test stand, leg along
//                  -y_s at q = 0 (hanging), hip flexion about +z_s, knee flexion about -z_s, gravity along -y_s.
//                  Lab 9 uses its THIGH segment (joint 1, the hip): the thigh cuff carries a 6-axis IMU.
//   imu_home, imu_pose (product of exponentials, MR Eq. 4.14, via mrcpp fkin_space) and imu_specific_force (the
//   ideal accelerometer reading of that IMU).
// ILLUSTRATIVE VALUES: geometry and inertias are teaching values, not data of a real product or person.
// Notation: Modern Robotics (MR) -- screw axes S = (w, v), T_sb = e^{[S1] theta} M_imu.
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

namespace lab {
namespace {
constexpr double G0 = 9.81;

/// Screw axis S = (w, -w x q + h w) of a joint about unit axis w through point q (MR Sec. 3.3.2).
mr::Vector6d screw_revolute(const Eigen::Vector3d& w, const Eigen::Vector3d& q, double h = 0.0) {
    mr::Vector6d S;
    S << w, -w.cross(q) + h * w;
    return S;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity()) {
    return mr::rp_to_trans(R, p);
}

/// G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32).
mr::Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib) {
    mr::Matrix6d G = mr::Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = Ib.asDiagonal();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

std::vector<Eigen::Matrix4d> mlist_from_absolute(const std::vector<Eigen::Matrix4d>& com_frames,
                                                 const Eigen::Matrix4d& M_ee) {
    std::vector<Eigen::Matrix4d> frames;
    frames.push_back(Eigen::Matrix4d::Identity());
    frames.insert(frames.end(), com_frames.begin(), com_frames.end());
    frames.push_back(M_ee);
    std::vector<Eigen::Matrix4d> out;
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) out.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    return out;
}
}  // namespace

ExoModel gait_exo_2r() {
    const double L1 = 0.42, L2 = 0.43;
    const double m1 = 9.0, c1 = 0.18, I1 = 0.16;
    const double m2 = 5.8, c2 = 0.24, I2 = 0.10;
    ExoModel r;
    r.key = "gait_exo_2r";
    r.name = "Hip-knee gait exoskeleton 2R";
    r.Slist.resize(6, 2);
    r.Slist.col(0) = screw_revolute(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
    r.Slist.col(1) = screw_revolute(Eigen::Vector3d(0, 0, -1), Eigen::Vector3d(0, -L1, 0));
    r.M = trans(Eigen::Vector3d(0, -(L1 + L2), 0));
    const std::vector<Eigen::Matrix4d> com = {trans(Eigen::Vector3d(0, -c1, 0)),
                                              trans(Eigen::Vector3d(0, -(L1 + c2), 0))};
    r.Mlist = mlist_from_absolute(com, r.M);
    r.Glist = {body_inertia(m1, Eigen::Vector3d(I1, 0.03, I1)), body_inertia(m2, Eigen::Vector3d(I2, 0.01, I2))};
    r.g = Eigen::Vector3d(0.0, -G0, 0.0);
    r.limits.resize(2, 2);
    r.limits << -0.5, 1.2,
                0.0, 1.6;
    r.qdot_max = Eigen::Vector2d(8.0, 10.0);
    r.tau_max = Eigen::Vector2d(60.0, 45.0);
    r.L1 = L1;
    r.L2 = L2;
    r.gear_ratio = 80.0;
    r.rotor_inertia = 1.5e-5;
    return r;
}

Eigen::Matrix4d imu_home(double r_imu) {
    return trans(Eigen::Vector3d(0.0, -r_imu, 0.0), Eigen::Matrix3d::Identity());  // R_MOUNT = I
}

Eigen::Matrix4d imu_pose(double theta_hip, double r_imu) {
    const Eigen::MatrixXd S1 = gait_exo_2r().Slist.leftCols(1);
    return mr::fkin_space(imu_home(r_imu), S1, Eigen::VectorXd::Constant(1, theta_hip));
}

Eigen::Vector3d imu_specific_force(double theta, double dtheta, double ddtheta, double r_imu) {
    const auto [R, p] = mr::trans_to_rp(imu_pose(theta, r_imu));
    const ExoModel model = gait_exo_2r();
    const Eigen::Vector3d w1 = model.Slist.col(0).head<3>();
    const Eigen::Vector3d w = w1 * dtheta;
    const Eigen::Vector3d alpha = w1 * ddtheta;
    const Eigen::Vector3d a_s = alpha.cross(p) + w.cross(w.cross(p));  // hip axis passes through the {s} origin
    return R.transpose() * (a_s - model.g);
}

}  // namespace lab
