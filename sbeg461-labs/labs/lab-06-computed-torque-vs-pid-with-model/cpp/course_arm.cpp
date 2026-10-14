// SBEG 461 Lab 6 -- C++ track: GIVEN code (do not edit).  The course arm (C++ copy of course_arm.py).
#include "course_arm.hpp"

#include <array>
#include <vector>

namespace course_arm {

namespace {
struct Link {
    double mass;          // kg
    Eigen::Vector3d com;  // CoM at home in {s} [m]
    Eigen::Vector3d box;  // box edges a, b, c [m] along x_s, y_s, z_s
};

// link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
const std::array<Link, 6>& links() {
    static const std::array<Link, 6> L = {{
        {0.20, Eigen::Vector3d(0.000, 0.010, 0.105), Eigen::Vector3d(0.050, 0.060, 0.060)},   // 1 shoulder housing
        {0.16, Eigen::Vector3d(0.055, 0.030, 0.1312), Eigen::Vector3d(0.130, 0.040, 0.040)},  // 2 upper arm
        {0.12, Eigen::Vector3d(0.158, 0.005, 0.1312), Eigen::Vector3d(0.120, 0.035, 0.035)},  // 3 forearm
        {0.08, Eigen::Vector3d(0.206, 0.040, 0.1312), Eigen::Vector3d(0.035, 0.050, 0.035)},  // 4 wrist 1 (J5 servo)
        {0.08, Eigen::Vector3d(0.206, 0.0634, 0.090), Eigen::Vector3d(0.035, 0.035, 0.050)},  // 5 wrist 2 (J6 servo)
        {0.03, Eigen::Vector3d(0.206, 0.100, 0.0561), Eigen::Vector3d(0.040, 0.015, 0.040)},  // 6 flange
    }};
    return L;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p) { return mr::rp_to_trans(Eigen::Matrix3d::Identity(), p); }

mr::TransformList mlist_from_absolute(const mr::TransformList& com_frames, const Eigen::Matrix4d& M_ee) {
    mr::TransformList frames;
    frames.push_back(Eigen::Matrix4d::Identity());
    for (const auto& T : com_frames) frames.push_back(T);
    frames.push_back(M_ee);
    mr::TransformList out;
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) out.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    return out;
}
}  // namespace

mr::Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib_diag) {
    mr::Matrix6d G = mr::Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = Ib_diag.asDiagonal();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

mr::Matrix6d box_inertia(double m, double a, double b, double c) {
    return body_inertia(m, Eigen::Vector3d(m * (b * b + c * c) / 12, m * (a * a + c * c) / 12,
                                           m * (a * a + b * b) / 12));
}

mr::Matrix6d transform_inertia(const mr::Matrix6d& G_b, const Eigen::Matrix4d& T_ba) {
    const mr::Matrix6d Ad = mr::adjoint(T_ba);
    return Ad.transpose() * G_b * Ad;
}

Robot course_arm(double tool, double payload) {
    const double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;
    Eigen::Matrix<double, 6, 6> St;  // rows = screw axes (as written in course_arm.py, before .T)
    St << 0, 0, 1, 0, 0, 0,
          0, 1, 0, -H1, 0, 0,
          0, 1, 0, -H1, 0, L1,
          0, 1, 0, -H1, 0, L1 + L2,
          0, 0, -1, -W1, L1 + L2, 0,
          0, 1, 0, H2 - H1, 0, L1 + L2;
    Eigen::Matrix4d M;
    M << -1, 0, 0, L1 + L2,
         0, 0, 1, W1 + W2,
         0, 1, 0, H1 - H2,
         0, 0, 0, 1;
    M = (M * trans(Eigen::Vector3d(0.0, 0.0, tool))).eval();

    mr::TransformList coms;
    mr::InertiaList Glist;
    for (const Link& l : links()) {
        coms.push_back(trans(l.com));
        Glist.push_back(box_inertia(l.mass, l.box(0), l.box(1), l.box(2)));
    }
    mr::TransformList Mlist = mlist_from_absolute(coms, M);
    if (payload != 0.0) {
        const Eigen::Matrix4d T_np = Mlist.back();  // payload CoM = tool point, in the last-link CoM frame
        Glist.back() = Glist.back() + transform_inertia(body_inertia(payload, Eigen::Vector3d::Zero()),
                                                        mr::trans_inv(T_np));
    }

    Robot r;
    r.name = "course arm";
    r.n = 6;
    r.Slist = St.transpose();
    r.M = M;
    r.limits.resize(6, 2);
    for (int i = 0; i < 5; ++i) r.limits.row(i) << -165.0 * DEG, 165.0 * DEG;
    r.limits.row(5) << -175.0 * DEG, 175.0 * DEG;
    r.qdot_max = Eigen::VectorXd::Constant(6, COURSE_ARM_RATED_SPEED_DEG_S * DEG);
    r.tau_max.resize(6);
    r.tau_max << 1.8, 3.0, 1.5, 0.6, 0.6, 0.4;  // N m, illustrative smart-servo ratings
    r.q_home.resize(6);
    r.q_home << 0.0, -0.6, 1.4, -2.4, -1.57, 0.0;
    r.Mlist = Mlist;
    r.Glist = Glist;
    r.g = Eigen::Vector3d(0.0, 0.0, -G0);
    r.payload_max = COURSE_ARM_PAYLOAD_MAX;
    r.derated_speed_fraction = 0.5;
    r.derated_current_fraction = 0.6;
    return r;
}

}  // namespace course_arm
