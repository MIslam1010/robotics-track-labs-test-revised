// SBEG 461 Hardware rotation 1 (Week 8) -- the course arm (C++ track).  GIVEN code: do not edit.
//
// C++ copy of course_arm.py: the SBEG course arm (desktop UR-type 6R, myCobot-280 class) with its illustrative
// dynamics.  Numbers are IDENTICAL to the shared model (Course resources: "Course robot models and hardware safety
// checklist").  Kinematics: MR Example 4.5 layout (pp. 145-146) scaled down, W1 = 0.0634, W2 = 0.0456,
// L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751 m.  Rated speed, torques, masses and inertias are ILLUSTRATIVE
// VALUES; replace them with the commissioned / identified values published on the course site.
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <array>
#include <vector>

namespace lab {
namespace {

constexpr double G0 = 9.81;
constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;
constexpr double RATED_SPEED_DEG_S = 160.0;  // illustrative; re-measure at commissioning
constexpr double PAYLOAD_MAX = 0.25;         // kg, illustrative

struct Link {
    double m;
    std::array<double, 3> com;    // CoM at home in {s} (m)
    std::array<double, 3> edges;  // box edges a, b, c along x_s, y_s, z_s (m)
};

// link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
const std::array<Link, 6> LINKS = {{
    {0.20, {0.000, 0.010, 0.105}, {0.050, 0.060, 0.060}},   // 1 shoulder housing (turns with J1)
    {0.16, {0.055, 0.030, 0.1312}, {0.130, 0.040, 0.040}},  // 2 upper arm
    {0.12, {0.158, 0.005, 0.1312}, {0.120, 0.035, 0.035}},  // 3 forearm
    {0.08, {0.206, 0.040, 0.1312}, {0.035, 0.050, 0.035}},  // 4 wrist 1 (J5 servo)
    {0.08, {0.206, 0.0634, 0.090}, {0.035, 0.035, 0.050}},  // 5 wrist 2 (J6 servo)
    {0.03, {0.206, 0.100, 0.0561}, {0.040, 0.015, 0.040}},  // 6 flange
}};

Eigen::Matrix4d trans(const Eigen::Vector3d& p) { return mr::rp_to_trans(Eigen::Matrix3d::Identity(), p); }

/// G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32).
sbeg::Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib) {
    sbeg::Matrix6d G = sbeg::Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = Ib.asDiagonal();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

sbeg::Matrix6d box_inertia(double m, double a, double b, double c) {
    return body_inertia(m, Eigen::Vector3d(m * (b * b + c * c) / 12, m * (a * a + c * c) / 12,
                                           m * (a * a + b * b) / 12));
}

/// G_a = [Ad_Tba]^T G_b [Ad_Tba] (MR Eq. 8.42).
sbeg::Matrix6d transform_inertia(const sbeg::Matrix6d& G_b, const Eigen::Matrix4d& T_ba) {
    const sbeg::Matrix6d Ad = mr::adjoint(T_ba);
    return Ad.transpose() * G_b * Ad;
}

}  // namespace

Model course_arm_model(double payload) {
    Model r;
    r.name = "course arm";
    r.n = 6;
    Eigen::Matrix<double, 6, 6> S;  // one screw axis per ROW here, transposed below (as in Python)
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    r.Slist = S.transpose();
    r.M << -1, 0, 0, L1 + L2,
           0, 0, 1, W1 + W2,
           0, 1, 0, H1 - H2,
           0, 0, 0, 1;  // tool = 0
    std::vector<Eigen::Matrix4d> frames = {Eigen::Matrix4d::Identity()};
    for (const Link& l : LINKS) {
        frames.push_back(trans(Eigen::Vector3d(l.com[0], l.com[1], l.com[2])));
        r.Glist.push_back(box_inertia(l.m, l.edges[0], l.edges[1], l.edges[2]));
    }
    frames.push_back(r.M);
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) r.Mlist.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    if (payload != 0.0) {  // payload CoM = tool point, in the last-link CoM frame
        const Eigen::Matrix4d T_np = r.Mlist.back();
        r.Glist.back() += transform_inertia(body_inertia(payload, Eigen::Vector3d::Zero()), mr::trans_inv(T_np));
    }
    r.limits.resize(6, 2);
    for (int j = 0; j < 6; ++j) {
        const double lim = (j < 5 ? 165.0 : 175.0) * DEG;
        r.limits(j, 0) = -lim;
        r.limits(j, 1) = lim;
    }
    r.qdot_max = Eigen::VectorXd::Constant(6, RATED_SPEED_DEG_S * DEG);
    r.tau_max.resize(6);
    r.tau_max << 1.8, 3.0, 1.5, 0.6, 0.6, 0.4;  // N m, illustrative smart-servo ratings
    r.q_home.resize(6);
    r.q_home << 0.0, -0.6, 1.4, -2.4, -1.57, 0.0;
    r.g = Eigen::Vector3d(0.0, 0.0, -G0);
    r.payload_max = PAYLOAD_MAX;
    r.derated_speed_fraction = 0.5;
    r.derated_current_fraction = 0.6;
    return r;
}

}  // namespace lab
