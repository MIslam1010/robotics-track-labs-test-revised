// SBEG 461 enrichment lab "parameter identification" -- robot models (C++ track).  GIVEN code: do not edit.
//
// C++ copy of robots.py: the course arm and the EOD arm 3R in Modern Robotics list form (MR Sec. 8.3.2).
// The numbers are IDENTICAL to the shared models (Course resources: "Course robot models and hardware safety
// checklist"); the inertial data are ILLUSTRATIVE VALUES (the nominal "CAD" model that identification is meant to
// correct).  Units: m, kg, s, rad, N m.
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <array>
#include <stdexcept>
#include <vector>

namespace lab {
namespace {

constexpr double G0 = 9.81;
constexpr double DEG = PI / 180.0;

/// G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32).
Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib) {
    Matrix6d G = Matrix6d::Zero();
    G.topLeftCorner<3, 3>() = Ib.asDiagonal();
    G.bottomRightCorner<3, 3>() = m * Eigen::Matrix3d::Identity();
    return G;
}

/// Solid box with edges a, b, c along the CoM-frame x, y, z axes.
Matrix6d box_inertia(double m, double a, double b, double c) {
    return body_inertia(m, Eigen::Vector3d(m * (b * b + c * c) / 12, m * (a * a + c * c) / 12,
                                           m * (a * a + b * b) / 12));
}

/// MR's Mlist from the absolute home CoM frames M_{s,i} and the end-effector home frame.
std::vector<Eigen::Matrix4d> mlist_from_absolute(const std::vector<Eigen::Matrix4d>& com_frames,
                                                 const Eigen::Matrix4d& M_ee) {
    std::vector<Eigen::Matrix4d> frames;
    frames.push_back(Eigen::Matrix4d::Identity());
    for (const auto& F : com_frames) frames.push_back(F);
    frames.push_back(M_ee);
    std::vector<Eigen::Matrix4d> out;
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) out.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    return out;
}

// ------------------------------------------------------------------------------------------- course arm
constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;
constexpr double COURSE_ARM_RATED_SPEED_DEG_S = 160.0;  // illustrative; re-measure at commissioning

struct BoxLink {
    double m;
    std::array<double, 3> com;    // CoM at home in {s} (m)
    std::array<double, 3> edges;  // box edges a, b, c along x_s, y_s, z_s (m)
};

// link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
const std::array<BoxLink, 6> COURSE_ARM_LINKS = {{
    {0.20, {0.000, 0.010, 0.105}, {0.050, 0.060, 0.060}},   // 1 shoulder housing (turns with J1)
    {0.16, {0.055, 0.030, 0.1312}, {0.130, 0.040, 0.040}},  // 2 upper arm
    {0.12, {0.158, 0.005, 0.1312}, {0.120, 0.035, 0.035}},  // 3 forearm
    {0.08, {0.206, 0.040, 0.1312}, {0.035, 0.050, 0.035}},  // 4 wrist 1 (J5 servo)
    {0.08, {0.206, 0.0634, 0.090}, {0.035, 0.035, 0.050}},  // 5 wrist 2 (J6 servo)
    {0.03, {0.206, 0.100, 0.0561}, {0.040, 0.015, 0.040}},  // 6 flange
}};

/// Planar n-R chain in the vertical x-y plane (gravity -y), joints along z, zero pose along +x; CoM r_i from
/// joint i, Izz about the CoM; Glist = diag(Ixx_factor*Izz, Izz, Izz, m, m, m).
Robot planar_chain(const std::vector<double>& L, const std::vector<double>& m, const std::vector<double>& r,
                   const std::vector<double>& Izz, double Ixx_factor = 0.2) {
    const int n = static_cast<int>(L.size());
    std::vector<double> jx(L.size() + 1, 0.0);
    for (std::size_t i = 0; i < L.size(); ++i) jx[i + 1] = jx[i] + L[i];
    Robot R;
    R.n = n;
    std::vector<Eigen::Matrix4d> coms;
    R.Slist = Eigen::MatrixXd::Zero(6, n);
    for (int i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        coms.push_back(trans(Eigen::Vector3d(jx[k] + r[k], 0.0, 0.0)));
        R.Slist.col(i) << 0, 0, 1, 0, -jx[k], 0;
        Vector6d d;
        d << Ixx_factor * Izz[k], Izz[k], Izz[k], m[k], m[k], m[k];
        R.Glist.push_back(d.asDiagonal());
    }
    R.M = trans(Eigen::Vector3d(jx[L.size()], 0.0, 0.0));
    R.Mlist = mlist_from_absolute(coms, R.M);
    R.g = Eigen::Vector3d(0.0, -G0, 0.0);
    return R;
}

}  // namespace

Eigen::Matrix4d trans(const Eigen::Vector3d& p) { return mr::rp_to_trans(Eigen::Matrix3d::Identity(), p); }

Robot course_arm() {
    Robot R;
    R.name = "course arm";
    R.key = "course_arm";
    R.n = 6;
    Eigen::Matrix<double, 6, 6> S;  // rows = screw axes (transposed below, as in robots.py)
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    R.Slist = S.transpose();
    R.M << -1, 0, 0, L1 + L2,
           0, 0, 1, W1 + W2,
           0, 1, 0, H1 - H2,
           0, 0, 0, 1;
    R.limits.resize(6, 2);
    for (int i = 0; i < 6; ++i) {
        const double lim = (i < 5 ? 165.0 : 175.0) * DEG;
        R.limits(i, 0) = -lim;
        R.limits(i, 1) = lim;
    }
    R.qdot_max = Eigen::VectorXd::Constant(6, COURSE_ARM_RATED_SPEED_DEG_S * DEG);
    R.tau_max.resize(6);
    R.tau_max << 1.8, 3.0, 1.5, 0.6, 0.6, 0.4;  // N m, illustrative smart-servo ratings
    R.q_home.resize(6);
    R.q_home << 0.0, -0.6, 1.4, -2.4, -1.57, 0.0;
    std::vector<Eigen::Matrix4d> coms;
    for (const auto& l : COURSE_ARM_LINKS) {
        coms.push_back(trans(Eigen::Vector3d(l.com[0], l.com[1], l.com[2])));
        R.Glist.push_back(box_inertia(l.m, l.edges[0], l.edges[1], l.edges[2]));
    }
    R.Mlist = mlist_from_absolute(coms, R.M);
    R.g = Eigen::Vector3d(0.0, 0.0, -G0);
    return R;
}

Robot eod_3r() {
    Robot R = planar_chain({0.60, 0.50, 0.15}, {8.0, 5.0, 3.5}, {0.30, 0.25, 0.10}, {0.24, 0.10, 0.008});
    R.name = "EOD arm 3R";
    R.key = "eod_3r";
    R.limits.resize(3, 2);
    R.limits << -1.0, 2.0, -2.6, 2.6, -2.0, 2.0;
    R.qdot_max = Eigen::VectorXd();  // None in robots.py
    R.tau_max = Eigen::Vector3d(155.0, 70.0, 15.0);  // N m, illustrative
    R.q_home = Eigen::VectorXd::Zero(3);
    return R;
}

Robot get_robot(const std::string& key) {
    if (key == "course_arm") return course_arm();
    if (key == "eod_3r") return eod_3r();
    throw std::invalid_argument("unknown robot key: " + key);
}

}  // namespace lab
