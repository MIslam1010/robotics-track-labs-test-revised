// SBEG 46X Lab 9 -- provided helpers (C++ track).  GIVEN code: do not edit; use these functions freely.
//
// The "Provided" part of starter.py, translated to C++:
//   load_meta, load_recording, rms_deg (with and without mask), quat_from_accel, heading_from_quat
// plus, C++ only, accel_tilt(double, double) for a single sample (it calls your vector version).
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {
namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

ImuMeta load_meta(const std::string& path) {
    const sbeg::json J = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/imu_meta.json") : path);
    ImuMeta m;
    m.h = J.at("h").get<double>();
    m.g = J.at("g").get<double>();
    m.r_imu = J.at("r_imu").get<double>();
    const sbeg::json& s = J.at("imu_spec");
    m.gyro_noise_rad_s = s.at("gyro_noise_rad_s").get<double>();
    m.acc_noise_m_s2 = s.at("acc_noise_m_s2").get<double>();
    m.gyro_bias_class_rad_s = s.at("gyro_bias_class_rad_s").get<double>();
    for (const auto& p : J.at("phases")) {
        Phase ph;
        ph.name = p.at("name").get<std::string>();
        ph.t0 = p.at("t0").get<double>();
        ph.t1 = p.at("t1").get<double>();
        m.phases.push_back(ph);
    }
    return m;
}

Recording load_recording(const std::string& path) {
    const sbeg::CsvTable c = sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/thigh_imu.csv") : path);
    Recording r;
    r.t = c.data.col(c.col("t"));
    r.gx = c.data.col(c.col("gx"));
    r.gy = c.data.col(c.col("gy"));
    r.gz = c.data.col(c.col("gz"));
    r.ax = c.data.col(c.col("ax"));
    r.ay = c.data.col(c.col("ay"));
    r.az = c.data.col(c.col("az"));
    r.theta_enc = c.data.col(c.col("theta_enc"));
    return r;
}

double rms_deg(const Eigen::VectorXd& est, const Eigen::VectorXd& ref) {
    if (est.size() != ref.size() || est.size() == 0)
        throw std::invalid_argument("rms_deg: est and ref must have the same, non-zero length");
    const Eigen::VectorXd e = est - ref;
    return std::sqrt(e.squaredNorm() / static_cast<double>(e.size())) * 180.0 / PI;
}

double rms_deg(const Eigen::VectorXd& est, const Eigen::VectorXd& ref, const std::vector<bool>& mask) {
    if (est.size() != ref.size() || static_cast<std::size_t>(est.size()) != mask.size())
        throw std::invalid_argument("rms_deg: est, ref and mask must have the same length");
    double s = 0.0;
    int n = 0;
    for (Eigen::Index i = 0; i < est.size(); ++i) {
        if (!mask[static_cast<std::size_t>(i)]) continue;
        const double e = est(i) - ref(i);
        s += e * e;
        ++n;
    }
    if (n == 0) throw std::invalid_argument("rms_deg: empty mask");
    return std::sqrt(s / n) * 180.0 / PI;
}

Eigen::Vector4d quat_from_accel(const Eigen::Vector3d& acc) {
    const Eigen::Vector3d a = acc / acc.norm();
    const Eigen::Vector3d up(0.0, 1.0, 0.0);
    // R_sb^T up = a  <=>  R_sb a = up: rotate a onto up about axis a x up
    const Eigen::Vector3d ax = a.cross(up);
    const double s = ax.norm(), c = a.dot(up);
    if (s < 1e-12) return c > 0 ? Eigen::Vector4d(1.0, 0.0, 0.0, 0.0) : Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    const double phi = std::atan2(s, c);
    Eigen::Vector4d q;
    q << std::cos(phi / 2), ax / s * std::sin(phi / 2);
    return q;
}

double heading_from_quat(const Eigen::Vector4d& q) {
    const Eigen::Vector3d z = quat_to_rot(q).col(2);
    return std::atan2(z(0), z(2));
}

double accel_tilt(double ax, double ay) {
    const Eigen::VectorXd th = accel_tilt(Eigen::VectorXd::Constant(1, ax), Eigen::VectorXd::Constant(1, ay));
    if (th.size() != 1) throw std::invalid_argument("accel_tilt must return one value per sample");
    return th(0);
}

}  // namespace lab
