// SBEG 361 Hardware rotation 1 (Week 8) -- C++ track: the GIVEN station code (do not edit).
// C++ copies of course_arm_model.py, safety_guard.py and sim_arm.py; see arm_station.hpp.
#include "arm_station.hpp"

#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;

struct ArmParams {
    double zero_err[6];
    double backlash[6];
    unsigned seed;
};

// hidden parameters of the emulated arms (degrees) -- ILLUSTRATIVE VALUES, identical to sim_arm.ARMS
const std::map<std::string, ArmParams>& arms() {
    static const std::map<std::string, ArmParams> A = {
        {"A5", {{0.0, -0.8, 1.2, 0.0, 0.0, 0.0}, {0.2, 0.5, 0.6, 0.3, 0.3, 0.3}, 8}},
        {"S1", {{0.0, 0.0, 1.5, 0.0, 0.0, 0.0}, {0.2, 0.4, 0.5, 0.3, 0.3, 0.3}, 11}},
        {"S2", {{0.0, -1.2, 0.0, 0.0, 0.0, 0.0}, {0.2, 0.6, 0.4, 0.3, 0.3, 0.3}, 12}},
        {"S3", {{0.0, 0.7, -0.9, 0.0, 0.0, 0.0}, {0.3, 0.5, 0.5, 0.4, 0.3, 0.3}, 13}},
        {"S4", {{0.0, 0.0, -1.3, 0.0, 0.0, 0.0}, {0.2, 0.8, 0.6, 0.3, 0.3, 0.3}, 14}},
        {"S5", {{0.0, 1.0, 0.6, 0.0, 0.0, 0.0}, {0.2, 0.4, 0.4, 0.3, 0.3, 0.3}, 15}},
        {"S6", {{0.0, -0.5, -0.5, 0.0, 0.0, 0.0}, {0.2, 0.5, 0.7, 0.3, 0.3, 0.3}, 16}},
    };
    return A;
}

const double STIFFNESS[6] = {40.0, 80.0, 60.0, 30.0, 30.0, 30.0};  // N m / rad
constexpr double TAU_SERVO = 0.05;                                 // s
constexpr double RATED_DEG_S = 160.0;
constexpr double NOISE_DEG = 0.05;
constexpr double PROBE_SIGMA_MM = 0.4;
constexpr double DT = 0.002;
constexpr long GRAVITY_EVERY = 5;

// link: mass (kg), CoM at home in {s} (m), box edges a, b, c (m) -- ILLUSTRATIVE, as COURSE_ARM_LINKS
struct Link {
    double m;
    double c[3];
    double e[3];
};
const Link LINKS[6] = {
    {0.20, {0.000, 0.010, 0.105}, {0.050, 0.060, 0.060}},
    {0.16, {0.055, 0.030, 0.1312}, {0.130, 0.040, 0.040}},
    {0.12, {0.158, 0.005, 0.1312}, {0.120, 0.035, 0.035}},
    {0.08, {0.206, 0.040, 0.1312}, {0.035, 0.050, 0.035}},
    {0.08, {0.206, 0.0634, 0.090}, {0.035, 0.035, 0.050}},
    {0.03, {0.206, 0.100, 0.0561}, {0.040, 0.015, 0.040}},
};

Eigen::VectorXd from6(const double* a) {
    Eigen::VectorXd v(6);
    for (int i = 0; i < 6; ++i) v(i) = a[i];
    return v;
}

std::string fmt1(double x) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << x;
    return s.str();
}
}  // namespace

// ------------------------------------------------------------------ course arm
CourseArm course_arm_reference(double tool) {
    const double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;
    Eigen::Matrix<double, 6, 6> St;  // rows = screw axes (as the Python array before .T)
    St << 0, 0, 1, 0, 0, 0,
          0, 1, 0, -H1, 0, 0,
          0, 1, 0, -H1, 0, L1,
          0, 1, 0, -H1, 0, L1 + L2,
          0, 0, -1, -W1, L1 + L2, 0,
          0, 1, 0, H2 - H1, 0, L1 + L2;
    CourseArm r;
    r.Slist = St.transpose();
    r.M << -1, 0, 0, L1 + L2,
            0, 0, 1, W1 + W2,
            0, 1, 0, H1 - H2,
            0, 0, 0, 1;
    Eigen::Matrix4d Tt = Eigen::Matrix4d::Identity();
    Tt(2, 3) = tool;
    r.M = (r.M * Tt).eval();
    return r;
}

CourseArmDynamics course_arm_dynamics() {
    const Eigen::Matrix4d M = course_arm_reference().M;
    std::vector<Eigen::Matrix4d> frames{Eigen::Matrix4d::Identity()};
    for (const Link& l : LINKS) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 1>(0, 3) << l.c[0], l.c[1], l.c[2];
        frames.push_back(T);
    }
    frames.push_back(M);
    CourseArmDynamics d;
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) d.Mlist.push_back(mr::trans_inv(frames[i]) * frames[i + 1]);
    for (const Link& l : LINKS) {
        const double a = l.e[0], b = l.e[1], c = l.e[2];
        Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
        G(0, 0) = l.m * (b * b + c * c) / 12;
        G(1, 1) = l.m * (a * a + c * c) / 12;
        G(2, 2) = l.m * (a * a + b * b) / 12;
        G.bottomRightCorner<3, 3>() = l.m * Eigen::Matrix3d::Identity();
        d.Glist.push_back(G);
    }
    d.g = Eigen::Vector3d(0.0, 0.0, -9.81);
    return d;
}

// ------------------------------------------------------------------ safety guard
SafetyGuard::SafetyGuard(const Eigen::VectorXd& offsets_deg, double limit_margin_deg, double rated_speed_deg_s,
                         double speed_fraction, double tolerance)
    : soft_limits(6, 2), speed_cap(speed_fraction * rated_speed_deg_s), offsets_(offsets_deg),
      tolerance_(tolerance) {
    if (offsets_deg.size() != 6) throw std::invalid_argument("SafetyGuard: six offsets expected");
    for (int j = 0; j < 6; ++j) {
        const double hw = j < 5 ? 165.0 : 175.0;  // hardware limits (model degrees)
        soft_limits(j, 0) = -hw + limit_margin_deg;
        soft_limits(j, 1) = hw - limit_margin_deg;
    }
}

void SafetyGuard::trip(const std::string& why) {
    tripped = true;
    reason = why;
}

void SafetyGuard::reset() {
    tripped = false;
    reason.clear();
    has_last_ = false;
}

void SafetyGuard::reset(double t, const Eigen::VectorXd& q_servo_deg) {
    reset();
    if (q_servo_deg.size() == 6 && q_servo_deg.allFinite()) {  // a glitch reading cannot seed the rate check
        has_last_ = true;
        last_t_ = t;
        last_q_ = q_servo_deg;
    }
}

Eigen::VectorXd SafetyGuard::check(double t, const Eigen::VectorXd& q) {
    if (tripped) throw SafetyViolation("guard is latched (" + reason + "); operator reset required");
    if (q.size() != 6 || !q.allFinite()) {
        trip("command is not six finite joint angles");
        throw SafetyViolation(reason);
    }
    const Eigen::VectorXd qm = q - offsets_;
    for (int j = 0; j < 6; ++j) {
        if (qm(j) < soft_limits(j, 0) || qm(j) > soft_limits(j, 1)) {
            trip("joint " + std::to_string(j + 1) + " command " + fmt1(qm(j)) +
                 " deg (model) outside software limits");
            throw SafetyViolation(reason);
        }
    }
    if (has_last_) {
        const double dt = std::max(t - last_t_, 1e-3);  // same or earlier time stamp: no free jump
        for (int j = 0; j < 6; ++j) {
            const double rate = std::abs(q(j) - last_q_(j)) / dt;
            if (rate > tolerance_ * speed_cap + 1e-9) {
                // report the joint with the largest rate, as np.argmax in Python
                Eigen::Index jm = 0;
                ((q - last_q_).cwiseAbs() / dt).maxCoeff(&jm);
                const double rm = std::abs(q(jm) - last_q_(jm)) / dt;
                trip("joint " + std::to_string(jm + 1) + " command rate " + fmt1(rm) + " deg/s > cap " +
                     fmt1(speed_cap) + " deg/s");
                throw SafetyViolation(reason);
            }
        }
    }
    has_last_ = true;
    last_t_ = t;
    last_q_ = q;
    return q;
}

// ------------------------------------------------------------------ emulator
SimArm::SimArm(const std::string& arm_id, const Eigen::VectorXd& offsets_deg, const Eigen::VectorXd& start_servo_deg,
               bool noise)
    : guard(offsets_deg), arm_id_(arm_id), offsets_(offsets_deg), noise_(noise),
      arm_(course_arm_reference()), dyn_(course_arm_dynamics()) {
    const auto it = arms().find(arm_id);
    if (it == arms().end()) throw std::invalid_argument("unknown emulated arm '" + arm_id + "'; choose A5 or S1...S6");
    zero_err_ = from6(it->second.zero_err);
    backlash_ = from6(it->second.backlash);
    rng_.seed(it->second.seed);
    Eigen::VectorXd q0(6);
    if (start_servo_deg.size() == 0) {
        q0 << 0.0, -60.0, 90.0, -120.0, -90.0, 0.0;
        q0 += offsets_;
    } else {
        if (start_servo_deg.size() != 6) throw std::invalid_argument("SimArm: six start angles expected");
        q0 = start_servo_deg;
    }
    motor_ = link_ = target_ = q0;
    droop_ = gravity_droop();
    motor_ = target_ - droop_;  // power-up: the servos have settled under gravity
    link_ = motor_;
    droop_ = gravity_droop();
}

SimArm& SimArm::connect() {
    connected = true;
    return *this;
}

void SimArm::close() { connected = false; }

double SimArm::time() const { return t_; }

bool SimArm::send(const Eigen::VectorXd& q_servo_deg) {
    if (stopped) return false;
    target_ = guard.check(t_, q_servo_deg);
    return true;
}

Eigen::VectorXd SimArm::read() {
    Eigen::VectorXd q = motor_;
    if (noise_) {
        std::normal_distribution<double> n(0.0, NOISE_DEG);
        for (int i = 0; i < 6; ++i) q(i) += n(rng_);
    }
    for (int i = 0; i < 6; ++i) q(i) = std::round(q(i) / ENC_RES_DEG) * ENC_RES_DEG;
    return q;
}

void SimArm::stop() {
    target_ = motor_ + velocity() * 0.03;
    stopped = true;
    guard.trip("operator stop");
}

void SimArm::reset() {
    stopped = false;
    target_ = motor_;
    guard.reset(t_, motor_);
}

void SimArm::wait_until(double t) {
    while (t_ < t - 1e-12) step(std::min(DT, t - t_));
}

Eigen::VectorXd SimArm::true_model_angles() const { return (link_ + zero_err_ - offsets_) * (PI / 180.0); }

Eigen::Vector3d SimArm::measure_tip(const Eigen::Vector3d& p_tool) {
    const Eigen::Matrix4d T = mr::fkin_space(arm_.M, arm_.Slist, true_model_angles());
    Eigen::Vector3d p = 1e3 * (T.topLeftCorner<3, 3>() * p_tool + T.topRightCorner<3, 1>());
    if (noise_) {
        std::normal_distribution<double> n(0.0, PROBE_SIGMA_MM);
        for (int i = 0; i < 3; ++i) p(i) += n(rng_);
    }
    for (int i = 0; i < 3; ++i) p(i) = std::round(p(i) * 10.0) / 10.0;
    return p;
}

Eigen::VectorXd SimArm::gravity_droop() const {
    const Eigen::VectorXd tau = mr::gravity_forces(true_model_angles(), dyn_.g, dyn_.Mlist, dyn_.Glist, arm_.Slist);
    Eigen::VectorXd d(6);
    for (int i = 0; i < 6; ++i) d(i) = tau(i) / STIFFNESS[i] * (180.0 / PI);
    return d;
}

Eigen::VectorXd SimArm::velocity() const {
    const Eigen::VectorXd v = (target_ - droop_ - motor_) / TAU_SERVO;
    return v.cwiseMax(-RATED_DEG_S).cwiseMin(RATED_DEG_S);
}

void SimArm::step(double h) {
    if (k_ % GRAVITY_EVERY == 0) droop_ = gravity_droop();
    motor_ = (motor_ + h * velocity()).eval();
    for (int i = 0; i < 6; ++i) {
        const double half = backlash_(i) / 2.0, gap = motor_(i) - link_(i);
        if (gap > half) link_(i) = motor_(i) - half;
        else if (gap < -half) link_(i) = motor_(i) + half;
    }
    t_ += h;
    ++k_;
}

}  // namespace lab
