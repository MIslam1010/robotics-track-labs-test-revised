// SBEG 361 Hardware rotation 1 (Week 8) -- C++ track: the GIVEN station code (do not edit).
//
// C++ copies of the given Python files of this lab, for the C++ tests and your own experiments:
//   * course_arm_reference / course_arm_dynamics   = course_arm_model.py (the course arm; tests only -- do not call
//                                                    them in starter.cpp, exactly as in Python)
//   * SafetyViolation, SafetyGuard                 = safety_guard.py (independent software guard)
//   * Arm                                          = the interface shared by the emulator and the real arm
//   * SimArm                                       = sim_arm.py (emulated arm: lag, gravity droop, backlash, zero error,
//                                                    12-bit quantisation, latched stop; virtual time)
// The real course arm is driven only from Python (run_session.py + vendor_arm.py): the vendor SDK is a Python package.
// Rated speed, limits, masses and all emulator parameters are ILLUSTRATIVE VALUES, as in the Python files.
#ifndef SBEG361_W08_ARM_STATION_HPP
#define SBEG361_W08_ARM_STATION_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// ------------------------------------------------------------------ course arm (course_arm_model.py)
/// Encoder resolution of the 12-bit servos (degrees).
constexpr double ENC_RES_DEG = 360.0 / 4096.0;

/// Home configuration M and space screw axes Slist (6 x 6, axes are the COLUMNS) of the course arm;
/// `tool` moves {b} along z_b (pointer / needle guide: 0.06 m).
struct CourseArm {
    Eigen::Matrix4d M;
    Eigen::MatrixXd Slist;
};
CourseArm course_arm_reference(double tool = 0.0);

/// (Mlist, Glist, g) of the course arm in MR list form (MR Sec. 8.3.2), as course_arm_dynamics() in Python.
struct CourseArmDynamics {
    std::vector<Eigen::Matrix4d> Mlist;
    std::vector<Eigen::Matrix<double, 6, 6>> Glist;
    Eigen::Vector3d g;
};
CourseArmDynamics course_arm_dynamics();

// ------------------------------------------------------------------ safety guard (safety_guard.py)
/// Raised when a command would leave the software limits or exceed the speed cap (Python: SafetyViolation).
class SafetyViolation : public std::runtime_error {
public:
    explicit SafetyViolation(const std::string& what) : std::runtime_error(what) {}
};

/// Independent software safety layer: software joint limits (hardware limits minus 10 deg), command-rate limit
/// (speed_fraction x rated speed, 20 % tolerance), latched stop until reset().  Angles in SERVO degrees.
class SafetyGuard {
public:
    explicit SafetyGuard(const Eigen::VectorXd& offsets_deg, double limit_margin_deg = 10.0,
                         double rated_speed_deg_s = 160.0, double speed_fraction = 0.25, double tolerance = 1.2);
    void trip(const std::string& reason);
    /// Operator reset after a stop; with a pose, restart the rate check from it (t, q_servo_deg).
    void reset();
    void reset(double t, const Eigen::VectorXd& q_servo_deg);
    /// Return the command unchanged if it is safe; otherwise trip and throw SafetyViolation.
    Eigen::VectorXd check(double t, const Eigen::VectorXd& q_servo_deg);

    Eigen::MatrixXd soft_limits;  ///< 6 x 2, model degrees
    double speed_cap;             ///< deg/s
    bool tripped = false;
    std::string reason;

private:
    Eigen::VectorXd offsets_;
    double tolerance_;
    bool has_last_ = false;
    double last_t_ = 0.0;
    Eigen::VectorXd last_q_;
};

// ------------------------------------------------------------------ arm interface (SimArm / VendorArm)
/// What run_logged needs from an arm.  Angles in SERVO degrees, time in seconds.
class Arm {
public:
    virtual ~Arm() = default;
    /// Wait until time t (emulator: advance the virtual clock).
    virtual void wait_until(double t) = 0;
    /// Send a joint command; throws SafetyViolation if the guard refuses it; returns false while a stop is latched.
    virtual bool send(const Eigen::VectorXd& q_servo_deg) = 0;
    /// Angles reported by the servos.
    virtual Eigen::VectorXd read() = 0;
    /// Latched stop.
    virtual void stop() = 0;
    /// Current time (s).
    virtual double time() const { return 0.0; }
    /// Operator reset after a stop.
    virtual void reset() {}
};

// ------------------------------------------------------------------ emulator (sim_arm.py)
/// Emulated course arm (arm ids A5, S1 ... S6).  Same behaviour as sim_arm.SimArm; the encoder noise uses the C++
/// random generator, so individual noisy readings differ from Python (statistics are the same).
class SimArm : public Arm {
public:
    /// start_servo_deg empty: start at the ready pose R.  Throws std::invalid_argument for an unknown arm id.
    SimArm(const std::string& arm_id, const Eigen::VectorXd& offsets_deg,
           const Eigen::VectorXd& start_servo_deg = Eigen::VectorXd(), bool noise = true);
    SimArm& connect();
    void close();
    double time() const override;
    void wait_until(double t) override;
    bool send(const Eigen::VectorXd& q_servo_deg) override;
    Eigen::VectorXd read() override;
    void stop() override;
    void reset() override;
    /// TRUE link angles in model radians (backlash and zero error included; hidden from the encoder).
    Eigen::VectorXd true_model_angles() const;
    /// Tool point in {s} (mm) as measured on the grid / scale photos (noise 0.4 mm per axis, rounded to 0.1 mm).
    Eigen::Vector3d measure_tip(const Eigen::Vector3d& p_tool = Eigen::Vector3d(0.0, 0.0, 0.06));

    SafetyGuard guard;
    bool connected = false;
    bool stopped = false;

private:
    Eigen::VectorXd gravity_droop() const;
    Eigen::VectorXd velocity() const;
    void step(double h);

    std::string arm_id_;
    Eigen::VectorXd offsets_, zero_err_, backlash_;
    bool noise_;
    std::mt19937 rng_;
    CourseArm arm_;
    CourseArmDynamics dyn_;
    Eigen::VectorXd motor_, link_, target_, droop_;
    double t_ = 0.0;
    long k_ = 0;
};

}  // namespace lab

#endif  // SBEG361_W08_ARM_STATION_HPP
