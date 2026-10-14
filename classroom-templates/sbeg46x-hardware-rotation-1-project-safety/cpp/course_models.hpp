// SBEG 46X Hardware rotation 1 (Week 8) -- C++ track: local copy of the course hardware models (GIVEN, do not edit).
//
// C++ copy of course_models.py; numbers identical to Course resources: "Course robot models and hardware safety
// checklist" (Hapkit-type device, course-arm checklist) and to the project starter package (devices.HAPKIT,
// devices.IMU). All values are ILLUSTRATIVE teaching values, not the data of a product.
//
//   hapkit()            -- Hapkit-type 1-DOF haptic paddle (geometry, mass properties, derated limits)
//   handle_model()      -- the handle-level model m_h x'' + b_h x' + k_g x = F used on the station (nominal values)
//   IMU                 -- low-cost 6-axis IMU module (bias and noise class, sample rate)
//   COURSE_ARM_SAFETY   -- the course-arm hardware safety checklist values (rules A1-A7 and the current limit)
//   HAPTIC_SAFETY       -- the haptic-device limits (rule B3)
#ifndef SBEG46X_LAB08_COURSE_MODELS_HPP
#define SBEG46X_LAB08_COURSE_MODELS_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <string>
#include <vector>

namespace lab {

constexpr double G0 = 9.81;
constexpr double PI = 3.14159265358979323846;
constexpr double DEG = PI / 180.0;

/// Hapkit-type 1-DOF haptic paddle (Python: the dict returned by course_models.hapkit(), same field names).
struct Hapkit {
    std::string key;
    bool illustrative = true;
    double rh = 0.0, rs = 0.0, rp = 0.0, ratio = 0.0;  ///< grip radius, sector radius, pulley radius (m); rs / rp
    double mass = 0.0, com_below_pivot = 0.0;          ///< moving mass (kg); CoM 10 mm below the pivot (m)
    Eigen::Matrix3d I_com = Eigen::Matrix3d::Zero();   ///< rotational inertia about the CoM (kg m^2)
    double rotor_inertia = 0.0;                        ///< motor rotor inertia (kg m^2)
    double force_max = 0.0, x_range = 0.0, loop_rate_hz = 0.0;
};

/// Hapkit-type 1-DOF haptic paddle.  Handle pivots about z_s; the grip is rh = 0.090 m from the pivot;
/// sector radius rs = 0.073 m, motor pulley rp = 0.0042 m (ratio rs/rp ~ 17.4).  Handle displacement x ~ rh*theta.
/// Moving mass 0.10 kg with its centre of mass 10 mm below the pivot; rotational inertia about the CoM
/// diag(2.0e-4, 1.0e-4, 1.5e-4) kg m^2; motor rotor inertia 1.0e-6 kg m^2.
/// Derated limits: handle force <= 2 N (saturated in firmware), watchdog cuts the motor outside +-40 mm.
Hapkit hapkit();

/// Handle-level model of one device (Python: the dict returned by course_models.handle_model()).
struct HandleModel {
    double m_h = 0.0, b_h = 0.0, k_g = 0.0, T = 0.0, x_res = 0.0, force_max = 0.0, x_range = 0.0;
};

/// Handle-level model of one device (nominal, before identification):
///     m_h = (I_zz,pivot + ratio^2 I_rotor) / rh^2     (rigid sector + reflected rotor)
///     k_g = m g r_c / rh^2                              (the CoM below the pivot acts as a soft spring)
/// b_h = 0.15 N s/m, loop period T = 1 ms and position resolution 20 um are the nominal firmware/device data
/// (project starter devices.HAPKIT). Re-identify m_h and b_h on your own unit (Part B).
HandleModel handle_model();
HandleModel handle_model(const Hapkit& h);

/// Low-cost 6-axis IMU module (illustrative): gyro white noise 0.5 deg/s (1 sigma per sample at 200 Hz),
/// gyro bias of the order of 2 deg/s (measure it at rest every session), accelerometer noise 0.15 m/s^2, 200 Hz.
struct ImuSpec {
    double gyro_noise, gyro_bias, acc_noise, rate;
};
inline const ImuSpec IMU{0.5 * DEG, 2.0 * DEG, 0.15, 200.0};

/// The course-arm hardware safety checklist values (Python: course_models.COURSE_ARM_SAFETY, same field names).
struct CourseArmSafety {
    double soft_limit_margin_deg;              ///< A1: software joint limits >= 10 deg inside the hardware limits
    std::vector<double> hardware_limits_deg;
    double reach_m;
    double keepout_extra_m;                    ///< A2: taped keep-out radius = reach + 0.30 m
    double rated_speed_deg_s;
    double speed_fraction_first_run;           ///< A3: first run of any new motion
    double speed_fraction;                     ///< A3: otherwise
    double payload_rated_kg;
    double payload_max_kg;                     ///< A4: 50 % of the rated payload
    double current_fraction;                   ///< derated servo current limit (set in the servo firmware)
};
inline const CourseArmSafety COURSE_ARM_SAFETY{10.0, {165.0, 165.0, 165.0, 165.0, 165.0, 175.0}, 0.28, 0.30, 160.0,
                                               0.25, 0.50, 0.25, 0.125, 0.60};

/// B3: force saturated at 2 N; watchdog outside +-40 mm.
struct HapticSafety {
    double force_max, x_range;
};
inline const HapticSafety HAPTIC_SAFETY{2.0, 0.040};

}  // namespace lab

#endif  // SBEG46X_LAB08_COURSE_MODELS_HPP
