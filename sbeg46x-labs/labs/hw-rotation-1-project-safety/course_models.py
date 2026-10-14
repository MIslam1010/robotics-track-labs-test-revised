"""Local copy of the course hardware models used in hardware rotation 1 (Week 8).

Numbers identical to Course resources: "Course robot models and hardware safety checklist" (Hapkit-type device,
course-arm checklist) and to the project starter package (`devices.HAPKIT`, `devices.IMU`), so that this lab is
self-contained. All values are ILLUSTRATIVE teaching values, not the data of a product.

* hapkit()            -- Hapkit-type 1-DOF haptic paddle (geometry, mass properties, derated limits)
* handle_model()      -- the handle-level model m_h x'' + b_h x' + k_g x = F used on the station (nominal values)
* IMU                 -- low-cost 6-axis IMU module (bias and noise class, sample rate)
* COURSE_ARM_SAFETY   -- the course-arm hardware safety checklist values (rules A1-A7 and the current limit)
* HAPTIC_SAFETY       -- the haptic-device limits (rule B3)
"""
import numpy as np

G0 = 9.81
DEG = np.pi / 180.0


# copied from the course robot models (Course resources), function hapkit; numbers identical
def hapkit() -> dict:
    """Hapkit-type 1-DOF haptic paddle.  Handle pivots about z_s; the grip is rh = 0.090 m from the pivot;
    sector radius rs = 0.073 m, motor pulley rp = 0.0042 m (ratio rs/rp ~ 17.4).  Handle displacement x ~ rh*theta.
    Moving mass 0.10 kg with its centre of mass 10 mm below the pivot; rotational inertia about the CoM
    diag(2.0e-4, 1.0e-4, 1.5e-4) kg m^2; motor rotor inertia 1.0e-6 kg m^2.
    Derated limits: handle force <= 2 N (saturated in firmware), watchdog cuts the motor outside +-40 mm."""
    rh, rs, rp = 0.090, 0.073, 0.0042
    return dict(key="hapkit", illustrative=True, rh=rh, rs=rs, rp=rp, ratio=rs / rp,
                mass=0.10, com_below_pivot=0.010, I_com=np.diag([2.0e-4, 1.0e-4, 1.5e-4]),
                rotor_inertia=1.0e-6, force_max=2.0, x_range=0.040, loop_rate_hz=1000.0)


def handle_model(h=None) -> dict:
    """Handle-level model of one device (nominal, before identification):
        m_h = (I_zz,pivot + ratio^2 I_rotor) / rh^2     (rigid sector + reflected rotor)
        k_g = m g r_c / rh^2                              (the CoM below the pivot acts as a soft spring)
    b_h = 0.15 N s/m, loop period T = 1 ms and position resolution 20 um are the nominal firmware/device data
    (project starter `devices.HAPKIT`). Re-identify m_h and b_h on your own unit (Part B)."""
    h = h or hapkit()
    Izz_pivot = h["I_com"][2, 2] + h["mass"] * h["com_below_pivot"] ** 2
    m_h = (Izz_pivot + h["ratio"] ** 2 * h["rotor_inertia"]) / h["rh"] ** 2
    k_g = h["mass"] * G0 * h["com_below_pivot"] / h["rh"] ** 2
    return dict(m_h=m_h, b_h=0.15, k_g=k_g, T=1.0e-3, x_res=2.0e-5,
                force_max=h["force_max"], x_range=h["x_range"])


# copied from the project starter package (devices.IMU); numbers identical
IMU = dict(gyro_noise=np.radians(0.5), gyro_bias=np.radians(2.0), acc_noise=0.15, rate=200.0)
"""Low-cost 6-axis IMU module (illustrative): gyro white noise 0.5 deg/s (1 sigma per sample at 200 Hz),
gyro bias of the order of 2 deg/s (measure it at rest every session), accelerometer noise 0.15 m/s^2, 200 Hz.
Low-voltage USB or battery device, no actuator."""

# copied from the course robot models (Course resources), COURSE_ARM_SAFETY and the course-arm ratings
COURSE_ARM_SAFETY = dict(
    soft_limit_margin_deg=10.0,      # A1: software joint limits >= 10 deg inside the hardware limits
    hardware_limits_deg=[165.0] * 5 + [175.0],
    reach_m=0.28,
    keepout_extra_m=0.30,            # A2: taped keep-out radius = reach + 0.30 m
    rated_speed_deg_s=160.0,
    speed_fraction_first_run=0.25,   # A3: first run of any new motion
    speed_fraction=0.50,             # A3: otherwise
    payload_rated_kg=0.25,
    payload_max_kg=0.125,            # A4: 50 % of the rated payload
    current_fraction=0.60,           # derated servo current limit (set in the servo firmware)
)

HAPTIC_SAFETY = dict(force_max=2.0, x_range=0.040)   # B3: force saturated at 2 N; watchdog outside +-40 mm
