// SBEG 46X Lab 4 -- Haptic rendering on a Hapkit-type device: virtual wall, passivity and the stable stiffness
// limit (Z-width), virtual fixtures with a proxy, tool-tissue contact and a needle-insertion force profile
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp (the functions marked TODO A1 ... D6). Do not change these declarations:
// test_lab.cpp calls them. The provided functions (data loaders, rk4_step, the device simulator, the hand model,
// the hapkit model, the needle helpers) are in helpers.cpp (given code, do not edit).
// These functions become the `haptics` module of your cumulative library sbeg46x_medrob.
//
// Notation and units: SI (s, m, kg, N).  Impedance m x'' + b x' + k x = f (MR Eq. 11.62); impedance control
// with M = 0 on a lightweight backdrivable device (MR Sec. 11.7.1, p. 446); current control of a geared motor
// (MR Sec. 11.8, pp. 447-449).
//
// Sign conventions used everywhere
//   * x is the handle position (m), positive away from the user; F_motor is the force the device applies TO the
//     hand, f_hand the force the hand applies TO the device.  Device:  m_h x'' + b_h x' + k_g x + f_c sgn(x') =
//     F_motor + f_hand.
//   * Needle and tissue: z = x - x_skin is the needle-tip depth below the undeformed skin surface (m), v = z'.
//     The tissue force f_tissue(z, v) >= 0 resists insertion; the device renders F_motor = -f_tissue during
//     insertion (and +friction during withdrawal, see NeedleTissue).
//
// Parts
//   A. Passivity limits of a sampled virtual wall: Colgate's condition b > K T / 2 + |B|, its frequency-by-
//      frequency form, the quantisation condition f_c > K dx / 2, and the energy that a zero-order hold
//      creates in one push into a wall
//   B. Measuring the stable stiffness limit: bounce (restitution) test in simulation, threshold from logged trials
//   C. Virtual fixtures: unilateral forbidden-region wall, a proxy (god-object) for a thin keep-out slab,
//      a 2-D forbidden disc around a vessel, an admittance-type guidance fixture
//   D. Tool-tissue and needle-insertion force models: Kelvin-Voigt, Hunt-Crossley, a layered
//      stiffness-friction-cutting needle model, puncture detection in a force log
//
// Python -> C++: the dicts of device.json / needle_tissue.json -> lab::Device / lab::TissueParams (same field
// names); the dict of load_csv -> lab::Columns (std::map from the header name to an Eigen::VectorXd); the result
// dicts of simulate_device / render_needle -> lab::DeviceSim / lab::NeedleSim; call-backs (law, f_hand, stop) ->
// std::function (an empty one = Python None); **kw -> the explicit arguments quantize, friction; the classes
// ProxySlab and NeedleTissue -> C++ classes with the same public fields; a tuple (kind, z) -> lab::Event
// (std::pair: .first = kind, .second = z); (F, q) -> std::pair; a list of indices -> std::vector<int>;
// ValueError -> std::invalid_argument.
#ifndef SBEG46X_LAB4_HPP
#define SBEG46X_LAB4_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types only)

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Device and loop parameters of data/device.json (Python: the dict returned by load_device()).
struct Device {
    double T = 0.0;          ///< sample time [s]
    double x_res = 0.0;      ///< position sensor resolution [m]
    double force_max = 0.0;  ///< force limit [N] (commands are saturated to +-force_max)
    double x_range = 0.0;    ///< handle range +- [m]
    double k_t = 0.0;        ///< motor torque constant [N m/A]
    double m_h = 0.0;        ///< handle mass [kg]
    double b_h = 0.0;        ///< handle damping [N s/m]
    double k_g = 0.0;        ///< gravity stiffness [N/m]
    double f_c = 0.0;        ///< handle-level Coulomb friction [N]
    std::string note;
};

/// Layered needle-decompression trainer model of data/needle_tissue.json (Python: load_tissue()).
/// Units N, m, s.  See NeedleTissue for the meaning of every field.
struct TissueParams {
    struct Skin {
        double a1 = 0.0, a2 = 0.0, f_p = 0.0;  ///< f = a1 z + a2 z^2 until f_p
    } skin;
    struct Tissue {
        double f_cut = 0.0, mu = 0.0;          ///< cutting force [N], shaft friction per length [N/m]
    } tissue;
    struct Pleura {
        double z_pl = 0.0, k_pl = 0.0, f_p = 0.0;  ///< depth [m], membrane stiffness [N/m], puncture force [N]
    } pleura;
    double v_eps = 0.0;   ///< friction smoothing speed [m/s]
    double x_skin = 0.0;  ///< handle position of the undeformed skin [m]
    std::string note;
};

/// A CSV file of data/ as columns keyed by the header names (Python: the dict returned by load_csv).
/// `cols.at("K")` is an Eigen::VectorXd; `cols.size()` the number of columns.
using Columns = std::map<std::string, Eigen::VectorXd>;

/// law(t, x_meas, v_est) -> commanded force, and f_hand(t, x, v) -> hand force (simulate_device).
using ForceLaw = std::function<double(double t, double x, double v)>;
/// stop(k, x, v) -> true to end the run after sample k (simulate_device).
using StopCondition = std::function<bool(int k, double x, double v)>;
/// s' = f(s) for rk4_step.
using StateFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd& s)>;

/// Result of simulate_device (Python: dict t, x, v, x_meas, v_est, F, saturated).
/// Every vector has one entry per simulated sample k = 0 .. last.
struct DeviceSim {
    Eigen::VectorXd t, x, v;      ///< sample times and TRUE state at the samples
    Eigen::VectorXd x_meas, v_est;  ///< measured (quantised) position and backward-difference velocity
    Eigen::VectorXd F;            ///< applied (saturated) force
    std::vector<bool> saturated;  ///< true where the commanded force exceeded force_max
};

/// A puncture event (kind, z): kind is "skin" or "pleura", z the depth (m) at which it happened.
using Event = std::pair<std::string, double>;

/// Result of render_needle (Python: the simulate_device dict plus z, f_tissue, events).
struct NeedleSim : DeviceSim {
    Eigen::VectorXd z;          ///< true depth x - x_skin
    Eigen::VectorXd f_tissue;   ///< = -F
    std::vector<Event> events;  ///< punctures, in order
};

/// Transmission and limits of the Hapkit-type device (Python: course_models.hapkit()["params"]).
struct HapkitParams {
    double rh = 0.0, rs = 0.0, rp = 0.0, ratio = 0.0;
    double rotor_inertia = 0.0, reflected_rotor_inertia = 0.0;
    double force_max = 0.0, x_range = 0.0, loop_rate_hz = 0.0;
};

/// The 1-DOF haptic paddle in MR list form (Python: the dict returned by course_models.hapkit()).
struct HapkitModel {
    Eigen::MatrixXd Slist;    ///< 6 x 1 screw axis (one per COLUMN)
    Eigen::Matrix4d M;        ///< home configuration of the handle frame {b}
    mr::TransformList Mlist;  ///< 2 frames M_{i-1,i}
    mr::InertiaList Glist;    ///< 1 spatial inertia (MR Eq. 8.32)
    Eigen::Vector3d g;        ///< gravity in {s} [m/s^2]
    Eigen::MatrixXd limits;   ///< 1 x 2 joint limits [rad]
    Eigen::VectorXd qdot_max, tau_max;
    HapkitParams params;
};

// ---------------------------------------------------------------------------
// Provided (given in helpers.cpp; do not edit): data files, the device simulator, the hand, the hapkit model
// ---------------------------------------------------------------------------

/// Device and loop parameters from data/device.json (illustrative values): T (s), x_res (m), force_max (N),
/// x_range (m), m_h (kg), b_h (N s/m), k_g (N/m), f_c (N), k_t (N m/A).
Device load_device();

/// Layered needle-decompression trainer model from data/needle_tissue.json (illustrative values).
TissueParams load_tissue();

/// A CSV file of data/ (e.g. "zwidth_trials.csv") as columns keyed by the header names.
Columns load_csv(const std::string& name);

/// One classical Runge-Kutta step of s' = f(s) with step h.
Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& s, double h);

/// Sampled-data simulation of the haptic device rendering `law` (same loop as SBEG 461 Lab 8).
///
/// Plant (continuous):  m_h x'' + b_h x' + k_g x + f_c sgn(x') = F_applied + f_hand(t, x, v), with the Coulomb
/// term only if friction = true (smoothed as f_c tanh(v / 1 mm/s)); RK4 with `substeps` steps per sample.
/// Controller every T = dev.T at t_k = k T:
///   1. x_meas = x rounded to the sensor resolution dev.x_res (if quantize), else x;
///   2. v_est = (x_meas_k - x_meas_{k-1}) / T  (backward difference; v_est = v0 at k = 0);
///   3. F_cmd = law(t_k, x_meas, v_est), saturated to +-dev.force_max, held constant until t_{k+1}
///      (zero-order hold).
/// `stop(k, x, v)` may return true to end the run early (after sample k).  An empty f_hand = no hand.
DeviceSim simulate_device(const Device& dev, const ForceLaw& law, double T_end, double x0 = 0.0, double v0 = 0.0,
                          const ForceLaw& f_hand = ForceLaw(), bool quantize = true, bool friction = false,
                          int substeps = 10, const StopCondition& stop = StopCondition());

/// A hand that drives the handle along x_ref(t) = x_start + speed (t - t_start) through a grasp
/// spring-damper (k_hand, b_hand): f_hand = k_hand (x_ref - x) + b_hand (speed - v).  A passive model of a
/// trainee pushing a needle in at a steady speed.
ForceLaw hand_ramp(double x_start, double speed, double k_hand = 400.0, double b_hand = 4.0, double t_start = 0.0);

/// The shared course hapkit model (= course_models.hapkit(), numbers identical).
HapkitModel hapkit();

// ---------------------------------------------------------------------------
// Part A -- passivity of a sampled virtual wall
// ---------------------------------------------------------------------------

/// A1. Largest wall stiffness K (N/m) for which the sampled wall F = -K x - B v_est (v_est by backward
/// difference, zero-order hold, period T) on a device with physical damping b is passive, from Colgate's
/// condition  b > K T / 2 + |B|  (Colgate & Brown 1994):  K_max = 2 (b - |B|) / T, or 0 if |B| >= b.
double colgate_K_max(double b, double B, double T);

/// A2. Physical damping needed for passivity at each normalised frequency theta = omega T in (0, pi]:
///
///     b_req(theta) = (T/2) Re{ (1 - e^{-j theta}) H(e^{j theta}) } / (1 - cos theta),
///     H(z) = K + B (1 - z^-1) / T   (spring + backward-difference damper),
///
/// the frequency-by-frequency form of Colgate's passivity condition.  Returns a vector like theta.
/// (Its maximum over theta is K T/2 + |B|: at theta -> 0 if B < 0, at the Nyquist frequency theta = pi if B > 0.)
Eigen::VectorXd required_damping(double K, double B, double T, const Eigen::VectorXd& theta);

/// The same for one frequency (given in helpers.cpp: it calls your vector version).
double required_damping(double K, double B, double T, double theta);

/// A3. Largest K (N/m) for which the device's Coulomb friction f_c (N) can absorb the energy created by a
/// position sensor of resolution x_res (m): f_c > K x_res / 2  ->  K_max = 2 f_c / x_res.
double quantisation_K_max(double f_c, double x_res);

/// A4. Net work (J) done BY a sampled unilateral wall (surface at x = 0, wall at x > 0) ON a hand that pushes
/// into it at constant `speed` to `depth` and comes straight back out at the same speed.
///
/// The wall samples x at t_k = k T (starting at the moment of contact, t = 0) and holds F_k = -K max(x(t_k), 0)
/// until t_{k+1}, so the work on interval k is exactly F_k (x(t_{k+1}) - x(t_k)).  Positive result = energy
/// CREATED by the wall (a continuous spring would give exactly 0).  Assumes depth / (speed T) is close to an
/// integer N; then the result is K speed T depth.
double sampled_wall_work(double K, double T, double speed, double depth);

// ---------------------------------------------------------------------------
// Part B -- measuring the stable stiffness limit
// ---------------------------------------------------------------------------

/// B1. Unilateral virtual wall occupying side*(x - x_wall) > 0 (a forbidden region).  Outside: F = 0.  Inside:
/// F = -K (x - x_wall) - B v, clipped so that the wall only PUSHES the hand out (side*F <= 0).
double wall_force(double x, double v, double K, double B, double x_wall = 0.0, int side = +1);

/// B2. Coefficient of restitution e = |v_out| / |v_before| of the free handle bouncing once on a virtual wall
/// at x = 0 (no hand).
///
/// Start half a sample before the wall, x0 = -v_in T / 2, with speed v_in toward it; render
/// wall_force(x, v, K, B) with simulate_device and stop at the first sample after contact at which the TRUE
/// position is back outside (x < 0).  v_before is the true velocity at the last sample before contact (here the
/// first sample), v_out the true velocity at the stop sample.  e > 1 means the wall CREATED energy during the
/// contact (an active wall; on the real device a growing bounce or a buzz).  Max. run 1 s; return 0 if the
/// handle never leaves the wall.
double bounce_restitution(const Device& dev, double K, double B = 0.0, double v_in = 0.05, bool quantize = false,
                          bool friction = false);

/// B3. Stiffness (N/m) at which the bounce restitution crosses 1, by bisection on [K_lo, K_hi] (to within tol),
/// assuming e(K_lo) < 1 < e(K_hi).  This is the simulated version of the bench test 'raise K until the wall
/// starts to buzz'.  quantize, friction are passed on to bounce_restitution (Python: **kw).
double measure_K_limit(const Device& dev, double B = 0.0, double v_in = 0.05, double K_lo = 20.0,
                       double K_hi = 3000.0, double tol = 1.0, bool quantize = false, bool friction = false);

/// B4. Stiffness at which a logged restitution e(K) first crosses 1 (K sorted ascending): straight-line
/// interpolation between the last trial with e < 1 and the first with e >= 1.  Throws std::invalid_argument if
/// e never reaches 1 or already starts at or above 1.
double threshold_from_trials(const Eigen::VectorXd& K, const Eigen::VectorXd& e);

// ---------------------------------------------------------------------------
// Part C -- virtual fixtures
// ---------------------------------------------------------------------------

/// C1. Penalty rendering of a thin forbidden slab a < x < a + w: push toward the NEAREST surface,
/// F = K (a - x) if x < a + w/2 else K (a + w - x); F = 0 outside.  (Shows pop-through past the mid-plane.)
double penalty_slab_force(double x, double a, double w, double K);

/// Proxy (god-object) rendering of the thin forbidden slab a < x < a + w on a 1-DOF device.
///
/// The proxy p is a point that is never allowed inside the slab and follows the handle x when it can:
///   * if x is outside the slab and the straight move from p to x does not cross the slab, p = x;
///   * otherwise p stays on the surface it was last outside of (entry side remembered), i.e. p = a if the
///     proxy was at or below a, p = a + w if it was at or above a + w.
/// Force F = K (p - x) (a spring from the handle to the proxy; zero when p = x).
class ProxySlab {
public:
    /// Given: stores a, w, K and p = x0; throws std::invalid_argument("start the proxy outside the slab") if
    /// a < x0 < a + w.
    ProxySlab(double a, double w, double K, double x0);

    /// C2. Move the proxy for the new handle position x by the two rules above; store it in p and return it.
    double update(double x);

    /// Given: K (update(x) - x).
    double force(double x);

    double a, w, K, p;
};

/// C3. Forbidden-region fixture around a vessel: the disc |p - c| < R (vessel radius + safety margin) in the
/// plane.  Outside: F = 0 and the proxy is p.  Inside: the proxy is the closest boundary point
/// q = c + R (p - c) / |p - c| and F = K (q - p).  Returns (F, q).  (At p = c exactly the direction is
/// undefined: use n = (1, 0).)
std::pair<Eigen::VectorXd, Eigen::VectorXd> forbidden_disc_force(const Eigen::VectorXd& p, const Eigen::VectorXd& c,
                                                                 double R, double K);

/// C4. Admittance-type guidance virtual fixture (steady-hand cooperative robot): the commanded tool velocity is
/// v = c (f_par + c_tau f_perp), with f_par = (f . d) d the component of the user's force along the preferred
/// unit direction d and f_perp = f - f_par.  c_tau = 1: no fixture; c_tau = 0: hard fixture (motion only along
/// d); 0 < c_tau < 1: soft fixture.  d need not be unit length (normalise it).  Returns v.
Eigen::VectorXd guidance_velocity(const Eigen::VectorXd& f, const Eigen::VectorXd& d, double c, double c_tau);

// ---------------------------------------------------------------------------
// Part D -- tool-tissue and needle-insertion force models
// ---------------------------------------------------------------------------

/// D1. Kelvin-Voigt contact (spring k in parallel with damper b) for penetration delta (m) and rate ddelta:
/// f = k delta + b ddelta while delta > 0, else 0.  The result is NOT clipped: it jumps to b ddelta at first
/// touch and can be negative (a non-physical pull) when withdrawing fast.
double kelvin_voigt_force(double delta, double ddelta, double k, double b);

/// D2. Hunt-Crossley contact f = k delta^n + lam delta^n ddelta for delta > 0, else 0, clipped at >= 0
/// (contact can push, never pull).  Continuous at first touch (f = 0 at delta = 0).
double hunt_crossley_force(double delta, double ddelta, double k, double lam, double n);

/// D3. Membrane deflection delta_p (m) at which the stiffness force a1 delta + a2 delta^2 reaches the puncture
/// force f_p (positive root of the quadratic; a2 = 0 allowed).
double puncture_depth(double a1, double a2, double f_p);

/// Layered stiffness-friction-cutting needle model (after Okamura, Simone & O'Leary 2004), for the
/// needle-decompression trainer (illustrative values in data/needle_tissue.json).
///
/// Depth z (m) below the undeformed skin, speed v = dz/dt.  Parameters (TissueParams p):
///   skin:   a1 (N/m), a2 (N/m^2), f_p (N)  -- membrane stiffness f = a1 z + a2 z^2 until it reaches f_p
///   tissue: f_cut (N), mu (N/m)           -- after skin puncture, while inserting (v > 0) a constant cutting
///                                           force f_cut at the tip, and shaft friction mu * L with
///                                           L = min(z, z_pl) the shaft length inside the tissue
///   pleura: z_pl (m), k_pl (N/m), f_p (N)  -- at depth z_pl the tip meets the pleura: an extra membrane force
///                                           k_pl (z - z_pl) until it reaches f_p (then the pleura is punctured:
///                                           no cutting force any more, friction only)
///   v_eps (m/s): friction is mu L sgn(v) for |v| >= v_eps and mu L v / v_eps below (Karnopp-type smoothing)
///
/// Force on the needle (resisting insertion is positive):
///   before skin puncture:  f = a1 z + a2 z^2 for z > 0 (elastic: the same curve on the way out), 0 for z <= 0;
///   after skin puncture:   f = fric + [v > 0] * f_cut              (tip in tissue, z < z_pl, pleura intact)
///                          f = fric + [v > 0] * f_cut + k_pl (z - z_pl)   (z >= z_pl, pleura intact; the
///                                                                   membrane term is elastic, any v sign)
///                          f = fric                                   (pleura punctured)
///   with fric = mu L sgn_eps(v) (negative when withdrawing: the friction then pulls the needle back in).
/// Punctures are permanent (state).  `events` records (kind, z) at each puncture.
class NeedleTissue {
public:
    /// Given: stores p; no punctures, no events.
    explicit NeedleTissue(const TissueParams& p);

    /// D4. The force on the needle at depth z and speed v, as described above (updates the puncture state
    /// and events).
    double force(double z, double v);

    TissueParams p;
    bool skin_punctured = false;
    bool pleura_punctured = false;
    std::vector<Event> events;
};

/// Given: insertion force (v > 0, slow) against depth for the NeedleTissue model, evaluated on the increasing
/// depth grid z with a fresh model (punctures happen in order along the grid).
Eigen::VectorXd needle_quasistatic_profile(const TissueParams& p, const Eigen::VectorXd& z);

/// D5. Puncture events in an insertion log (z depth, F resisting force, one entry per sample).
///
/// A puncture starts at sample i if the force falls by more than `drop` (N) within `window` samples AND stays
/// down: F[i] - max(F[i + window : i + window + hold]) > drop, while the depth is increasing
/// (z[i + window] > z[i]).  The event index is the sample of largest force in F[i : i + window + 1] (the peak
/// just before the drop; the first one if several are equal); after an event, skip to that index + window +
/// hold.  Short dips (the friction changing sign for a few samples when the needle briefly moves back) are
/// therefore not punctures.  Examine i = 0, 1, ... while i < len(F) - window - hold.  Returns the indices.
std::vector<int> detect_punctures(const Eigen::VectorXd& z, const Eigen::VectorXd& F, double drop, int window = 5,
                                  int hold = 20);

/// D6. Largest positive slope df/dz (N/m) of a force-depth profile between consecutive samples (the stiffness the
/// device must render there; compare with colgate_K_max).
double max_profile_slope(const Eigen::VectorXd& z, const Eigen::VectorXd& f);

/// Given: render the needle model on the device while a hand (hand_ramp from x0 at `speed`) pushes the handle in.
///
/// Law at each sample: z = x_meas - x_skin; velocity from the last n_vel + 1 measured positions,
/// v_f = (x_meas_k - x_meas_{k-n_vel}) / (n_vel T) (a backward difference over n_vel samples: the one-sample
/// difference of a 20 um sensor at 1 kHz only takes the values 0, 20, 40 mm/s ... and would switch the
/// friction and cutting forces on and off); F_motor = -NeedleTissue::force(z, v_f).
/// quantize, friction are passed on to simulate_device (Python: **kw).
NeedleSim render_needle(const Device& dev, const TissueParams& p, double x_skin, double speed, double T_end,
                        double x0, int n_vel = 20, bool quantize = true, bool friction = false);

}  // namespace lab

#endif  // SBEG46X_LAB4_HPP
