// SBEG 46X Hardware rotation 1 (Week 8) -- C++ track: course hardware models (GIVEN, do not edit).
// C++ copy of course_models.py (numbers identical).
#include "course_models.hpp"

#include <cmath>

namespace lab {

Hapkit hapkit() {
    Hapkit h;
    h.key = "hapkit";
    h.illustrative = true;
    h.rh = 0.090;
    h.rs = 0.073;
    h.rp = 0.0042;
    h.ratio = h.rs / h.rp;
    h.mass = 0.10;
    h.com_below_pivot = 0.010;
    h.I_com = Eigen::Vector3d(2.0e-4, 1.0e-4, 1.5e-4).asDiagonal();
    h.rotor_inertia = 1.0e-6;
    h.force_max = 2.0;
    h.x_range = 0.040;
    h.loop_rate_hz = 1000.0;
    return h;
}

HandleModel handle_model() { return handle_model(hapkit()); }

HandleModel handle_model(const Hapkit& h) {
    const double Izz_pivot = h.I_com(2, 2) + h.mass * h.com_below_pivot * h.com_below_pivot;
    HandleModel m;
    m.m_h = (Izz_pivot + h.ratio * h.ratio * h.rotor_inertia) / (h.rh * h.rh);
    m.k_g = h.mass * G0 * h.com_below_pivot / (h.rh * h.rh);
    m.b_h = 0.15;
    m.T = 1.0e-3;
    m.x_res = 2.0e-5;
    m.force_max = h.force_max;
    m.x_range = h.x_range;
    return m;
}

}  // namespace lab
