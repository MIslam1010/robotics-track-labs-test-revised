// SBEG 46X Lab 11 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_scene, trans, rotz, camera_K, target_points, load_calib     data/scene.json, data/calib_points.csv
//   decompose_camera_matrix                                          C = K [R | p] -> K, T_cw
//   interaction_matrix(p, double Z, K)                               same depth for every point (calls yours)
//   servo_metrics                                                    metrics of run_ibvs / run_pbvs
//   endoscope_setup, instrument_tip, simulate_endoscope,             the endoscope holder in closed loop
//   endoscope_metrics                                                (calls YOUR functions)
//
// These are the "Provided" functions of starter.py, translated to C++ line by line.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;

Camera camera_from(const sbeg::json& j) {
    Camera c;
    c.f = j.at("f").get<double>();
    c.rho_w = j.at("rho_w").get<double>();
    c.rho_h = j.at("rho_h").get<double>();
    c.u0 = j.at("u0").get<double>();
    c.v0 = j.at("v0").get<double>();
    c.W = j.at("W").get<int>();
    c.H = j.at("H").get<int>();
    return c;
}

/// Pose given as {"p": [x, y, z], "rz": angle} (optionally "look_at": [x, y, z]) -> trans(p) * [look-at] * rotz(rz).
Eigen::Matrix4d pose_from_spec(const sbeg::json& spec) {
    Eigen::Matrix4d T = trans(sbeg::as<Eigen::Vector3d>(spec.at("p")));
    if (spec.contains("look_at")) {
        Eigen::Vector3d z = sbeg::as<Eigen::Vector3d>(spec.at("look_at")) - T.block<3, 1>(0, 3);
        z /= z.norm();
        Eigen::Vector3d y = Eigen::Vector3d(0.0, 1.0, 0.0) - z(1) * z;
        y /= y.norm();
        T.block<3, 1>(0, 0) = y.cross(z);
        T.block<3, 1>(0, 1) = y;
        T.block<3, 1>(0, 2) = z;
    }
    const double rz = spec.contains("rz") ? spec.at("rz").get<double>() : 0.0;
    return T * rotz(rz);
}
}  // namespace

Scene load_scene(const std::string& path) {
    const sbeg::json j = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/scene.json") : path);
    Scene s;
    s.inspection_camera = camera_from(j.at("inspection_camera"));
    s.endoscope_camera = camera_from(j.at("endoscope_camera"));
    s.target_side = j.at("target").at("side").get<double>();
    s.T_wt = pose_from_spec(j.at("target").at("T_wt"));
    const sbeg::json& ib = j.at("ibvs");
    s.ibvs.lam = ib.at("lam").get<double>();
    s.ibvs.h = ib.at("h").get<double>();
    s.ibvs.n_max = ib.at("n_max").get<int>();
    s.ibvs.tol = ib.at("tol").get<double>();
    s.ibvs.Z_star = ib.at("Z_star").get<double>();
    s.ibvs.T_wc_star = pose_from_spec(ib.at("T_wc_star"));
    const sbeg::json& sp = j.at("start_poses");
    s.start_poses.corke = pose_from_spec(sp.at("corke"));
    s.start_poses.fov = pose_from_spec(sp.at("fov"));
    s.start_poses.roll_angles = sp.at("roll_angles").get<std::vector<double>>();
    const sbeg::json& ca = j.at("calib");
    s.calib.sigma_px = ca.at("sigma_px").get<double>();
    s.calib.seed = ca.at("seed").get<int>();
    s.calib.T_wc = sbeg::as<Eigen::Matrix4d>(ca.at("T_wc"));
    const sbeg::json& e = j.at("endoscope");
    s.endoscope.theta0 = sbeg::as_vec(e.at("theta0"));
    s.endoscope.ell0 = e.at("ell0").get<double>();
    s.endoscope.Z0 = e.at("Z0").get<double>();
    s.endoscope.Z_des = e.at("Z_des").get<double>();
    s.endoscope.a = e.at("a").get<double>();
    s.endoscope.dz = e.at("dz").get<double>();
    s.endoscope.f_hz = e.at("f_hz").get<double>();
    s.endoscope.t_still = e.at("t_still").get<double>();
    s.endoscope.T_end = e.at("T_end").get<double>();
    s.endoscope.h = e.at("h").get<double>();
    s.endoscope.lam = e.at("lam").get<double>();
    s.endoscope.delay = e.at("delay").get<int>();
    s.endoscope.k_z = e.at("k_z").get<double>();
    s.endoscope.k_rcm = e.at("k_rcm").get<double>();
    s.endoscope.ell_min = e.at("ell_min").get<double>();
    return s;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = p;
    return T;
}

Eigen::Matrix4d rotz(double a) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const double c = std::cos(a), s = std::sin(a);
    T(0, 0) = c;
    T(0, 1) = -s;
    T(1, 0) = s;
    T(1, 1) = c;
    return T;
}

Eigen::Matrix3d camera_K(const Camera& cam) { return intrinsic_matrix(cam.f, cam.rho_w, cam.rho_h, cam.u0, cam.v0); }

Target target_points(const Scene& scene) {
    const double s = scene.target_side / 2.0;
    Target t;
    t.xy_t.resize(4, 2);
    t.xy_t << -s, -s, -s, s, s, s, s, -s;
    t.P_w.resize(4, 3);
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector3d q(t.xy_t(i, 0), t.xy_t(i, 1), 0.0);
        t.P_w.row(i) = (scene.T_wt.topLeftCorner<3, 3>() * q + scene.T_wt.block<3, 1>(0, 3)).transpose();
    }
    return t;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> load_calib(const std::string& path) {
    const sbeg::CsvTable c = sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/calib_points.csv") : path);
    Eigen::MatrixXd P(c.data.rows(), 3), p(c.data.rows(), 2);
    P << c.data.col(c.col("X")), c.data.col(c.col("Y")), c.data.col(c.col("Z"));
    p << c.data.col(c.col("u")), c.data.col(c.col("v"));
    return {P, p};
}

std::pair<Eigen::Matrix3d, Eigen::Matrix4d> decompose_camera_matrix(const Matrix34d& C) {
    const Eigen::Vector3d m1 = C.block<1, 3>(0, 0).transpose(), m2 = C.block<1, 3>(1, 0).transpose(),
                          m3 = C.block<1, 3>(2, 0).transpose();
    const double k33 = m3.norm();
    const Eigen::Vector3d r3 = m3 / k33;
    const double k23 = m2.dot(r3);
    const Eigen::Vector3d w2 = m2 - k23 * r3;
    const double k22 = w2.norm();
    const Eigen::Vector3d r2 = w2 / k22;
    const double k13 = m1.dot(r3), k12 = m1.dot(r2);
    const Eigen::Vector3d w1 = m1 - k12 * r2 - k13 * r3;
    const double k11 = w1.norm();
    const Eigen::Vector3d r1 = w1 / k11;
    Eigen::Matrix3d K;
    K << k11, k12, k13, 0.0, k22, k23, 0.0, 0.0, k33;
    Eigen::Matrix3d R;
    R.row(0) = r1.transpose();
    R.row(1) = r2.transpose();
    R.row(2) = r3.transpose();
    const Eigen::Vector3d p = K.partialPivLu().solve(C.col(3));
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.block<3, 1>(0, 3) = p;
    return {K / k33, T};
}

Eigen::MatrixXd interaction_matrix(const Eigen::MatrixXd& p, double Z, const Eigen::Matrix3d& K) {
    return interaction_matrix(p, Eigen::VectorXd::Constant(p.rows(), Z), K);
}

ServoMetrics servo_metrics(const ServoRun& run, const Camera& cam) {
    ServoMetrics m;
    m.n_iter = run.n_iter;
    m.converged = run.converged;
    m.final_err = run.err(run.err.size() - 1);
    for (std::size_t k = 1; k < run.T.size(); ++k)
        m.path_len += (run.T[k].block<3, 1>(0, 3) - run.T[k - 1].block<3, 1>(0, 3)).norm();
    const Eigen::VectorXd mz = run.Z.rowwise().mean();
    m.retreat = mz.maxCoeff() - mz(0);
    double margin = 1e300;
    for (const Eigen::MatrixXd& pk : run.p)
        for (Eigen::Index i = 0; i < pk.rows(); ++i)
            margin = std::min({margin, pk(i, 0), cam.W - pk(i, 0), pk(i, 1), cam.H - pk(i, 1)});
    m.min_margin = margin;
    const Eigen::Index N = run.p.front().rows();
    double s = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        double seg = 0.0;
        for (std::size_t k = 1; k < run.p.size(); ++k) seg += (run.p[k].row(i) - run.p[k - 1].row(i)).norm();
        const double chord = (run.p.back().row(i) - run.p.front().row(i)).norm();
        s += seg / std::max(chord, 1e-12);
    }
    m.straightness = s / static_cast<double>(N);
    return m;
}

EndoSetup endoscope_setup(const Scene& scene) {
    EndoSetup st;
    st.robot = cm::ur5_laparoscope();
    st.theta0 = scene.endoscope.theta0;
    st.T0 = mr::fkin_body(st.robot.M, st.robot.Blist, st.theta0);
    const Eigen::Vector3d z0 = st.T0.block<3, 1>(0, 2), p0 = st.T0.block<3, 1>(0, 3);
    st.port = p0 - scene.endoscope.ell0 * z0;
    st.centre = p0 + scene.endoscope.Z0 * z0;
    st.K = camera_K(scene.endoscope_camera);
    return st;
}

Eigen::Vector3d instrument_tip(const Scene& scene, const EndoSetup& setup, double t) {
    const Scene::Endoscope& e = scene.endoscope;
    const double s = std::max(t - e.t_still, 0.0);
    const double w = 2.0 * PI * e.f_hz;
    return setup.centre + e.a * (std::cos(w * s) * setup.T0.block<3, 1>(0, 0) +
                                 std::sin(w * s) * setup.T0.block<3, 1>(0, 1)) +
           e.dz * std::sin(w * s) * setup.T0.block<3, 1>(0, 2);
}

EndoSim simulate_endoscope(const Scene& scene, const EndoOptions& opt) {
    const Scene::Endoscope& e = scene.endoscope;
    const double lam = opt.lam.value_or(e.lam);
    const int delay = opt.delay.value_or(e.delay);
    const double k_z = opt.k_z.value_or(e.k_z);
    const double k_rcm = opt.k_rcm.value_or(e.k_rcm);
    const double T_end = opt.T_end.value_or(e.T_end);
    const double h = e.h;
    const EndoSetup st = endoscope_setup(scene);
    const Camera& cam = scene.endoscope_camera;
    Eigen::MatrixXd p_star(1, 2);
    p_star << cam.u0, cam.v0;
    const int N = static_cast<int>(std::lround(T_end / h));
    Eigen::VectorXd th = st.theta0;
    EndoSim out;
    out.t.resize(N + 1);
    out.Z.resize(N + 1);
    out.err.resize(N + 1);
    out.ell.resize(N + 1);
    out.e_lat.resize(N + 1);
    out.p.resize(N + 1, 2);
    out.theta.resize(N + 1, 6);
    out.theta_dot.resize(N, 6);
    out.xi.resize(N, 4);
    out.V.resize(N, 6);
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> hist;
    for (int k = 0; k <= N; ++k) {
        const double t = k * h;
        const Eigen::Matrix4d T_sc = mr::fkin_body(st.robot.M, st.robot.Blist, th);
        const Eigen::MatrixXd tip = instrument_tip(scene, st, t).transpose();
        const std::pair<Eigen::MatrixXd, Eigen::VectorXd> pz = project_points(st.K, T_sc, tip);
        if (pz.first.rows() != 1 || pz.first.cols() != 2 || pz.second.size() != 1)
            throw std::invalid_argument("project_points returned the wrong size");
        const std::pair<double, Eigen::Vector2d> re = rcm_errors(T_sc, st.port);
        hist.push_back(pz);
        out.t(k) = t;
        out.p.row(k) = pz.first.row(0);
        out.Z(k) = pz.second(0);
        out.err(k) = (pz.first.row(0) - p_star.row(0)).norm();
        out.ell(k) = re.first;
        out.e_lat(k) = re.second.norm();
        out.theta.row(k) = th.transpose();
        out.in_view.push_back(std::abs(pz.first(0, 0) - cam.u0) <= cam.W / 2.0 &&
                              std::abs(pz.first(0, 1) - cam.v0) <= cam.H / 2.0);
        if (k == N) break;
        const std::pair<Eigen::MatrixXd, Eigen::VectorXd>& pm = hist[static_cast<std::size_t>(std::max(0, k - delay))];
        Eigen::VectorXd Z_used(1);
        double d_dot = 0.0;
        if (!opt.Z_hat) {
            Z_used = pm.second;
            d_dot = k_z * (pm.second(0) - e.Z_des);
        } else {
            Z_used(0) = *opt.Z_hat;
        }
        const std::pair<Eigen::Vector4d, Vector6d> xv = rcm_ibvs_velocity(pm.first, p_star, Z_used, st.K, lam,
                                                                          re.first, d_dot);
        Vector6d V = xv.second;
        V(3) += k_rcm * re.second(0);
        V(4) += k_rcm * re.second(1);
        const Eigen::VectorXd th_dot = joint_rates(st.robot.Blist, th, V, st.robot.qdot_max);
        const Eigen::VectorXd th_free = joint_rates(st.robot.Blist, th, V);
        if (th_dot.size() != 6 || th_free.size() != 6) throw std::invalid_argument("joint_rates returned the wrong size");
        out.sat.push_back(th_free.cwiseAbs().maxCoeff() > st.robot.qdot_max * (1 + 1e-12));
        out.theta_dot.row(k) = th_dot.transpose();
        out.xi.row(k) = xv.first.transpose();
        out.V.row(k) = V.transpose();
        th = th + h * th_dot;
    }
    return out;
}

EndoMetrics endoscope_metrics(const EndoSim& sim, double t_from) {
    EndoMetrics m;
    double s2 = 0.0;
    int n = 0;
    Eigen::Index k3 = 0;
    for (Eigen::Index k = 0; k < sim.t.size(); ++k) {
        if (sim.t(k) >= t_from - 1e-9) {
            s2 += sim.err(k) * sim.err(k);
            m.max_err = std::max(m.max_err, sim.err(k));
            ++n;
        }
        if (std::abs(sim.t(k) - 3.0) < std::abs(sim.t(k3) - 3.0)) k3 = k;
    }
    m.rms_err = std::sqrt(s2 / n);
    m.final_still_err = sim.err(k3);
    m.max_e_lat = sim.e_lat.maxCoeff();
    m.ell_min = sim.ell.minCoeff();
    m.ell_max = sim.ell.maxCoeff();
    m.peak_qdot = sim.theta_dot.cwiseAbs().maxCoeff();
    m.sat_frac = static_cast<double>(std::count(sim.sat.begin(), sim.sat.end(), true)) / static_cast<double>(sim.sat.size());
    m.view_frac = static_cast<double>(std::count(sim.in_view.begin(), sim.in_view.end(), true)) /
                  static_cast<double>(sim.in_view.size());
    return m;
}

}  // namespace lab
