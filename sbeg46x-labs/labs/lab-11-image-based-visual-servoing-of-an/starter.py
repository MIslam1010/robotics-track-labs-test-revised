"""SBEG 46X Lab 11 -- image-based visual servoing of an endoscope-holding arm with a simulated pinhole camera
(formal report 3).
STARTER CODE -- you edit this file.  Its functions join the module `vision` of your library sbeg46x_medrob.
Write every function marked TODO (keep the names and signatures); the helpers in the "Provided" section are given
(do not change them).  Run the checks from your Lab code folder with:   LAB_IMPL=starter python -m pytest -q test_lab.py
(rename the file to lab11_<ID>.py before you submit and run LAB_IMPL=lab11_<ID>).

Models (all values illustrative, data/scene.json):
    Pinhole camera (Corke Sec. 11.1, Eqs. 11.2 and 11.7-11.10, pp. 252-257): camera frame {c}, optical axis z_c,
    x_c to the right and y_c down in the image; intrinsic matrix K = [[f/rho_w, 0, u0], [0, f/rho_h, v0], [0, 0, 1]];
    a point P_c = (X, Y, Z) in {c} (Z > 0) has pixel coordinates (u, v) = (u0 + f/rho_w X/Z, v0 + f/rho_h Y/Z).
    Camera poses are T_wc (camera in the world, as T_sb in MR); P_c = T_wc^-1 P_w.
    Camera matrix (Corke Eq. 11.9): C = K [R_cw | p_cw] (3 x 4); DLT calibration (Corke Sec. 11.2.1, Eqs. 11.15-11.16,
    pp. 262-263), solved here by SVD and scaled so that ||C[2, :3]|| = 1 (then C = K [R | p] exactly).
    Camera twist: the BODY twist of {c}, V_c = (omega, v) in MR order, angular part first (MR Eq. 3.70, p. 96).
    Corke writes the camera velocity (v, omega), linear first: the interaction matrix here is Corke's Eq. 15.6
    (p. 461) with its two 2 x 3 blocks swapped.  Normalised coordinates x = (u - u0)/f_u, y = (v - v0)/f_v:
        d/dt [u; v] = L(u, v, Z) V_c,
        L = [ f_u (x y)       -f_u (1 + x^2)   f_u y    -f_u/Z      0       f_u x/Z ]
            [ f_v (1 + y^2)   -f_v x y          -f_v x    0       -f_v/Z    f_v y/Z ]
    IBVS (Corke Sec. 15.2.2, Eqs. 15.8-15.11, pp. 464-466): e = p - p* (stacked u1, v1, u2, v2, ...),
        V_c = -lambda L^+ e,  pose update T_wc <- T_wc exp([V_c] h)  (MR Eq. 3.88, p. 103).
    PBVS (Corke Sec. 15.1, pp. 456-459): estimate T_ct from the image (planar target: homography, Corke Sec. 14.2.4,
    Eq. 14.12, p. 396, then H ~ K [r1 r2 t]); V_c = lambda log(T_ct T_c*t^-1)  (MR Sec. 3.3.3.2, p. 104).
    Endoscope through a port (remote centre of motion, Week 1): the scope axis must pass through the port point,
    which lies on the optical axis at distance ell BEHIND the camera, r = (0, 0, -ell) in {c}.  Allowed body twists
        V_c = W(ell) xi,   xi = (omega_x, omega_y, omega_z, d_dot),   v = (ell omega_y, -ell omega_x, d_dot)
    (the point of the scope that is in the port then moves with v + omega x r = (0, 0, d_dot): along the scope axis
    only -- the scope pivots about the port and slides through it, and the port stays on the optical axis).  Joint rates by MR Eq. 6.7 (p. 231) with the body
    Jacobian of the course model (MR Eq. 5.18, p. 183; camera {c} = end-effector {b} at the scope tip).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import modern_robotics as mr
import numpy as np

import course_models as cm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): scene, poses, data, decomposition, endoscope simulator, metrics
# ==========================================================================================
def load_scene(path=DATA / "scene.json") -> dict:
    """Lab parameters (data/scene.json, illustrative values): 'inspection_camera', 'endoscope_camera', 'target',
    'ibvs', 'start_poses', 'calib', 'endoscope'."""
    with open(path) as f:
        return json.load(f)


def trans(p) -> np.ndarray:
    """Pure translation T = (I, p)."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(p, float)
    return T


def rotz(a: float) -> np.ndarray:
    """Rotation about z by a (4 x 4)."""
    T = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def pose_from_spec(spec) -> np.ndarray:
    """Pose given in scene.json as {"p": [x, y, z], "rz": angle} -> trans(p) @ rotz(rz); with "look_at": [x, y, z],
    the rotation before rotz(rz) is the 'look-at' frame: z_c towards look_at, y_c as close as possible to +y_w."""
    T = trans(spec["p"])
    if "look_at" in spec:
        z = np.asarray(spec["look_at"], float) - T[:3, 3]
        z /= np.linalg.norm(z)
        y = np.array([0.0, 1.0, 0.0]) - z[1] * z
        y /= np.linalg.norm(y)
        T[:3, :3] = np.c_[np.cross(y, z), y, z]
    return T @ rotz(spec.get("rz", 0.0))


def camera_K(cam: dict) -> np.ndarray:
    """Intrinsic matrix of a camera entry of scene.json (calls your intrinsic_matrix)."""
    return intrinsic_matrix(cam["f"], cam["rho_w"], cam["rho_h"], cam["u0"], cam["v0"])


def target_points(scene: dict):
    """Square planar target (fiducial) of side s in the plane z_t = 0 of its frame {t}; returns (P_w (4 x 3) world
    points, xy_t (4 x 2) target-plane coordinates).  Order: (-s/2, -s/2), (-s/2, s/2), (s/2, s/2), (s/2, -s/2)."""
    s = scene["target"]["side"] / 2.0
    xy = np.array([[-s, -s], [-s, s], [s, s], [s, -s]])
    T_wt = pose_from_spec(scene["target"]["T_wt"])
    P = (T_wt[:3, :3] @ np.c_[xy, np.zeros(4)].T).T + T_wt[:3, 3]
    return P, xy


def load_calib(path=DATA / "calib_points.csv"):
    """Calibration data of the endoscope camera: P (N x 3) points of the 3-D calibration object (m, in its own frame,
    which is the world frame of the calibration) and p (N x 2) their measured pixel coordinates (noisy)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    P = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in rows])
    p = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    return P, p


def decompose_camera_matrix(C):
    """Split C = K [R | p] (scaled as camera_matrix_dlt returns it) into K (upper triangular, K[2, 2] = 1) and the
    camera pose T_cw = (R, p) (world in camera), by Gram-Schmidt from the third row (an RQ decomposition,
    Corke Sec. 11.2.2, p. 264)."""
    C = np.asarray(C, float)
    m1, m2, m3 = C[0, :3], C[1, :3], C[2, :3]
    k33 = np.linalg.norm(m3)
    r3 = m3 / k33
    k23 = m2 @ r3
    w2 = m2 - k23 * r3
    k22 = np.linalg.norm(w2)
    r2 = w2 / k22
    k13, k12 = m1 @ r3, m1 @ r2
    w1 = m1 - k12 * r2 - k13 * r3
    k11 = np.linalg.norm(w1)
    r1 = w1 / k11
    K = np.array([[k11, k12, k13], [0.0, k22, k23], [0.0, 0.0, k33]])
    R = np.array([r1, r2, r3])
    p = np.linalg.solve(K, C[:, 3])
    K = K / k33
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return K, T


def endoscope_setup(scene: dict) -> dict:
    """Start of the endoscope experiments: robot model, theta0, camera pose T0 = T_sc(theta0) (camera {c} = {b} at the
    scope tip), the port point (ell0 behind the tip on the optical axis), the centre c of the instrument path
    (Z0 in front of the tip) and the endoscope intrinsic matrix K."""
    e = scene["endoscope"]
    robot = cm.ur5_laparoscope()
    th0 = np.array(e["theta0"], float)
    T0 = mr.FKinBody(robot["M"], robot["Blist"], th0)
    z0, p0 = T0[:3, 2], T0[:3, 3]
    return dict(robot=robot, theta0=th0, T0=T0, port=p0 - e["ell0"] * z0, centre=p0 + e["Z0"] * z0,
                K=camera_K(scene["endoscope_camera"]))


def instrument_tip(scene: dict, setup: dict, t: float) -> np.ndarray:
    """World position of the instrument tip that the camera must keep centred: still for t < t_still, then a circle
    of radius a in the plane of x_c0, y_c0 (frequency f_hz) while it moves +-dz along z_c0 (same frequency)."""
    e = scene["endoscope"]
    T0 = setup["T0"]
    s = max(t - e["t_still"], 0.0)
    w = 2.0 * math.pi * e["f_hz"]
    return (setup["centre"] + e["a"] * (math.cos(w * s) * T0[:3, 0] + math.sin(w * s) * T0[:3, 1])
            + e["dz"] * math.sin(w * s) * T0[:3, 2])


def simulate_endoscope(scene: dict, lam=None, delay=None, Z_hat=None, k_z=None, k_rcm=None, T_end=None) -> dict:
    """Closed loop of the endoscope holder at the camera rate h (Euler integration of the joints).  At sample k:
        T_sc = FK(theta_k);  (p_k, Z_k) = project_points(K, T_sc, tip(t_k));  ell, e_port = rcm_errors(T_sc, port)
        the controller sees the image `delay` samples old: p_m = p_{k-delay}, Z_m = Z_{k-delay} (p_0 before that)
        depth used: Z_m (stereo endoscope) or the constant Z_hat;  zoom d_dot = k_z (Z_m - Z_des) (only with Z_m)
        xi, V = rcm_ibvs_velocity(p_m, p*, Z_used, K, lam, ell, d_dot);  V += k_rcm (0, 0, 0, e_port_x, e_port_y, 0)
        theta_dot = joint_rates(Blist, theta_k, V, qdot_max);  theta_{k+1} = theta_k + h theta_dot
    p* = principal point.  Arguments left as None take the values of scene['endoscope'].
    Returns dict t, p (N x 2), Z, err (pixel distance from p*), ell, e_lat (m), theta, theta_dot, xi, V, sat
    (common scale factor applied), in_view."""
    e = scene["endoscope"]
    lam = e["lam"] if lam is None else lam
    delay = e["delay"] if delay is None else int(delay)
    k_z = e["k_z"] if k_z is None else k_z
    k_rcm = e["k_rcm"] if k_rcm is None else k_rcm
    T_end = e["T_end"] if T_end is None else T_end
    h = e["h"]
    st = endoscope_setup(scene)
    robot, K = st["robot"], st["K"]
    cam = scene["endoscope_camera"]
    p_star = np.array([[cam["u0"], cam["v0"]]], float)
    N = int(round(T_end / h))
    th = st["theta0"].copy()
    out = {k: [] for k in ("t", "p", "Z", "err", "ell", "e_lat", "theta", "theta_dot", "xi", "V", "sat", "in_view")}
    hist = []
    for k in range(N + 1):
        t = k * h
        T_sc = mr.FKinBody(robot["M"], robot["Blist"], th)
        p, Z = project_points(K, T_sc, instrument_tip(scene, st, t))
        ell, e_port = rcm_errors(T_sc, st["port"])
        hist.append((p, Z))
        out["t"].append(t)
        out["p"].append(p[0])
        out["Z"].append(Z[0])
        out["err"].append(float(np.linalg.norm(p[0] - p_star[0])))
        out["ell"].append(ell)
        out["e_lat"].append(float(np.linalg.norm(e_port)))
        out["theta"].append(th.copy())
        out["in_view"].append(bool(abs(p[0, 0] - cam["u0"]) <= cam["W"] / 2 and abs(p[0, 1] - cam["v0"]) <= cam["H"] / 2))
        if k == N:
            break
        p_m, Z_m = hist[max(0, k - delay)]
        if Z_hat is None:
            Z_used, d_dot = Z_m, k_z * (Z_m[0] - e["Z_des"])
        else:
            Z_used, d_dot = np.array([float(Z_hat)]), 0.0
        xi, V = rcm_ibvs_velocity(p_m, p_star, Z_used, K, lam, ell, d_dot)
        V = V + k_rcm * np.array([0.0, 0.0, 0.0, e_port[0], e_port[1], 0.0])
        th_dot = joint_rates(robot["Blist"], th, V, robot["qdot_max"])
        th_free = joint_rates(robot["Blist"], th, V)
        out["sat"].append(bool(np.max(np.abs(th_free)) > robot["qdot_max"] * (1 + 1e-12)))
        out["theta_dot"].append(th_dot)
        out["xi"].append(xi)
        out["V"].append(V)
        th = th + h * th_dot
    return {k: np.array(v) for k, v in out.items()}


def endoscope_metrics(sim: dict, t_from: float = 5.0) -> dict:
    """rms_err / max_err (pixels, t >= t_from), final_still_err (pixels at t = 3 s, end of the still phase),
    max_e_lat (m, whole run), ell_min / ell_max (m), peak_qdot (rad/s), sat_frac, view_frac."""
    sel = sim["t"] >= t_from - 1e-9
    k3 = int(np.argmin(np.abs(sim["t"] - 3.0)))
    return dict(rms_err=float(np.sqrt(np.mean(sim["err"][sel] ** 2))), max_err=float(np.max(sim["err"][sel])),
                final_still_err=float(sim["err"][k3]), max_e_lat=float(np.max(sim["e_lat"])),
                ell_min=float(np.min(sim["ell"])), ell_max=float(np.max(sim["ell"])),
                peak_qdot=float(np.max(np.abs(sim["theta_dot"]))), sat_frac=float(np.mean(sim["sat"])),
                view_frac=float(np.mean(sim["in_view"])))


def servo_metrics(run: dict, cam: dict, P_w=None) -> dict:
    """Metrics of a run_ibvs / run_pbvs result: n_iter, converged, final_err (px), path_len (camera centre, m),
    retreat (largest increase of the mean feature depth above its start value, m), min_margin (smallest distance of
    any feature to the image border, px; negative = it left the image), straightness (image path length of the
    features / straight-line distance, mean over points; 1 = straight)."""
    T, p, Z = run["T"], run["p"], run["Z"]
    path = float(np.sum(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)))
    mz = Z.mean(axis=1)
    margin = np.minimum.reduce([p[:, :, 0], cam["W"] - p[:, :, 0], p[:, :, 1], cam["H"] - p[:, :, 1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=2).sum(axis=0)
    chord = np.linalg.norm(p[-1] - p[0], axis=1)
    return dict(n_iter=int(run["n_iter"]), converged=bool(run["converged"]), final_err=float(run["err"][-1]),
                path_len=path, retreat=float(np.max(mz) - mz[0]), min_margin=float(np.min(margin)),
                straightness=float(np.mean(seg / np.maximum(chord, 1e-12))))


# ==========================================================================================
# Part A -- pinhole camera and calibration
# ==========================================================================================
def intrinsic_matrix(f: float, rho_w: float, rho_h: float, u0: float, v0: float) -> np.ndarray:
    """A1. Intrinsic (camera parameter) matrix K of Corke Eq. 11.7 with the focal length folded in:
    K = [[f/rho_w, 0, u0], [0, f/rho_h, v0], [0, 0, 1]] (f and pixel sizes in metres, u0, v0 in pixels)."""
    # TODO: 3 x 3 array; f/rho_w and f/rho_h on the diagonal, (u0, v0) in the last column.
    raise NotImplementedError("intrinsic_matrix")


def project_points(K, T_wc, P_w):
    """A2. Project world points into the camera at pose T_wc (camera frame in the world).
    P_w: (N, 3) or (3,).  Returns (p, Z): p (N, 2) pixel coordinates (u, v), Z (N,) depths in {c}.
    Raise ValueError if any point is not in front of the camera (Z <= 0)."""
    # TODO: P = np.atleast_2d(...); P_c = R^T (P - p) with (R, p) of T_wc (or mr.TransInv(T_wc));
    #       Z = third column of P_c; raise ValueError if any Z <= 0;
    #       (u, v) = first two entries of K @ (P_c / Z) for every point.
    raise NotImplementedError("project_points")


def camera_matrix_dlt(P_w, p) -> np.ndarray:
    """A3. Direct linear transform (Corke Eqs. 11.15-11.16): the 3 x 4 camera matrix C with p ~ C [P; 1] from N >= 6
    non-coplanar points.  Stack two rows per point, take the right singular vector of the smallest singular value,
    then scale C so that ||C[2, :3]|| = 1 and the points have positive depth (C[2] . [P; 1] > 0 on average).
    Raise ValueError if N < 6."""
    # TODO: raise ValueError for fewer than 6 points.  For each point build the two rows
    #       [X, Y, Z, 1, 0, 0, 0, 0, -uX, -uY, -uZ, -u] and [0, 0, 0, 0, X, Y, Z, 1, -vX, -vY, -vZ, -v];
    #       C = last row of Vt of np.linalg.svd(A), reshaped (3, 4); divide by ||C[2, :3]||;
    #       flip the sign if the mean of C[2] . [P; 1] over the points is negative.
    raise NotImplementedError("camera_matrix_dlt")


def reprojection_rms(C, P_w, p) -> float:
    """A4. Root-mean-square reprojection error (pixels): sqrt(mean_i ||pi(C, P_i) - p_i||^2), pi = projection by C."""
    # TODO: homogeneous projection q = C [P; 1] for all points, (u, v) = q[:2] / q[2];
    #       return sqrt(mean of squared distances to p).
    raise NotImplementedError("reprojection_rms")


# ==========================================================================================
# Part B -- interaction matrix
# ==========================================================================================
def interaction_matrix(p, Z, K) -> np.ndarray:
    """B1. Stacked interaction matrix (image Jacobian) of N point features, (2N, 6), rows u1, v1, u2, v2, ...,
    columns in MR twist order (omega_x, omega_y, omega_z, v_x, v_y, v_z) -- Corke Eq. 15.6 with the blocks swapped
    (see the module docstring).  p: (N, 2) pixels; Z: scalar (same depth for all points) or (N,)."""
    # TODO: Z = np.broadcast_to(np.asarray(Z, float), (N,)); f_u = K[0, 0], f_v = K[1, 1], u0 = K[0, 2],
    #       v0 = K[1, 2]; for point i: x = (u - u0)/f_u, y = (v - v0)/f_v; row 2i and row 2i+1 as in the
    #       module docstring (MR order: the three angular columns first).
    raise NotImplementedError("interaction_matrix")


# ==========================================================================================
# Part C -- image-based visual servoing (free-flying camera)
# ==========================================================================================
def ibvs_velocity(p, p_star, Z_hat, K, lam: float) -> np.ndarray:
    """C1. IBVS law (Corke Eq. 15.11): V_c = -lam L^+ (p - p*), L = interaction_matrix(p, Z_hat, K), the error
    stacked as (u1 - u1*, v1 - v1*, u2 - u2*, ...).  Returns the camera body twist (6,), MR order."""
    # TODO: e = (p - p_star).reshape(-1); return -lam * np.linalg.pinv(interaction_matrix(p, Z_hat, K)) @ e
    raise NotImplementedError("ibvs_velocity")


def camera_step(T_wc, V_c, h: float) -> np.ndarray:
    """C2. Move the camera with the constant body twist V_c for h seconds: T_wc exp([V_c] h) (MR Eq. 3.88)."""
    # TODO: T_wc @ mr.MatrixExp6(mr.VecTose3(V_c * h))
    raise NotImplementedError("camera_step")


def run_ibvs(T_wc0, P_w, p_star, K, lam: float, h: float, Z_hat=None, n_max: int = 400, tol: float = 0.5,
             K_hat=None) -> dict:
    """C3. IBVS loop.  For k = 0, 1, ...: (p_k, Z_k) = project_points(K, T_k, P_w);
    err_k = sqrt(mean_i ||p_ik - p_i*||^2) (pixels); stop if err_k <= tol (converged) or k == n_max; otherwise
    V_k = ibvs_velocity(p_k, p*, Z_used, K_model, lam) with Z_used = Z_k if Z_hat is None else Z_hat (scalar or (N,))
    and K_model = K if K_hat is None else K_hat;  cond_k = 2-norm condition number of the interaction matrix used;
    T_{k+1} = camera_step(T_k, V_k, h).
    Returns dict T (n+1, 4, 4), p (n+1, N, 2), Z (n+1, N), err (n+1,), V (n, 6), cond (n,), converged, n_iter = n."""
    # TODO: loop k = 0 .. n_max as described above; append T, p, Z, err to lists; break when err <= tol
    #       (converged) or k == n_max; otherwise compute V (and cond with np.linalg.cond of the same
    #       interaction matrix) and step the camera.  Convert the lists to arrays (V with shape (n, 6),
    #       also when n = 0: np.array(V).reshape(-1, 6)) and add converged and n_iter.
    raise NotImplementedError("run_ibvs")


# ==========================================================================================
# Part D -- pose from the image and position-based visual servoing
# ==========================================================================================
def homography_dlt(xy_t, p) -> np.ndarray:
    """D1. Planar homography (Corke Eq. 14.12) with p ~ H [x; y; 1] from N >= 4 points of the target plane
    (xy_t (N, 2), metres) and their pixels p (N, 2): DLT with two rows per point, SVD null vector, scaled so that
    H[2, 2] = 1.  Raise ValueError if N < 4."""
    # TODO: raise ValueError for fewer than 4 points; rows [-x, -y, -1, 0, 0, 0, u x, u y, u] and
    #       [0, 0, 0, -x, -y, -1, v x, v y, v]; H = last row of Vt reshaped (3, 3); return H / H[2, 2].
    raise NotImplementedError("homography_dlt")


def pose_from_homography(H, K) -> np.ndarray:
    """D2. Pose T_ct of the target plane in the camera from H ~ K [r1 r2 t]:  [m1 m2 m3] = K^-1 H,
    s = 2 / (||m1|| + ||m2||), with the sign of s chosen so that t_z = s m3_z > 0;  r1 = s m1, r2 = s m2,
    r3 = r1 x r2; R = the rotation closest to [r1 r2 r3] (SVD: U diag(1, 1, det(U V^T)) V^T); t = s m3."""
    # TODO: Mx = np.linalg.solve(K, H); s = 2 / (||Mx[:, 0]|| + ||Mx[:, 1]||); flip s if s * Mx[2, 2] < 0;
    #       r1, r2 = s Mx[:, 0], s Mx[:, 1]; R0 = [r1 r2 r1 x r2]; U, S, Vt = svd(R0);
    #       R = U diag(1, 1, det(U Vt)) Vt; t = s Mx[:, 2]; return the 4 x 4 transform (R, t).
    raise NotImplementedError("pose_from_homography")


def pbvs_velocity(T_ct, T_cstar_t, lam: float) -> np.ndarray:
    """D3. PBVS law: the desired camera {c*} seen from the current camera is T_cc* = T_ct T_c*t^-1; command the
    body twist V_c = lam * se3ToVec(MatrixLog6(T_cc*)) (6,), MR order (so that exp([V_c] 1/lam) reaches {c*})."""
    # TODO: T_cc* = T_ct @ mr.TransInv(T_cstar_t); return lam * mr.se3ToVec(mr.MatrixLog6(T_cc*))
    raise NotImplementedError("pbvs_velocity")


def run_pbvs(T_wc0, P_w, xy_t, T_cstar_t, p_star, K, lam: float, h: float, n_max: int = 400, tol: float = 0.5,
             K_hat=None) -> dict:
    """D4. PBVS loop with the same structure, stopping rule and returned dict as run_ibvs (cond is omitted):
    at each step project (with the true K), estimate T_ct = pose_from_homography(homography_dlt(xy_t, p_k), K_model)
    with K_model = K if K_hat is None else K_hat, V_k = pbvs_velocity(T_ct, T_c*t, lam), T_{k+1} = camera_step.
    The image error err_k (to p*) is used only for the stopping test and the record."""
    # TODO: same loop as run_ibvs (no cond); the velocity is
    #       pbvs_velocity(pose_from_homography(homography_dlt(xy_t, p), K_model), T_cstar_t, lam).
    raise NotImplementedError("run_pbvs")


# ==========================================================================================
# Part E -- the endoscope through a port (RCM-constrained IBVS on the UR5)
# ==========================================================================================
def rcm_twist_basis(ell: float) -> np.ndarray:
    """E1. The 6 x 4 matrix W(ell) with V_c = W xi for xi = (omega_x, omega_y, omega_z, d_dot): the body twists of the
    camera that keep the port point r = (0, 0, -ell) of {c} on the optical axis (v = (ell omega_y, -ell omega_x,
    d_dot); the scope point in the port then moves along z_c only, with speed d_dot)."""
    # TODO: 6 x 4 zeros; identity in the top-left 3 x 3 (omega = (xi0, xi1, xi2));
    #       v_x = ell * xi1, v_y = -ell * xi0, v_z = xi3.
    raise NotImplementedError("rcm_twist_basis")


def rcm_ibvs_velocity(p, p_star, Z_hat, K, lam: float, ell: float, d_dot: float = 0.0):
    """E2. Keep the instrument centred with pan and tilt only.  A = interaction_matrix(p, Z_hat, K) W(ell) (2N x 4);
    the roll omega_z = 0 (keep the image horizon) and the insertion speed d_dot is given (zoom loop);
    (omega_x, omega_y) = A[:, :2]^+ (-lam e - A[:, 3] d_dot), e = p - p* stacked.
    Returns (xi (4,) = (omega_x, omega_y, 0, d_dot), V_c = W xi (6,))."""
    # TODO: e = (p - p_star).reshape(-1); W = rcm_twist_basis(ell); A = interaction_matrix(p, Z_hat, K) @ W;
    #       w = np.linalg.pinv(A[:, :2]) @ (-lam * e - A[:, 3] * d_dot);
    #       xi = (w0, w1, 0, d_dot); return xi, W @ xi.
    raise NotImplementedError("rcm_ibvs_velocity")


def joint_rates(Blist, theta, V_c, qdot_max=None) -> np.ndarray:
    """E3. Joint velocities that give the camera (= end-effector {b}) body twist V_c: theta_dot = J_b(theta)^+ V_c
    (MR Eq. 6.7, J_b from mr.JacobianBody).  If qdot_max is given and max_i |theta_dot_i| > qdot_max, scale the WHOLE
    vector by one factor so that the largest is qdot_max (the twist direction is kept)."""
    # TODO: qd = np.linalg.pinv(mr.JacobianBody(Blist, theta)) @ V_c; if qdot_max is given and
    #       max |qd| > qdot_max, multiply qd by qdot_max / max |qd|.
    raise NotImplementedError("joint_rates")


def rcm_errors(T_sc, p_port):
    """E4. Where is the port seen from the camera?  p_c = T_sc^-1 p_port; returns (ell, e): ell = -p_c[2] (insertion
    depth of the tip beyond the port, m) and e = p_c[:2] (2,) (lateral offset of the port from the scope axis, m;
    zero when the remote centre of motion is respected)."""
    # TODO: p_c = mr.TransInv(T_sc) @ [p_port, 1]; return float(-p_c[2]), p_c[:2] (a copy).
    raise NotImplementedError("rcm_errors")
