"""Run the BASELINE of each project scenario and print its metrics.

    python baseline.py            # all scenarios
    python baseline.py 3 5        # selected scenarios
    python baseline.py 2 --plot   # also save figures/baseline_s2.png

The baselines are deliberately simple (PD + gravity, stiffness without inertia compensation,
RRT + stop-and-go quintics). Your project must beat them and explain why (see scenario_sheets.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sbeg461_project import control, dyn, metrics, planning, tasks, traj
from sbeg461_project.models import load_scenario

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- S1
def baseline_s1(T_push: float = 2.5, T_reach: float = 3.0, engine: str = "auto") -> dict:
    s = load_scenario(1)
    p = s.params
    ctrl = control.tunnel_stiffness(s, p["path_A"], p["path_B"], p["K_normal"], p["B_normal"],
                                    p["B_tangent"])
    # static push test at the middle of the path: 5 N normal (+x) force held constant
    mid = 0.5 * (p["path_A"] + p["path_B"])
    q0 = tasks.planar_ik(s, *mid, elbow_sign=p["elbow_sign"])
    sim = dyn.simulate(s, ctrl, q0, np.zeros(2), T_push, 1e-3,
                       f_ext=lambda t, q, qd: np.array([5.0, 0, 0]), engine=engine)
    dx = s.fk(sim["q"][-1])[0, 3] - mid[0]
    K_meas = 5.0 / dx
    # reach with an active patient and a 10 N lateral pulse at t = 0.8 s
    qA = tasks.planar_ik(s, *p["path_A"], elbow_sign=p["elbow_sign"])
    sim2 = dyn.simulate(s, ctrl, qA, np.zeros(2), T_reach, 1e-3,
                        f_ext=tasks.patient_force_s1(s), engine=engine)
    X = np.array([s.fk(q)[:2, 3] for q in sim2["q"]])
    dev = np.abs(X[:, 0] - p["path_A"][0])
    # interaction force = robot force on the hand, f = (J_v^T)^+ tau (planar, square J_v)
    F = np.array([np.linalg.solve(control.tip_jacobian(s, q)[:2].T, tau)
                  for q, tau in zip(sim2["q"], sim2["tau"])])
    return dict(K_rendered=float(K_meas), max_lateral_dev=float(dev.max()),
                final_lateral_dev=float(dev[-1]), reach_progress=float(X[-1, 1] - p["path_A"][1]),
                max_interaction_force=float(np.max(np.linalg.norm(F, axis=1))), sim=sim2)


# ----------------------------------------------------------------------------- S2
def baseline_s2(cycles: float = 2.0, model_error: bool = True, engine: str = "auto") -> dict:
    s = load_scenario(2)
    p = s.params
    me = p["model_error"]
    model = s.scaled(me["mass_scale"], me["com_shift"]) if model_error else s
    ref = lambda t: traj.gait_reference(t, p["gait_period"])  # noqa: E731
    ctrl = control.pd_gravity(model, Kp=[400.0, 300.0], Kd=[40.0, 30.0], ref=ref)
    q0, qd0, _ = ref(0.0)
    sim = dyn.simulate(s, ctrl, q0, qd0, cycles * p["gait_period"], 1e-3, engine=engine)
    m = metrics.tracking_metrics(sim, ref, skip=p["gait_period"] * 0.5)
    m["sim"] = sim
    return m


# ----------------------------------------------------------------------------- S3
def baseline_s3(seed: int = 1) -> dict:
    s = load_scenario(3)
    p = s.params
    out = dict(pre_pose_err=float(np.linalg.norm(
        s.fk(p["q_pre"])[:3, 3] - (p["p_rcm"] + [0, 0, p["pre_insertion_height"]]))))
    out["straight_line_clearance"] = planning.path_clearance(s, [s.q_home, p["q_pre"]])
    path = None
    for k in range(5):
        path = planning.rrt_bidirectional(s, s.q_home, p["q_pre"], margin=0.02,
                                          rng=np.random.default_rng(seed + k))
        if path is not None:
            break
    path = planning.shortcut(s, path, margin=0.02, rng=np.random.default_rng(seed))
    out["path"] = path
    out["n_waypoints"] = len(path)
    out["path_clearance"] = planning.path_clearance(s, path)
    out["stop_and_go_time"] = traj.stop_and_go_time(s, path)
    # why task-space control is needed: a straight JOINT-space line from the pre-insertion
    # configuration to the inserted configuration violates the RCM
    import modern_robotics as mr
    X_end = tasks.insertion_reference(s, 4.0 + 0.25 * p["scan_period"] + 1.0)
    q_end, ok = mr.IKinSpace(s.Slist, s.M, X_end, p["q_pre"], 1e-10, 1e-10)
    out["ik_ok"] = bool(ok)
    errs = [tasks.rcm_error(s.fk(traj.joint_quintic(p["q_pre"], q_end, 1.0, t)[0]), p["p_rcm"])
            for t in np.linspace(0, 1, 101)]
    out["joint_line_rcm_error_max"] = float(np.max(errs))
    out["reference_rcm_error_max"] = float(max(
        tasks.rcm_error(tasks.insertion_reference(s, t), p["p_rcm"]) for t in np.linspace(0, 12, 241)))
    return out


# ----------------------------------------------------------------------------- S4
def baseline_s4(cycles: float = 1.0, engine: str = "auto") -> dict:
    s_nom = load_scenario(4)
    p = s_nom.params
    me = p["model_error"]
    extra = me["pack_mass_actual"] - p["pack_mass_nominal"]
    # true plant: heavier pack, CoM further back; controller model: nominal
    plant = s_nom.scaled(1.0, [(0, 0, 0), me["com_shift"]])
    plant.Glist[1] = plant.Glist[1] * (p["frame_mass"] + p["pack_mass_nominal"] + extra) / \
        (p["frame_mass"] + p["pack_mass_nominal"])
    ref = lambda t: traj.squat_reference(t, p["cycle_period"], p["knee_range"], p["trunk_lean"])  # noqa: E731
    ctrl = control.pd_gravity(s_nom, Kp=[1500.0, 1200.0], Kd=[120.0, 100.0], ref=ref)
    q0, qd0, _ = ref(0.0)
    sim = dyn.simulate(plant, ctrl, q0, qd0, cycles * p["cycle_period"], 1e-3, engine=engine)
    m = metrics.tracking_metrics(sim, ref)
    m["sim"] = sim
    return m


# ----------------------------------------------------------------------------- S5
def baseline_s5(seed: int = 0, T_hold: float = 2.0, engine: str = "auto") -> dict:
    s = load_scenario(5)
    p = s.params
    loaded = s.with_payload(p["payload_mass"], Ib=np.array([0.05, 0.06, 0.06]) * p["payload_mass"] / 5)
    out = {}
    legs = [("stow_to_pregrasp", s, s.q_home, p["q_pregrasp"]),
            ("pregrasp_to_bin", loaded, p["q_pregrasp"], p["q_bin"])]
    for name, model, qa, qb in legs:
        path = None
        for k in range(5):
            path = planning.rrt_bidirectional(s, qa, qb, margin=p["clearance"],
                                              rng=np.random.default_rng(seed + k))
            if path is not None:
                break
        path = planning.shortcut(s, path, margin=p["clearance"], rng=np.random.default_rng(seed))
        out[name] = dict(path=path, clearance=planning.path_clearance(s, path),
                         stop_and_go_time=traj.stop_and_go_time(model, path),
                         straight_line_clearance=planning.path_clearance(s, [qa, qb]))
    # disturbance: hold the payload at the pre-grasp configuration on a 10 deg slope with
    # 1 m/s^2, 2 Hz heave, using PD + gravity compensation that assumes level, still ground
    g_fn = tasks.base_gravity_s5(s)
    q_hold = p["q_pregrasp"]
    ref = lambda t: (q_hold, np.zeros(6), np.zeros(6))  # noqa: E731
    ctrl = control.pd_gravity(loaded, Kp=[800, 1500, 800, 100, 150, 30],
                              Kd=[80, 150, 80, 10, 15, 3], ref=ref)
    sim = dyn.simulate(loaded, ctrl, q_hold, np.zeros(6), T_hold, 1e-3, g_fn=g_fn, engine=engine)
    tip0 = loaded.fk(q_hold)[:3, 3]
    dev = np.array([np.linalg.norm(loaded.fk(q)[:3, 3] - tip0) for q in sim["q"]])
    out["hold_max_tip_dev"] = float(dev.max())
    out["hold_peak_tau_ratio"] = (np.max(np.abs(sim["tau"]), axis=0) / s.tau_max).tolist()
    out["sim"] = sim
    return out


# ----------------------------------------------------------------------------- S6

def s6_start(s):
    """Start configuration: tip at (0.65, -0.05) with the cradle horizontal (phi = 0)."""
    return tasks.planar_pose_ik(s, 0.65, -0.05, 0.0, s.q_home)


def baseline_s6(K_wall: float = 20000.0, engine: str = "auto") -> dict:
    s = load_scenario(6)
    p = s.params
    q0 = s6_start(s)
    x0 = s.fk(q0)[:3, 3]
    out = {}
    # static stiffness: 10 N downward on the cradle, reference fixed
    hold = lambda t: (x0[:2], np.zeros(2), np.zeros(2), 0.0)  # noqa: E731
    ctrl = control.task_stiffness(s, p["K_imp"], p["B_imp"], hold, dims=(0, 1),
                                  K_rot=p["K_rot"], B_rot=p["B_rot"])
    sim = dyn.simulate(s, ctrl, q0, np.zeros(3), 2.5, 1e-3,
                       f_ext=lambda t, q, qd: np.array([0, -10.0, 0]), engine=engine)
    out["K_rendered_y"] = float(10.0 / (x0[1] - s.fk(sim["q"][-1])[1, 3]))
    # 6 kg limb resting on the cradle (static load): sag
    sim = dyn.simulate(s, ctrl, q0, np.zeros(3), 2.5, 1e-3,
                       f_ext=lambda t, q, qd: np.array([0, -p["limb_mass"] * 9.81, 0]), engine=engine)
    out["limb_sag"] = float(x0[1] - s.fk(sim["q"][-1])[1, 3])
    # unexpected contact: reference moves +x at approach_speed for 2 s into a wall at wall_x
    v = p["approach_speed"]

    def ref(t):
        tt = min(t, 2.0)
        return (np.array([x0[0] + v * tt, x0[1]]), np.array([v if t < 2.0 else 0.0, 0.0]),
                np.zeros(2), 0.0)
    ctrl = control.task_stiffness(s, p["K_imp"], p["B_imp"], ref, dims=(0, 1),
                                  K_rot=p["K_rot"], B_rot=p["B_rot"])
    wall = tasks.wall_force_s6(s, K_wall)
    sim = dyn.simulate(s, ctrl, q0, np.zeros(3), 3.0, 1e-3, f_ext=wall, engine=engine)
    F = np.array([-wall(t, q, qd)[0] for t, q, qd in zip(sim["t"], sim["q"], sim["qd"])])
    out["contact_peak_force"] = float(F.max())
    out["contact_final_force"] = float(F[-1])
    out["sim"] = sim
    return out


RUNNERS = {1: baseline_s1, 2: baseline_s2, 3: baseline_s3, 4: baseline_s4, 5: baseline_s5,
           6: baseline_s6}


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=4)
    if isinstance(v, list) and v and isinstance(v[0], float):
        return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
    return str(v)


def _plot(k, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sim = res.get("sim")
    if sim is None:
        return None
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(sim["t"], sim["q"])
    ax[0].set_ylabel("q (rad)")
    ax[1].plot(sim["t"], sim["tau"])
    ax[1].set_ylabel("tau (N m)")
    ax[1].set_xlabel("t (s)")
    fig.suptitle(f"Scenario {k} baseline")
    out = HERE / "figures" / f"baseline_s{k}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def main(argv):
    plot = "--plot" in argv
    keys = [int(a) for a in argv if a.isdigit()] or list(RUNNERS)
    for k in keys:
        s = load_scenario(k)
        res = RUNNERS[k]()
        print(f"\n=== Scenario {k}: {s.name} ({s.domain}) ===")
        for key, v in res.items():
            if key in ("sim", "path"):
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk != "path":
                        print(f"  {key}.{kk:24s} {_fmt(vv)}")
            else:
                print(f"  {key:30s} {_fmt(v)}")
        if plot:
            f = _plot(k, res)
            if f:
                print(f"  figure: {f}")


if __name__ == "__main__":
    main(sys.argv[1:])
