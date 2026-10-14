"""baseline.py - load each scenario robot and check the BASELINE design at its nominal task.

Usage:
    python baseline.py            # all six scenarios
    python baseline.py 4          # one scenario
    python baseline.py --json     # also write baseline_results.json

The baseline is the starting point, not a solution: several acceptance tests are
deliberately FAILED by the baseline design (e.g. S4 corridor clearance, S5 elbow torque),
and every test marked TEAM must be implemented by the team in team_design.py.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from kintools import (fk, jacobian_body_from_space, point_jacobian, manipulability,
                      sigma_min,
                      ik_body, trans, frame_from_z, moved_points, sample_segments,
                      gravity_wrench_space, holding_torques, point_line_distance,
                      within_limits, pose_error)
from scenarios import scenario, ALL, DEG

FREE_ROLL = (1, 1, 0, 1, 1, 1)   # leave rotation about the tool z-axis free


class Check:
    """One acceptance-test result: passed is True/False, or None for TEAM tests."""

    def __init__(self, tid, text, value=None, limit=None, passed=None):
        self.tid, self.text, self.value, self.limit, self.passed = tid, text, value, limit, passed

    def row(self):
        st = {True: "PASS", False: "FAIL", None: "TEAM"}[self.passed]
        val = "" if self.value is None else (f"{self.value:.4g}" if isinstance(self.value, (float, int, np.floating)) else str(self.value))
        return f"  {self.tid:<6} {st:<5} {self.text:<58} {val:>10} {'' if self.limit is None else self.limit}"

    def as_dict(self):
        v = self.value
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        return dict(id=self.tid, text=self.text, value=v, limit=self.limit, passed=self.passed)


def lin_manip(r, th):
    """Manipulability of the linear-velocity part of J_b (MR Sec. 5.4)."""
    return manipulability(jacobian_body_from_space(r.Slist, r.M, th)[3:])

# ----------------------------------------------------------------------------

def baseline_1():
    s = scenario(1); r, t = s.robot, s.task
    Td = trans(t["nominal"], t["R_d"])
    th, ok, it, _ = ik_body(r.Slist, r.M, Td, t["seed"], limits=r.limits)
    m = lin_manip(r, th)
    res = dict(theta_deg=th / DEG, mu1=m["mu1"], mu3=m["mu3"],
               sigma_min_Jb=sigma_min(jacobian_body_from_space(r.Slist, r.M, th)))
    checks = [
        Check("AT1-1", "nominal handle pose reachable within safe limits", None, None,
              ok and within_limits(th, r.limits)),
        Check("AT1-2", "mu1(J_v,b) at nominal pose <= limit", m["mu1"], t["mu1_max"], m["mu1"] <= t["mu1_max"]),
        Check("AT1-3", "all circle + line waypoints reachable, safe limits"),
        Check("AT1-4", "mu1 <= limit along both therapy paths"),
        Check("AT1-5", "therapy-grid coverage >= 80 % (reachable and mu1 <= limit)"),
        Check("AT1-6", "handle speed <= 0.25 m/s, no joint jump > 10 deg/waypoint"),
    ]
    return s, res, checks


def head_clearance(r, th, t):
    pts = sample_segments(moved_points(r.Slist, th, r.points)[:-1])  # links up to the wrist
    d = np.linalg.norm(pts - t["head_centre"], axis=1) - t["head_radius"]
    return float(d.min())


def elbow_head_distance(r, th, t):
    """Distance from the elbow (chain point 2) to the head centre."""
    return float(np.linalg.norm(moved_points(r.Slist, th, r.points)[2] - t["head_centre"]))


def baseline_2():
    s = scenario(2); r, t = s.robot, s.task
    p = t["mouth"] - t["stop_distance"] * t["approach_dir"]
    th, ok, it, _ = ik_body(r.Slist, r.M, trans(p, t["R_d"]), t["seed"], limits=r.limits)
    Jb = jacobian_body_from_space(r.Slist, r.M, th)
    rank = np.linalg.matrix_rank(Jb, tol=1e-6)
    m = lin_manip(r, th)
    clr = head_clearance(r, th, t)
    T = fk(r.Slist, r.M, th)
    tilt = float(np.arccos(np.clip(T[:3, 2] @ np.array([0, 0, 1.0]), -1, 1)))
    elbow = elbow_head_distance(r, th, t)
    res = dict(elbow_head=elbow, theta_deg=th / DEG, rank_Jb=int(rank), null_dim=r.n - int(rank), mu1=m["mu1"],
               head_clearance=clr, tilt_deg=tilt / DEG)
    checks = [
        Check("AT2-1", "spoon stop pose reachable within limits", None, None, ok and within_limits(th, r.limits)),
        Check("AT2-2", "spoon tilt <= 5 deg at stop pose", tilt / DEG, 5.0, tilt <= t["tilt_max"] + 1e-9),
        Check("AT2-3", "arm (to wrist) clearance from head sphere >= 0.10 m", clr, t["clearance_min"], clr >= t["clearance_min"]),
        Check("AT2-4", "redundancy: dim null(J_b) = 1 at stop pose", r.n - int(rank), 1, r.n - int(rank) == 1),
        Check("AT2-5", "0.15 m straight approach, v <= 0.05 m/s, tilt <= 5 deg"),
        Check("AT2-6", "elbow >= 0.52 m from head centre (use the null space)", elbow, t["elbow_head_min"], elbow >= t["elbow_head_min"]),
    ]
    return s, res, checks


def baseline_3():
    s = scenario(3); r, t = s.robot, s.task
    th0 = t["seed"]; sols = []; okall = True; rcm = []; err_wc = []; err_rms = []
    for u in t["directions"]:
        p_tip = t["entry"] - t["standoff"] * u
        th, ok, it, _ = ik_body(r.Slist, r.M, trans(p_tip, frame_from_z(u)), th0,
                                mask=FREE_ROLL, limits=r.limits)
        above_table = moved_points(r.Slist, th, r.points)[1:, 2].min() >= -1e-9
        okall &= ok and within_limits(th, r.limits) and above_table
        T = fk(r.Slist, r.M, th)
        rcm.append(point_line_distance(t["entry"], T[:3, 3], T[:3, 2]))
        # needle tip at the target, i.e. (standoff + depth) beyond the guide tip along z_b
        Jp = point_jacobian(r.Slist, r.M, th, [0, 0, t["standoff"] + t["depth"]])
        err_wc.append(float(np.sum(np.linalg.norm(Jp, axis=0)) * t["enc_res"]))
        err_rms.append(float(np.sqrt(np.sum(np.linalg.norm(Jp, axis=0) ** 2)) * t["enc_res"] / np.sqrt(3)))
        sols.append(th); th0 = th
    res = dict(theta_deg=[x / DEG for x in sols], rcm_err=max(rcm), err_worst_mm=max(err_wc) * 1e3,
               err_rms_mm=max(err_rms) * 1e3)
    checks = [
        Check("AT3-1", "guide on all 3 entry->target lines; limits; links above table", None, None, bool(okall)),
        Check("AT3-2", "guide axis passes through entry point (<= 0.1 mm)", max(rcm) * 1e3, 0.1, max(rcm) <= 1e-4),
        Check("AT3-3", "worst-case encoder-induced target error <= 1.5 mm", max(err_wc) * 1e3, 1.5, max(err_wc) <= t["accuracy_max"]),
        Check("AT3-4", "pivot between directions with RCM error <= 0.5 mm"),
        Check("AT3-5", "15 mm guide advance along its axis, lateral error <= 0.5 mm"),
    ]
    return s, res, checks


def eod_clearances(r, th, t):
    """(corridor margin, ground margin) in metres; positive = clear, accounting for link radius."""
    pts = sample_segments(moved_points(r.Slist, th, r.points)[1:], 0.01)
    inside = pts[:, 0] >= t["corridor"]["x"][0]
    corr = (t["underside_z"] - r.link_radius) - pts[inside, 2].max() if inside.any() else np.inf
    gnd = pts[:, 2].min() - (t["ground_z"] + r.link_radius)
    return float(corr), float(gnd)


def baseline_4():
    s = scenario(4); r, t = s.robot, s.task
    Td = trans(t["object"], t["R_d"])
    res = {}; checks = []
    for tag, seed in (("up", t["seed_up"]), ("down", t["seed_down"])):
        th, ok, it, _ = ik_body(r.Slist, r.M, Td, seed, limits=r.limits)
        corr, gnd = eod_clearances(r, th, t)
        m = lin_manip(r, th)
        res[tag] = dict(theta_deg=th / DEG, ok=ok, corridor_margin=corr, ground_margin=gnd,
                        theta5_deg=th[4] / DEG, mu1=m["mu1"])
    best = max(res, key=lambda k: min(res[k]["corridor_margin"], res[k]["ground_margin"]))
    b = res[best]
    checks = [
        Check("AT4-1", "grasp pose reachable within limits (either elbow branch)", None, None, res["up"]["ok"] or res["down"]["ok"]),
        Check("AT4-2", f"corridor margin >= {t['margin']} m (best branch: elbow-{best})", b["corridor_margin"], t["margin"], b["corridor_margin"] >= t["margin"]),
        Check("AT4-3", f"ground margin >= {t['margin']} m (best branch: elbow-{best})", b["ground_margin"], t["margin"], b["ground_margin"] >= t["margin"]),
        Check("AT4-4", "wrist singularity margin |theta5| >= 10 deg at grasp", abs(b["theta5_deg"]), 10.0, abs(b["theta5_deg"]) >= 10.0),
        Check("AT4-5", "teleop resolved-rate approach 0.20 m, error <= 5 mm, rates ok"),
        Check("AT4-6", "corridor + ground margins along the whole approach"),
    ]
    res["best_branch"] = best
    return s, res, checks


def baseline_5():
    s = scenario(5); r, t = s.robot, s.task
    res = {}; ok_all = True; tau_ratio = 0.0; elbow = np.inf
    th0 = t["seed"]
    for name, p in t["poses"].items():
        th, ok, it, _ = ik_body(r.Slist, r.M, trans(p, t["R_d"]), th0, limits=r.limits)
        ok_all &= ok and within_limits(th, r.limits)
        tau = holding_torques(r.Slist, th, gravity_wrench_space(p, t["mass"]))
        ratio = np.abs(tau) / (r.tau_rated / t["safety_factor"])
        tau_unit = np.abs(holding_torques(r.Slist, th, gravity_wrench_space(p, 1.0)))
        with np.errstate(divide="ignore"):
            m_max = float(np.min(np.where(tau_unit > 1e-9, r.tau_rated / t["safety_factor"] / tau_unit, np.inf)))
        m = lin_manip(r, th)
        res[name] = dict(theta_deg=th / DEG, tau=tau, ratio=ratio, m_max=m_max, mu3=m["mu3"])
        tau_ratio = max(tau_ratio, ratio.max()); elbow = min(elbow, abs(th[2]))
        th0 = th
    worst = max(res, key=lambda k: res[k]["ratio"].max())
    checks = [
        Check("AT5-1", "all 3 lifting poses reachable with level cradle", None, None, bool(ok_all)),
        Check("AT5-2", f"max |tau_i|/(tau_rated,i/SF) <= 1 (worst: {worst})", float(tau_ratio), 1.0, tau_ratio <= 1.0),
        Check("AT5-3", "elbow singularity margin |theta3| >= 10 deg", elbow / DEG, 10.0, elbow >= t["elbow_min"]),
        Check("AT5-4", "lift path A->B->C continuous, cradle tilt <= 5 deg"),
        Check("AT5-5", "lift speed 0.10 m/s within joint rate limits"),
    ]
    res["worst_pose"] = worst
    return s, res, checks


def baseline_6():
    s = scenario(6); r, t = s.robot, s.task
    tip = t["circle_centre"] + np.array([t["circle_radius"], 0, 0])
    u = tip - t["port"]; u /= np.linalg.norm(u)
    th, ok, it, _ = ik_body(r.Slist, r.M, trans(tip, frame_from_z(u)), t["seed"],
                            mask=FREE_ROLL, limits=r.limits)
    T = fk(r.Slist, r.M, th)
    rcm = point_line_distance(t["port"], T[:3, 3], T[:3, 2])
    tip_err = float(np.linalg.norm(T[:3, 3] - tip))
    # port must lie on the shaft, between instrument back end and tip
    s_port = float((T[:3, 3] - t["port"]) @ T[:3, 2])      # distance tip -> port along shaft
    on_shaft = 0.0 < s_port < r.shaft + th[6]
    Jb = jacobian_body_from_space(r.Slist, r.M, th)
    res = dict(theta=th, rcm_err=rcm, tip_err=tip_err, port_from_tip=s_port,
               insertion=th[6], null_dim=r.n - 5, sigma_min_Jb=sigma_min(Jb))
    checks = [
        Check("AT6-1", "start pose: tip on circle, shaft through port, limits", None, None,
              ok and within_limits(th, r.limits) and on_shaft),
        Check("AT6-2", "RCM error at start pose <= 2 mm", rcm * 1e3, 2.0, rcm <= t["rcm_max"]),
        Check("AT6-3", "tip error at start pose <= 1 mm", tip_err * 1e3, 1.0, tip_err <= t["tip_max"]),
        Check("AT6-4", "track 20 mm circle: RCM <= 2 mm, tip <= 1 mm throughout"),
        Check("AT6-5", "5:1 motion scaling, tip speed <= 20 mm/s, rates ok"),
    ]
    return s, res, checks


BASELINES = {1: baseline_1, 2: baseline_2, 3: baseline_3, 4: baseline_4, 5: baseline_5, 6: baseline_6}


def run(sids=ALL, verbose=True):
    out = {}
    for sid in sids:
        s, res, checks = BASELINES[sid]()
        out[sid] = (s, res, checks)
        if verbose:
            print(f"\nS{sid} [{s.domain}] {s.title} - robot: {s.robot.name} (n = {s.robot.n})")
            for c in checks:
                print(c.row())
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sids = tuple(int(a) for a in args) or ALL
    out = run(sids)
    if "--json" in sys.argv:
        data = {f"S{k}": dict(results=_jsonable(v[1]), checks=[c.as_dict() for c in v[2]])
                for k, v in out.items()}
        with open("baseline_results.json", "w") as f:
            json.dump(_jsonable(data), f, indent=1)
        print("\nwrote baseline_results.json")
