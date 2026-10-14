"""Run the validation plan VP-1 on an emulated arm (simulation station) or on a course arm (hardware station).

    python run_session.py --backend sim --arm S3 --team T07            # simulation station, your starter.py
    python run_session.py --backend vendor --port /dev/ttyUSB0 --team T07 --offsets my_offsets.json   # TA only

Uses YOUR implementation (module named by --impl, default `starter`) for check_plan and run_logged.  On the
hardware backend the script first runs the safety-critical tests of that module and refuses to move the arm if
any of them fails; the independent SafetyGuard (safety_guard.py) and the emergency stop remain active anyway.

Output (in --out, default the current folder):
    log_<arm>_<date>_<team>_run1.csv      logged trajectory with the stop test (format of hwcompare.py)
    tips_<arm>_<date>_<team>.csv          tip measurements (accuracy block + repeatability block)
    log_<arm>_<date>_<team>_run1.json     metadata (arm, date, team, operator, offsets, plan, incidents)
"""
import argparse
import datetime as _dt
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _soft_limits(plan):
    lim = np.asarray(plan["hw_limits_model_deg"], float)
    m = plan["soft_limit_margin_deg"]
    return np.column_stack([lim[:, 0] + m, lim[:, 1] - m])


def _move_time(plan, dq_deg, T_min):
    cap = plan["speed_fraction"] * plan["rated_speed_deg_s"]
    return max(T_min, 1.5 * float(np.max(np.abs(dq_deg))) / (0.8 * cap))


def _move(impl, arm, plan, offsets, q_to_servo, T_min=1.0, hold=0.0):
    """Planned, checked move from the current reading to q_to (servo degrees); returns the run log."""
    q_from = np.asarray(arm.read(), float)
    q_to = np.asarray(q_to_servo, float)
    T = _move_time(plan, q_to - q_from, T_min)
    t, q = impl.smooth_segments(np.vstack([q_from, q_to]), T, plan["rate_hz"], t_hold=hold)
    chk = impl.check_plan(t, q, offsets, _soft_limits(plan), plan["rated_speed_deg_s"], plan["speed_fraction"],
                          plan["hw4_margin_deg"])
    if not chk["ok"]:
        raise RuntimeError(f"move refused by check_plan: margins {np.round(chk['margin_deg'], 1)}, "
                           f"peak speeds {np.round(chk['peak_speed'], 1)} deg/s")
    t0 = arm.time() + 0.02
    log = impl.run_logged(arm, t + t0, q)
    if log["aborted"]:
        raise RuntimeError("move aborted by the safety guard: " + log["aborted"])
    return log


def trajectory_run(impl, arm, plan, offsets):
    """Block 1: the logged trajectory R-P1-P2-P3-P4-R with the stop test (HW-2 ... HW-6)."""
    poses = {k: np.asarray(v, float) + offsets for k, v in plan["poses_model_deg"].items()}
    _move(impl, arm, plan, offsets, poses[plan["trajectory"][0]], T_min=2.0, hold=plan["settle_s"])
    via = np.vstack([poses[k] for k in plan["trajectory"]])
    t, q = impl.smooth_segments(via, plan["T_seg_s"], plan["rate_hz"], t_hold=plan["t_hold_s"])
    chk = impl.check_plan(t, q, offsets, _soft_limits(plan), plan["rated_speed_deg_s"], plan["speed_fraction"],
                          plan["hw4_margin_deg"])
    if not chk["ok"]:
        raise RuntimeError("trajectory refused by check_plan")
    t0 = arm.time() + 0.02
    log = impl.run_logged(arm, t + t0, q, stop_at=t0 + plan["stop_at_s"])
    log["t"] = log["t"] - t0
    log["plan_check"] = chk
    return log


def measurement_block(impl, arm, plan, offsets, measure):
    """Blocks 2 and 3: accuracy poses (approach from +) and repeated, bidirectional visits to the repeat pose.
    `measure(arm)` returns the measured tool point in mm (emulator: arm.measure_tip; hardware: typed in)."""
    poses = {k: np.asarray(v, float) + offsets for k, v in plan["poses_model_deg"].items()}
    visits = [(k, 1) for k in plan["accuracy_block"]] + [(plan["repeat_pose"], s) for s in plan["repeat_approaches"]]
    rows = []
    for name, s in visits:
        target = poses[name]
        _move(impl, arm, plan, offsets, target + s * plan["approach_deg"], T_min=1.0)
        _move(impl, arm, plan, offsets, target, T_min=plan["approach_T_s"], hold=plan["settle_s"])
        rows.append(dict(pose=name, approach=s, servo=np.round(arm.read(), 2), p_mm=np.asarray(measure(arm), float)))
    return rows


def _gate(impl_name):
    """Hardware gate: the safety-critical tests of the student module must pass before the arm moves."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_lab.py"), "-k",
                        "check_plan or run_logged"], env={**__import__("os").environ, "LAB_IMPL": impl_name},
                       capture_output=True, text=True, cwd=HERE)
    print(r.stdout[-400:])
    return r.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run validation plan VP-1 (Hardware rotation 1).")
    ap.add_argument("--backend", choices=["sim", "vendor"], default="sim")
    ap.add_argument("--arm", default="S1", help="emulated arm id (sim) or the label on the course arm (vendor)")
    ap.add_argument("--port", default=None, help="serial port of the course arm (vendor backend)")
    ap.add_argument("--impl", default="starter", help="module with check_plan / run_logged (default starter)")
    ap.add_argument("--offsets", default=str(HERE / "data" / "servo_offsets_A5.json"))
    ap.add_argument("--plan", default=str(HERE / "data" / "validation_plan.json"))
    ap.add_argument("--team", default="T00")
    ap.add_argument("--operator", default="")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(HERE))
    impl = importlib.import_module(a.impl)
    plan = impl.load_plan(a.plan)
    offsets = impl.load_offsets(a.offsets)
    date = _dt.date.today().strftime("%Y%m%d")
    stem = Path(a.out) / f"log_{a.arm}_{date}_{a.team}_run1"

    if a.backend == "sim":
        from sim_arm import SimArm
        arm = SimArm(a.arm, offsets_deg=offsets).connect()

        def measure(x):
            return x.measure_tip(plan["p_tool_m"])
    else:
        if not _gate(a.impl):
            sys.exit("Safety-critical tests failed: the arm will not be moved. Use the simulation station.")
        print("TA: confirm the signed safety checklist, e-stop tested, keep-out zone clear.")
        if input("Type GO to start: ").strip() != "GO":
            sys.exit("not started")
        from vendor_arm import VendorArm
        arm = VendorArm(a.port, offsets_deg=offsets).connect()

        def measure(x):
            x.stop()                     # latched stop: motion refused while the photos are taken and read
            print("  ARM IN LATCHED STOP. Nobody enters the keep-out zone: take the top and side photos.")
            s = input("  tip x y (top photo) z (side photo) in mm: ").replace(",", " ").split()
            while input("  Keep-out zone clear and reading checked by the TA? type CLEAR: ").strip() != "CLEAR":
                pass
            x.reset()
            return [float(v) for v in s[:3]]

    log = trajectory_run(impl, arm, plan, offsets)
    impl.write_log(str(stem) + ".csv", log)
    print("trajectory logged:", str(stem) + ".csv", "| aborted:", log["aborted"])
    if a.backend == "vendor":
        input("Stop test done. Press Enter when the operator has released the stop to reset the arm: ")
    arm.reset()
    rows = measurement_block(impl, arm, plan, offsets, measure)
    tips = Path(a.out) / f"tips_{a.arm}_{date}_{a.team}.csv"
    with open(tips, "w") as fh:
        fh.write("pose,approach," + ",".join(f"servo_{i + 1}" for i in range(6)) + ",x_mm,y_mm,z_mm\n")
        for r in rows:
            fh.write(",".join([r["pose"], str(r["approach"])] + [f"{x:.2f}" for x in r["servo"]]
                              + [f"{x:.1f}" for x in r["p_mm"]]) + "\n")
    meta = dict(arm=a.arm, backend=a.backend, date=date, team=a.team, operator=a.operator,
                offsets_file=a.offsets, offsets_deg=offsets.tolist(), plan=plan["name"],
                rate_hz=plan["rate_hz"], p_tool_m=plan["p_tool_m"], aborted=log["aborted"], incidents="")
    Path(str(stem) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print("tips written:", tips)
    arm.close()


if __name__ == "__main__":
    main()
