# SBEG 46X project starter code

Python package `sbeg46x_project` with the models, reference simulators, baselines and acceptance-test runs of the six
project scenarios, the hardware-log helpers for the required hardware run, and the test suites. All values are
**illustrative**.

## Install and check (Week 7)

```bash
python -m venv .venv && source .venv/bin/activate      # or use the course Docker image
pip install -r requirements.txt
python -m pytest -q                                    # models, baselines and hardware helpers pass; TEAM TODO tests are skipped
python baseline.py                                     # every baseline through its acceptance tests (about 1 minute)
python baseline.py 3                                   # one scenario
```

Work in your own copy of this folder (your team repository) and run the tests there before you change anything.

## Layout

| File | Content |
|---|---|
| `sbeg46x_project/devices.py` | Hapkit-type device (`HAPKIT` data, saturation at 2 N, watchdog at ±40 mm, quantisation), operator and loss-of-resistance operator, course-arm tip axis `ServoAxis`, velocity estimator, IMU model `imu_from_angle`, single-device simulator `simulate_device` |
| `sbeg46x_project/teleop.py` | Delayed packet network (`DelayLine`, `network_delays`), the baseline `PositionPosition` architecture with the `setpoint` and `reflect` hooks, `WaveVariables` and `TimeDomainPassivity` (wired to **TEAM TODO** `wave_leader`, `wave_follower`, `tdpa_leader`, `tdpa_follower`), bilateral simulator `simulate` |
| `sbeg46x_project/tissue.py` | Layered needle model of Scenario 1 (`NEEDLE_S1`, `NeedleTissue`), force sensor, one-sided virtual wall |
| `sbeg46x_project/arm.py` | Course arm (shared course model, MR Example 4.5 layout): screw axes, FK, IK with a tool transform, point Jacobian, software limits, RCM and clearance geometry, servo model, quintic helpers |
| `sbeg46x_project/registration.py` | Transforms, FLE simulation, FRE and TRE; **TEAM TODO** `pivot_calibration`, `paired_point_registration`, `tre_prediction` |
| `sbeg46x_project/optctrl.py` | `c2d`, `dlqr`, SciPy QP fallback `solve_qp`; **TEAM TODO** `LinearMPC.solve` |
| `sbeg46x_project/estimation.py` | Baseline estimators (backward difference, accelerometer tilt, complementary filter, phase timer, phase portrait); **TEAM TODO** `KalmanFilter` |
| `sbeg46x_project/safety.py` | Baseline limits; **TEAM TODO** `cbf_filter_1d`, `cbf_qp` |
| `sbeg46x_project/tasks.py` | The test conditions, runs and metrics of every acceptance test: `sK_run`, `sK_metrics`, `sK_acceptance(design)` |
| `sbeg46x_project/baselines.py` | The baseline design of each scenario (same interfaces as your design) |
| `sbeg46x_project/hardware.py` | Hardware logs in the SBEG 461 project format (a 461 log reads unchanged) plus `aux_<name>` columns: `write_log`, `load_log`, `resample`, `estimate_delay`, `compare`, `normalised_rms`, `torque_constant`, `rendered_stiffness`, `within_limits`, `stop_time`, and `channel_energy`, `peak_to_peak`, `barrier_violations`, `rest_time`, `points_from_log` |
| `sbeg46x_project/hwtasks.py` | Required hardware run: provisional thresholds `HW[k]`; identification `fit_free_decay`, `fit_servo_step`, `imu_bias_at_rest`; replay simulations `replay_device`, `replay_follower_hapkit`, `replay_servo_axis`, `replay_arm`; reference signals `zero_phase_velocity`, `phase_from_events`, `cadence_from_events`, `effort_reduction`; geometric checks `rcm_errors`, `shaft_clearance`, `tre_at_checks`, `static_force_check` |
| `sbeg46x_project/metrics.py` | RMS, windows, phase wrapping |
| `team_design.py` | **Your** entry point: set `SCENARIO` and return your design from `make_design()` (interfaces in its docstring) |
| `baseline.py` | Runs the baselines through the acceptance tests |
| `tests/test_models.py` | Device, tissue, network, arm, optimisation and estimation checks with hand-computed or textbook answers |
| `tests/test_baselines.py` | Each baseline lands in its documented envelope (most deliberately fail several ATs) |
| `tests/test_hardware.py`, `tests/test_hwtasks.py` | Checks of the hardware helpers on synthetic logs with known answers |
| `tests/test_team_extensions.py` | Harness for your code: one test per ported lab function, then all five ATs of your scenario. Skipped until you port the code and set `SCENARIO` |
| `tests/test_hardware_team.py` | Your hardware acceptance tests on the committed logs in `results/hardware/` (skipped until the first log exists) |

## TEAM TODO stubs and the lab functions they come from

Port your own lab code; the formulas are those of the labs, only the names, argument order and units change as listed.

When each stub becomes available (the order of the revised course; `project_brief.md` §5, "What each milestone may rely on"):

| Stubs | Lab (week taught) | Suggested port by | Already available in the starter |
|---|---|---|---|
| `wave_leader`, `wave_follower`, `tdpa_leader`, `tdpa_follower` | Lab 3 (Week 3; the passivity theorem itself is from SBEG 461 Week 7) | Week 10 proposal (preliminary results) | — |
| `pivot_calibration`, `paired_point_registration`, `tre_prediction` | Lab 5 (Week 5) | Week 10 proposal | — |
| `LinearMPC.solve` | Lab 8 (Week 9; its LQR part is a warm-up on SBEG 461 Week 7) | Week 12 design review | `optctrl.c2d`, `dlqr`, `solve_qp`: LQR can be used from Week 7 |
| `KalmanFilter.predict`, `update` | Lab 9 (Week 10) | Week 12 design review | the baseline estimators of `estimation.py` |
| `cbf_filter_1d`, `cbf_qp` | Lab 10 (Week 11) | Week 13 (the CBF-based ATs may follow the design review) | the baseline limits of `safety.py` |

No stub needs the Week 12 material (Lab 11, visual servoing); a camera loop is an extension in your own module.

| TEAM TODO stub (project, SI units) | Your lab function | What changes |
|---|---|---|
| `teleop.wave_leader(xdot_m, v_m, b)` → `(F_m, u_m)` | Lab 3 `wave_leader(xdot_m, v_m, b)` | Nothing: same signature and Lab 3 waves $u = (b\dot x + F)/\sqrt{2b}$, $v = (b\dot x - F)/\sqrt{2b}$; $F_m = b\dot x_m - \sqrt{2b}\,v_m$, $u_m = \sqrt{2b}\,\dot x_m - v_m$ |
| `teleop.wave_follower(u_s, F_c, b)` → `(xdot_d, v_s)` | Lab 3 `wave_follower(u_s, F_c, b)` | Nothing: $\dot x_d = (\sqrt{2b}\,u_s - F_c)/b$, $v_s = \sqrt{2b}\,\dot x_d - u_s$ |
| `teleop.tdpa_leader`, `teleop.tdpa_follower` | Lab 3 `tdpa_leader`, `tdpa_follower` | Nothing (same arguments and returns) |
| `registration.pivot_calibration(R_list, p_list)` → `(t, p_pivot, rms)` | Lab 5 `pivot_calibration(Rs, ps)` | mm → m |
| `registration.paired_point_registration(P_b, P_a)` → `T_ab` | Lab 5 `register_points(X, Y)` with `X = P_b`, `Y = P_a` | Name; mm → m (a rigid transform is unit-free apart from $p$) |
| `registration.tre_prediction(P_b, p_target_b, fle_rms)` | Lab 5 `tre_predict(X, target, fle_rms)` | Name; mm → m (`P_b`, `p_target_b`, `fle_rms` and the result in metres) |
| `estimation.KalmanFilter.predict(u)` | Lab 9 `kf_predict(x, P, A, Q, B, u)` | Method: reads and updates `self.x`, `self.P` with `self.A`, `self.B`, `self.Qw` |
| `estimation.KalmanFilter.update(y, C, Rv)` | Lab 9 `kf_update(x, P, y, C, R)` | Method: updates `self.x`, `self.P`; `C`, `Rv` default to `self.C`, `self.Rv` |
| `optctrl.LinearMPC.solve(x0, x_ref)` | Lab 8 `mpc_qp` (condensed QP) + `mpc_control` (first input) | Method: uses `self.Ad`, `self.Bd`, `self.Q`, `self.R`, `self.P`, `self.N`, the input limits and the optional `G x <= g`; tracks `x_ref`; calls `optctrl.solve_qp`, the SLSQP solver `solve_qp(H, f, A_ub, b_ub, lb, ub, x0)` returning `(x, info)` with `info = dict(success, fun, nit)` (not the Lab 8 solver returning `(z, 'optimal'|'infeasible')`), so replace the Lab 8 status check by a check on `info['success']` |
| `safety.cbf_filter_1d`, `safety.cbf_qp` | Lab 10 `cbf_filter_1d`, `cbf_qp` | Nothing (course-project signatures) |

## Conventions

* SI units. Haptic position $x$ positive away from the operator; forces positive in $+x$. Teleoperation sign conventions of
  Lab 3: $F_m$ is the force of the teleoperator on the leader ($F_m = -a_m$ in the Week 2 two-port notation), the channel energy is $E_{ch} = h\sum(F_m v_m - F_c\dot x_d)$.
* Course arm in *Modern Robotics* form: space screw axes, $T = e^{[\mathcal S_1]\theta_1}\cdots e^{[\mathcal S_6]\theta_6}M$, tool
  frame {b} at the tip with $\hat z_b$ along the shaft.
* A device controller is a closure called once per 1 ms tick; it must be causal (no future samples) and keep its own state.

## Hardware logs (required hardware run)

Log every supervised session to `results/hardware/<date>_<session>_<test>.csv` with the header
`t, cmd_1..cmd_n, meas_1..meas_n[, cur_1..cur_n][, force][, aux_<name> ...]` (SI units; the column use of each scenario is in
`scenario_sheets.md`). Then compare with the simulation of the same command:

```python
from sbeg46x_project import devices, hardware, hwtasks
log = hardware.load_log("results/hardware/2026-11-25_session2_HW1-2.csv")
ident = hwtasks.fit_free_decay(t_decay, x_decay, K=100.0)             # your unit, session 1 (K = 100 N/m)
sim = hwtasks.replay_follower_hapkit(log["t"], log["aux"]["r"], 200.0, 2.0,
                                     dev=devices.Hapkit(m_h=ident["m_h"], b_h=ident["b_h"]),
                                     f_env_log=log["aux"]["f_env"])
out = hardware.compare(log, sim, skip=1.0)
print(out["rms_sim_meas"], out["delay"])
```

Write your HW*k*-1 … HW*k*-4 tests in `tests/test_hardware_team.py` so that they read the committed logs. Never edit a log by
hand: the raw CSV is the evidence that the session took place.

## Rules

Extend the package; do not edit `tasks.py`, the baselines or `tests/test_baselines.py` (the markers run the original ones).
Port your own lab code into the TEAM TODO stubs. Disclose AI use in the report appendix (`project_brief.md` §8).
