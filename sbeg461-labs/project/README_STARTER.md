# SBEG 461 project starter code

Python package `sbeg461_project` with the six scenario robots in *Modern Robotics* form. It also has a
reference simulator, the baselines, collision checking and planning, and three test suites. All values
are **illustrative**.

## Install and check (Week 7)

```bash
python -m venv .venv && source .venv/bin/activate      # or use the course Docker image
pip install -r requirements.txt
pytest                                                # model, baseline and hardware-helper tests must pass; 4 team tests are skipped
python baseline.py                                    # prints every baseline metric (about 1-2 minutes)
python print_models.py 3                              # Markdown tables of one scenario model
```

On the course machines, use `~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python`.
Pinocchio (`pin`) is optional. Without it, the simulator uses `modern_robotics`, which is much slower
for the 6-DOF scenarios, and the Pinocchio cross-checks are skipped.

## Layout

| File | Content |
|---|---|
| `sbeg461_project/models.py` | `load_scenario(k)` → `Scenario` with `M, Slist, Mlist, Glist, g`, limits, obstacles, `params`; `with_payload` (MR Eq. 8.42), `scaled` (model error) |
| `sbeg461_project/dyn.py` | `mass_matrix`, `gravity`, `inverse_dynamics`, `forward_dynamics` (MR Eq. 8.81), `energy`, `ftip_from_force`, and `simulate` (RK4 or Euler, torque saturation, delay, external force, time-varying gravity; engine `mr` or `pinocchio`) |
| `sbeg461_project/pin_bridge.py` | Builds a Pinocchio model from the PoE description (cross-check and fast ABA) |
| `sbeg461_project/traj.py` | Quintic time scaling, `gait_reference`, `squat_reference`, `torque_profile`, `check_limits`, stop-and-go baseline; **TODO** `time_optimal_scaling` (MR §9.4, Week 10) |
| `sbeg461_project/planning.py` | Capsule collision checking (Week 11), `rrt` (MR Algorithm 10.3), `rrt_bidirectional`, `shortcut` (MR §10.8), `path_clearance` (Week 12). Given complete, so Scenarios 3 and 5 can plan before Week 12 |
| `sbeg461_project/control.py` | Baselines `pd_gravity` (MR Eq. 11.38), `task_stiffness`, `tunnel_stiffness`; **TODO** `computed_torque` (11.37) and `task_space_computed_torque` (11.47), both Week 6; `impedance` (11.65) and `admittance` (11.66), both Week 9 |
| `sbeg461_project/tasks.py` | Patient model (S1), RCM pose and error (S3), insertion reference (S3), deck gravity (S5), contact wall (S6), IK helpers |
| `sbeg461_project/metrics.py` | RMS and maximum error, peak torque, saturation fraction, effort |
| `sbeg461_project/hardware.py` | Required hardware run: `load_log`/`write_log` (CSV of commanded vs measured, optional current and force), `resample`, `estimate_delay`, `compare` (commanded vs measured and simulation vs measured RMS/max error), `normalised_rms`, `torque_constant`, `rendered_stiffness`, `within_limits`, `stop_time` |
| `baseline.py` | Runs each scenario's baseline (`python baseline.py 2 --plot`) |
| `print_models.py` | Prints the model tables used in `scenario_sheets.md` |
| `tests/test_models.py` | Home configuration, inertias, $M$ symmetric positive definite, Christoffel $C$ with $\dot M-2C$ skew, hand-computed gravity torques, payload, wrench sign, mr = Pinocchio, energy conservation, references, stored IK solutions |
| `tests/test_baselines.py` | Each baseline lands in its documented envelope (several deliberately fail the ATs) |
| `tests/test_hardware.py` | Checks of the hardware helpers on synthetic logs with known answers (offset, delay, quantisation, stiffness fit, stop time) |
| `tests/test_team_extensions.py` | Acceptance-test harness for your code. Skipped until you implement the TODO stubs |

## When you need each TODO

The stubs follow the course order. Write the controllers first: `computed_torque` and
`task_space_computed_torque` after Week 6, `impedance` (and the optional `admittance`) after Week 9, so that a
first controller runs by the proposal (Week 10). Write `time_optimal_scaling` after Week 10, before the design
review (Week 12). The planner is given; tune and justify it after Weeks 11–12.

## Conventions

* Twists $(\omega, v)$ and wrenches $(m, f)$ with the angular part first. Space screw axes in {s}. The
  PoE is $T=e^{[\mathcal S_1]\theta_1}\cdots M$.
* `Ftip` is the wrench the robot applies **to** the environment, in {b}, as in `modern_robotics`. Use
  `dyn.ftip_from_force` for a force that the environment applies **to** the robot.
* A controller is a closure `ctrl(t, q, qd) -> tau`. The controller's model may differ from the
  simulated plant: use `Scenario.scaled` or `with_payload` to create the model error.

## Hardware logs (required hardware run)

Log every supervised hardware session to `results/hardware/<date>_<session>_<test>.csv` with the header
`t, cmd_1..cmd_n, meas_1..meas_n[, cur_1..cur_n][, force]` (SI units). Then compare with the simulation of
the same commanded signal:

```python
from sbeg461_project import hardware
log = hardware.load_log("results/hardware/2026-11-18_session1_HW2-1.csv")
out = hardware.compare(log, sim, skip=0.6)    # sim = dyn.simulate(...) of the hardware model
print(out["rms_cmd_meas"], out["rms_sim_meas"], out["delay"])
```

Add your hardware acceptance tests HWk-1 ... HWk-4 (`scenario_sheets.md`) to `tests/` so that they read the
committed logs. Never edit a log by hand: the raw CSV is the evidence that the session took place.

## Rules

Extend the package; do not edit the baselines or the tests in `tests/test_baselines.py`. Add your
acceptance tests to `tests/`. Disclose AI use in the report appendix (see `project_brief.md` §8).
