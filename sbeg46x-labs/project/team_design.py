"""YOUR TEAM'S DESIGN -- the entry point of the acceptance-test harness (tests/test_team_extensions.py).

1. Set SCENARIO to your scenario number (1-6).
2. Implement make_design() so that it returns your design object with the interface of your scenario
   (below). Reuse your sbeg46x_medrob library from the labs and the starter modules.
3. Run `python -m pytest -q`. The harness runs every simulation acceptance test of your scenario with
   tasks.sK_acceptance(make_design()) and reports each AT that fails, with the measured values.

Interfaces (the baselines in sbeg46x_project/baselines.py implement the same ones):

  Scenario 1  make_design() returns a FACTORY: a function with no argument that returns a fresh
              teleoperation architecture object (subclass teleop.PositionPosition, WaveVariables or
              TimeDomainPassivity; override setpoint/follower/leader as needed).
  Scenario 2  object with calibrate_tool(R_list, p_list) -> T_ft (4x4), register(P_p, P_s) -> T_sp,
              tre_prediction(P_p, p_target, fle_rms) -> float, plan_approach(q_start, q_goal, v_max,
              a_max) -> (t, Q), insertion_rate(t, q, ctx) -> joint rates (6,).
  Scenario 3  object with controller(case) -> ctrl(t, x_meas, sens) -> motor force, where ctrl.log is a
              dict of lists 't', 'v_hat', 'F_u_hat' appended at every call.
  Scenario 4  object with register(P_model, P_s) -> T, architecture(x_lim) -> teleoperation architecture
              for the course-arm follower (x_lim = boundary in follower coordinates).
  Scenario 5  as Scenario 4, with architecture(x_lim, fixture) and insertion_rate(t, q, ctx) as in S2.
  Scenario 6  object with controller(imu) -> ctrl(t, x_meas, sea) -> SEA motor command, where ctrl.log
              is a dict of lists 't', 'phase', 'theta', 'F_d' appended at every call.
"""
SCENARIO = None          # e.g. 3


def make_design():
    raise NotImplementedError("team_design.make_design: return your design (see the docstring)")
