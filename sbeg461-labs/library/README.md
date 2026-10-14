# sbeg461_dynamics — your cumulative SBEG 461 Robot Kinematics and Dynamics library

Each lab adds tested functions to this package; later labs, the project and the next course reuse it
(install it in editable mode: `python -m pip install -e .`). Copy each function into `sbeg461_dynamics/` once its lab
tests pass, keep the same names and signatures, and add the lab's tests to `tests/`.

| Week | Lab | Functions you write (from the lab starter) |
|---|---|---|
| 1 | Lab 1 — Lagrangian dynamics in SymPy vs. modern_robotics | `load_params`, `make_symbols`, `time_derivative`, `euler_lagrange`, `planar_chain_energy`, `mass_matrix`, `gravity_vector`, `christoffel`, `coriolis_matrix`, `numeric_model`, `mr_chain`, `mr_terms` … |
| 2 | Lab 2 — Implement the RNEA; verify against modern_robotics and Pinocchio | `load_robot`, `load_states`, `load_trajectory`, `spatial_inertia`, `transform_spatial_inertia`, `rigid_body_wrench`, `link_screw_axes`, `forward_iterations`, `backward_iterations`, `rnea`, `dynamics_terms`, `tip_wrench_torques` … |
| 3 | Lab 3 — Forward-dynamics simulator and integrator comparison (Formal report 1) | `load_robot`, `mass_matrix_rnea`, `link_body_jacobians`, `mass_matrix_closed_form`, `forward_dynamics`, `com_positions`, `energy`, `euler_step`, `semi_implicit_euler_step`, `rk4_step`, `simulate`, `observed_order` |
| 4 | Lab 4 — URDF dynamics, C++ RNEA vs Python timing, and the constrained 2R arm | `skew`, `rpy_to_R`, `RpToTrans`, `TransInv`, `Adjoint`, `ad`, `MatrixExp6`, `read_urdf_chain`, `export_for_cpp`, `load_states`, `urdf_to_mr`, `rnea` … |
| 5 | Lab 5 — Joint PID in simulation; a ROS 2 controller node | `load_plant`, `plant_gravity`, `plant_rhs`, `rk4_step`, `time_grid`, `step_reference`, `ramp_reference`, `load_knee_swing`, `second_order_params`, `char_roots`, `is_stable`, `unit_error_response` … |
| 6 | Lab 6 — Computed torque vs PID with model error on the course arm | `load_config`, `arm_model`, `make_case`, `tau_limits`, `time_grid`, `quintic_reference`, `cartesian_reference`, `rk4_step`, `simulate_arm`, `ct_gains`, `ct_ki_max`, `decentralized_gains` … |
| 7 | Lab 7 — Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm | `load_config`, `plant_2r`, `plant_course_arm`, `scaled_model`, `mass_matrix`, `coriolis_vector`, `gravity_torque`, `potential_energy`, `forward_dynamics`, `pd_gains_from_inertia`, `is_hurwitz`, `time_grid` … |
| 8 | Hardware rotation 1 — Joint control on the course arm: PID step response, gravity compensation, logging; safety induction | `course_arm_model`, `inverse_dynamics`, `gravity_torques`, `mass_diagonal`, `load_task`, `write_log`, `load_log`, `log_window`, `step_command`, `smooth_move`, `check_command`, `tracking_metrics` … |
| 9 | Lab 8 — Impedance control on a Hapkit-type haptic device and a simulated rehabilitation robot (Formal report 2) | `load_device`, `load_free_decay`, `rk4_step`, `rehab_fk`, `rehab_jacobian`, `rehab_jacobian_dot`, `rehab_ik`, `rehab_dynamics`, `simulate_rehab`, `handle_mass`, `gravity_stiffness`, `handle_force_to_current` … |
| 10 | Lab 9 — Trajectory library, torque feasibility and time-optimal time scaling (TOPP) on the course arm | `skew`, `TransInv`, `Adjoint`, `ad`, `MatrixExp6`, `rnea`, `load_course_arm`, `course_arm_model`, `load_tasks`, `load_topp_task`, `export_csv`, `write_log` … |
| 11 | Lab 10 — Grid planner for a 2R arm among obstacles | `load_scene`, `wrap_angle`, `fk_2r`, `point_segment_distance`, `capsule_circle_distance`, `clearance`, `config_free`, `grid_angles`, `cell_to_q`, `q_to_cell`, `cspace_grid`, `joint_step` … |
| 12 | Lab 11 — RRT for the course arm; path smoothing and time scaling (Formal report 3) | `load_course_arm`, `load_scene`, `exp6_batch`, `poe_frames`, `trapezoid_params`, `trapezoid_time_scaling`, `trapezoid_from_limits`, `rnea`, `link_points`, `segment_point_distance`, `clearance`, `within_limits` … |
