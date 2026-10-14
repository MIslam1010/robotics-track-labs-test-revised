# sbeg361_kinematics — your cumulative SBEG 361 Introduction to Robotics library

Each lab adds tested functions to this package; later labs, the project and the next course reuse it
(install it in editable mode: `python -m pip install -e .`). Copy each function into `sbeg361_kinematics/` once its lab
tests pass, keep the same names and signatures, and add the lab's tests to `tests/`.

| Week | Lab | Functions you write (from the lab starter) |
|---|---|---|
| 1 | Lab 0 — Environment, Git, modern_robotics and MR Studio | `env_report`, `joint_freedoms`, `joint_constraints`, `grubler`, `dof_table`, `wrap_to_pi`, `torus_distance`, `cspace_topology`, `latlong_to_xyz`, `xyz_to_latlong`, `longitude_rate`, `fk_2r_trig` … |
| 2 | Lab 1 — Rotations and transformations: SO(3)/SE(3) property tests and transform trees | `rot_x`, `rot_y`, `rot_z`, `rot_axis_angle`, `rp_to_trans`, `trans_to_rp`, `so3_errors`, `is_rotation`, `rot_inv`, `project_to_so3`, `is_se3`, `trans_inv` … |
| 3 | Lab 2 — Angular velocities and twists: space/body twists, the adjoint, and ROS 2 tf2 visualisation | `vec_to_so3`, `so3_to_vec`, `rp_to_trans`, `trans_to_rp`, `trans_inv`, `is_se3`, `transform_point`, `quat_xyzw_to_rot`, `transform_from_tf`, `load_tree_json`, `load_track_csv`, `axis_angle_between` … |
| 4 | Lab 3 — Exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation | `rp_to_trans`, `trans_to_rp`, `trans_inv`, `vec_to_se3`, `se3_to_vec`, `twist_to_screw`, `load_pose_track`, `vec_to_so3`, `so3_to_vec`, `axis_ang3`, `matrix_exp3`, `g_matrix` … |
| 5 | Lab 4 — Force–torque sensor frame change for a surgical tool and an EOD gripper (Formal report 1) | `vec_to_so3`, `so3_to_vec`, `matrix_exp3`, `matrix_log3`, `rp_to_trans`, `trans_to_rp`, `trans_inv`, `vec_to_se3`, `se3_to_vec`, `adjoint`, `matrix_exp6`, `matrix_log6` … |
| 6 | Lab 5 — Forward kinematics of the course arm: PoE, URDF, simulation and hardware | `vec_to_so3`, `vec_to_se3`, `trans_inv`, `adjoint`, `rot_axis`, `screw_axis`, `screw_exp`, `fk_space`, `fk_body`, `body_screws`, `course_arm_screws`, `read_urdf_joints` … |
| 7 | Lab 6 — DH models in the Robotics Toolbox vs PoE; C++/Eigen FK | `screw_exp`, `fk_space`, `trans`, `rotz_h`, `rotx_h`, `dh_std`, `dh_mod`, `link_transforms`, `fk_dh`, `link_frames`, `course_arm_dh`, `course_arm_mdh` … |
| 8 | Hardware Rotation 1 — FK Validation on the Course Arm | `vec_to_so3`, `screw_exp`, `fk_space`, `tool_points`, `servo_to_model`, `model_to_servo`, `load_offsets`, `load_plan`, `smooth_segments`, `write_log`, `read_log`, `read_tips` … |
| 9 | Lab 7 — Jacobian implementation and finite-difference check (Formal report 2) | `vec_to_so3`, `so3_to_vec`, `vec_to_se3`, `se3_to_vec`, `trans_inv`, `adjoint`, `matrix_exp3`, `matrix_log3`, `matrix_exp6`, `matrix_log6`, `fk_space`, `fk_body` … |
| 10 | Lab 8 — Manipulability maps of an assistive arm; workspace design | `vec_to_so3`, `vec_to_se3`, `trans_inv`, `adjoint`, `matrix_exp6`, `fk_space`, `body_screws`, `jacobian_space`, `jacobian_body`, `jacobian_2r`, `ik_2r_given`, `within_limits` … |
| 11 | Lab 9 — Analytic and numerical IK of the course arm; all-solutions search | `vec_to_so3`, `so3_to_vec`, `vec_to_se3`, `se3_to_vec`, `trans_inv`, `adjoint`, `rot`, `matrix_exp3`, `matrix_log3`, `matrix_exp6`, `matrix_log6`, `fk_space` … |
| 12 | Lab 10 — Resolved-rate control and IK near singularities (Formal report 3) | `vec_to_so3`, `so3_to_vec`, `vec_to_se3`, `se3_to_vec`, `trans_inv`, `adjoint`, `matrix_exp3`, `matrix_exp6`, `matrix_log3`, `matrix_log6`, `fk_space`, `body_screws` … |
