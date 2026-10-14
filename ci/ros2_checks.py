#!/usr/bin/env python3
"""ROS 2 parts of the labs, run inside the course ROS 2 container (ROS 2 Jazzy sourced), as a student would.

    python3 ci/ros2_checks.py 361|461|46X

361 Lab 2 Part B : Python + C++ (colcon) tf2 broadcasters; tf2 values vs nav_tree.json (1e-6), tf2_echo,
                   view_frames, the B6 static_transform_publisher command, a B7-style listener.
461 Lab 5 Ckpt 5 : simulator node + Python PID node (reference solution) and the C++ PID node (cmake build);
                   /joint_states rate, /pid_debug CSV, run-time 'ros2 param set'; untouched starter fails clearly.
46X Lab 11       : the optional Nav2 demo (tb3_simulation_launch.py, headless): bring-up, initial pose, AMCL, a goal.
Prints check names and numbers only (public logs: never solution code).
"""
import glob, json, math, os, shutil, signal, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
course = sys.argv[1]
results = []
procs = []


def check(name, ok, info=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{info}]" if info else ""), flush=True)


def start(cmd, cwd=None, env=None, log=None):
    f = open(log, "w") if log else subprocess.DEVNULL
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ, **(env or {})), stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True, shell=isinstance(cmd, str), executable="/bin/bash" if isinstance(cmd, str) else None)
    procs.append(p)
    return p


def stop_all():
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGINT)
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
    procs.clear()
    time.sleep(1)


def run(cmd, cwd=None, timeout=60, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=True, executable="/bin/bash", capture_output=True, text=True,
                           timeout=timeout, env=dict(os.environ, **(env or {})))
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, out


def lab_dir(repo, prefix):
    d = sorted(glob.glob(os.path.join(ROOT, repo, "labs", prefix + "*")))
    assert d, f"no lab {prefix} in {repo}"
    return d[0]


def solution_dir(repo, lab):
    return os.path.join(ROOT, "solutions", repo, os.path.relpath(lab, os.path.join(ROOT, repo)))


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_of(e):
    T = np.eye(4); T[:3, :3] = quat_to_R(*e["rotation_xyzw"]); T[:3, 3] = e["translation"]; return T


def tf_samples(target, source, duration=0.0, timeout=10.0):
    """B7-style listener: tf2 Buffer + TransformListener at ~5 Hz; returns a list of 4x4 T_target_source."""
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node("sbeg_ci_listener")
    buf = Buffer(); TransformListener(buf, node)
    out, t0, last = [], time.time(), 0.0
    while time.time() - t0 < max(timeout, duration):
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.time() - last < 0.2: continue
        try:
            tr = buf.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.05)).transform
            T = np.eye(4); T[:3, :3] = quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
            T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
            out.append(T); last = time.time()
            if not duration: break
        except Exception:
            pass
        if out and time.time() - t0 > duration: break
    node.destroy_node(); rclpy.shutdown()
    return out


def tf_lookup(target, source, timeout=10.0):
    s = tf_samples(target, source, 0.0, timeout)
    return s[0] if s else None


def tf_checks(tag, lab):
    tree = json.load(open(os.path.join(lab, "nav_tree.json")))
    E = {(e["parent"], e["child"]): T_of(e) for e in tree["transforms"]}
    expect = E[("patient_ref", "ct_image")] @ E[("ct_image", "plan_entry")]
    rc, out = run("timeout 8 ros2 topic echo /tf_static --once", timeout=15)
    check(f"{tag}: /tf_static published", rc == 0 and "child_frame_id" in out)
    rc, out = run("timeout 8 ros2 topic hz /tf", timeout=15)
    hz = [float(l.split(":")[1]) for l in out.splitlines() if "average rate" in l]
    check(f"{tag}: /tool marker on /tf at ~50 Hz", hz and 25 < hz[-1] < 75, f"{hz[-1]:.1f} Hz" if hz else "no rate")
    T = tf_lookup("patient_ref", "plan_entry")
    err = float(np.abs(T - expect).max()) if T is not None else float("nan")
    check(f"{tag}: lookup patient_ref<-plan_entry equals nav_tree.json (1e-6)", T is not None and err < 1e-6, f"max err {err:.2e}")
    S = tf_samples("plan_entry", "tool_tip", duration=8.0)
    z = np.array([T[2, 3] for T in S]); y = np.array([T[1, 3] for T in S])
    check(f"{tag}: B4 dynamic tool tip: z within 0-5 mm, y within 0.826-1.000 mm (handout)",
          len(S) > 10 and z.min() > -1e-6 and z.max() < 0.005 + 1e-6 and y.min() > 0.000826 - 1e-6 and y.max() < 0.00100 + 1e-6,
          f"{len(S)} samples, z {1e3 * z.min():.2f}..{1e3 * z.max():.2f} mm, y {1e3 * y.min():.3f}..{1e3 * y.max():.3f} mm" if len(S) else "none")
    rc, out = run("timeout 6 ros2 run tf2_ros tf2_echo patient_ref plan_entry", timeout=15)
    check(f"{tag}: handout command 'tf2_echo patient_ref plan_entry' prints a transform", "Translation" in out)


def course_361():
    lab = lab_dir("sbeg361-labs", "lab-02")
    # Part B in Python (the provided node needs only numpy and rclpy)
    start(["python3", "ros2/nav_tf_broadcaster.py"], cwd=lab, log="/tmp/b_py.log")
    time.sleep(4)
    tf_checks("361 Lab 2 Python node", lab)
    tmp = tempfile.mkdtemp()
    rc, out = run("timeout 25 ros2 run tf2_tools view_frames", cwd=tmp, timeout=40)
    pdfs = glob.glob(os.path.join(tmp, "frames*.pdf"))
    check("361 Lab 2 B5: 'ros2 run tf2_tools view_frames' writes the tree PDF", pdfs, os.path.basename(pdfs[0]) if pdfs else out[-200:])
    # B6: the handout's static_transform_publisher command, then lookup(ct_image, guide_tip)
    start("ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z -0.05 --qx 0 --qy 0 --qz 0 --qw 1 "
          "--frame-id plan_entry --child-frame-id guide_tip", log="/tmp/b6.log")
    time.sleep(3)
    tree = json.load(open(os.path.join(lab, "nav_tree.json")))
    E = {(e["parent"], e["child"]): T_of(e) for e in tree["transforms"]}
    G = np.eye(4); G[2, 3] = -0.05
    T = tf_lookup("ct_image", "guide_tip")
    err = float(np.abs(T - E[("ct_image", "plan_entry")] @ G).max()) if T is not None else float("nan")
    check("361 Lab 2 B6: handout static_transform_publisher; lookup ct_image<-guide_tip correct", T is not None and err < 1e-6,
          f"max err {err:.1e}")
    stop_all()
    # Part B in C++: ament_cmake package built with colcon, exactly as in the handout
    ws = tempfile.mkdtemp()
    os.makedirs(os.path.join(ws, "src"))
    os.symlink(os.path.join(lab, "ros2", "cpp"), os.path.join(ws, "src", "sbeg361_nav_tf"))
    rc, out = run("colcon build --packages-select sbeg361_nav_tf", cwd=ws, timeout=900)
    ok = rc == 0 and os.path.isdir(os.path.join(ws, "install", "sbeg361_nav_tf"))
    check("361 Lab 2 C++: colcon build sbeg361_nav_tf (handout commands)", ok)
    if not ok:  # provided course code (not a solution): safe to show
        for f in glob.glob(os.path.join(ws, "log", "latest_build", "sbeg361_nav_tf", "std*.log")):
            print("".join(open(f, errors="replace").readlines()[-25:]))
    # the same package without the system JSON library: falls back to cpp_support/third_party
    ws2 = tempfile.mkdtemp(); os.makedirs(os.path.join(ws2, "src"))
    os.symlink(os.path.join(lab, "ros2", "cpp"), os.path.join(ws2, "src", "sbeg361_nav_tf"))
    rc2, out2 = run("colcon build --packages-select sbeg361_nav_tf --cmake-args -DCMAKE_DISABLE_FIND_PACKAGE_nlohmann_json=TRUE",
                    cwd=ws2, timeout=900)
    log2 = "".join(open(f, errors="replace").read() for f in glob.glob(os.path.join(ws2, "log", "latest_build", "sbeg361_nav_tf", "stdout*.log")))
    check("361 Lab 2 C++: builds without nlohmann-json3-dev (copy in cpp_support)", rc2 == 0 and "using the copy in" in log2)
    start(f"source {ws}/install/setup.bash && ros2 run sbeg361_nav_tf nav_tf_broadcaster --ros-args -p lab_dir:='{lab}'",
          log="/tmp/b_cpp.log")
    time.sleep(4)
    tf_checks("361 Lab 2 C++ node", lab)
    stop_all()


def pid_run(tag, lab, pid_cmd, env=None):
    start(["python3", "ros2/joint_sim_node.py", "--ros-args", "-p", "plant:=arm_j2", "-p", "dt:=0.005"], cwd=lab,
          env=env, log=f"/tmp/sim_{tag}.log")
    time.sleep(3)
    start(pid_cmd, cwd=lab, env=env, log=f"/tmp/pid_{tag}.log")
    time.sleep(3)
    rc, out = run("timeout 8 ros2 topic hz /joint_states", timeout=15)
    hz = [float(l.split(":")[1]) for l in out.splitlines() if "average rate" in l]
    check(f"{tag}: /joint_states published (dt = 0.005 s -> 200 Hz target)", hz and hz[-1] > 50, f"{hz[-1]:.0f} Hz" if hz else "no rate")
    rows = []
    for _ in range(3):  # the ros2 CLI can need a few seconds to discover the topic: listen longer and retry
        rc, out = run("timeout 8 ros2 topic echo /pid_debug --csv", timeout=15)
        rows = [l.split(",") for l in out.splitlines() if l.count(",") >= 8]
        if len(rows) > 100: break
    vals = np.array([[float(x) for x in r[-9:]] for r in rows]) if rows else np.zeros((0, 9))
    check(f"{tag}: /pid_debug CSV rows [t,qd,q,e,tau,P,I,D,sat]", len(vals) > 100 and np.isfinite(vals).all(), f"{len(vals)} rows")
    e_before = float(np.abs(vals[-50:, 3]).mean()) if len(vals) >= 50 else float("nan")
    out = ""
    for _ in range(4):  # the ros2 CLI can need a few seconds to discover the node: retry
        rc, out = run("ros2 param set /joint_pid ki 38.5", timeout=30)
        if "successful" in out.lower(): break
        time.sleep(3)
    check(f"{tag}: run-time gain change 'ros2 param set /joint_pid ki 38.5'", "successful" in out.lower(), out.strip()[-80:])
    time.sleep(6)
    rc, out = run("timeout 3 ros2 topic echo /pid_debug --csv", timeout=10)
    rows = [l.split(",") for l in out.splitlines() if l.count(",") >= 8]
    vals2 = np.array([[float(x) for x in r[-9:]] for r in rows]) if rows else np.zeros((0, 9))
    e_after = float(np.abs(vals2[-50:, 3]).mean()) if len(vals2) >= 50 else float("nan")
    check(f"{tag}: integral action removes the steady-state sag", e_after < e_before or e_after < 1e-3,
          f"|e| {e_before:.2e} -> {e_after:.2e} rad")
    stop_all()


def course_461():
    lab = lab_dir("sbeg461-labs", "lab-05")
    sol = solution_dir("sbeg461-labs", lab)
    # untouched starter: the node must stop with a clear message, not hang or publish nonsense
    start(["python3", "ros2/joint_sim_node.py", "--ros-args", "-p", "plant:=arm_j2", "-p", "dt:=0.005"], cwd=lab, log="/tmp/sim0.log")
    time.sleep(3)
    p = start("LAB_IMPL=starter python3 ros2/joint_pid_node.py --ros-args -p plant:=arm_j2 -p wn:=20.0", cwd=lab, log="/tmp/pid0.log")
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < 20: time.sleep(0.5)
    log0 = open("/tmp/pid0.log").read()
    check("461 Lab 5: untouched starter stops with NotImplementedError (no hang)", p.poll() is not None and "NotImplementedError" in log0)
    stop_all()
    # reference solution (stands in for a finished starter), Python node
    work = tempfile.mkdtemp(); shutil.copytree(lab, work, dirs_exist_ok=True)
    for f in glob.glob(os.path.join(sol, "*.py")): shutil.copy(f, work)
    pid_run("461 Lab 5 Python node", work,
            "LAB_IMPL=solution python3 ros2/joint_pid_node.py --ros-args -p plant:=arm_j2 -p wn:=20.0 "
            "-p reference:=step -p q_target:=-0.3 -p t_step:=1.0 -p dt:=0.005", env={"LAB_IMPL": "solution"})
    # C++ node: cmake build linked with the C++ reference solution, simulator stays Python (untouched starter.py)
    shutil.copy(os.path.join(sol, "cpp", "solution.cpp"), os.path.join(work, "cpp", "solution.cpp"))
    # the copy lives outside the repository, so point the build at cpp_support (documented -D / environment option)
    cenv = {"SBEG_CPP_SUPPORT": os.path.join(ROOT, "sbeg461-labs", "cpp_support")}
    rc, out = run("cmake -S cpp/ros2 -B build_ros -DLAB_IMPL=solution && cmake --build build_ros", cwd=work, timeout=900, env=cenv)
    ok = os.path.exists(os.path.join(work, "build_ros", "joint_pid_node"))
    check("461 Lab 5 C++: 'cmake -S cpp/ros2 -B build_ros && cmake --build build_ros'", ok)
    if not ok:  # CMake messages only: compiler output could quote the reference solution
        print("\n".join(l for l in out.splitlines() if l.startswith(("CMake Error", "  ", "-- Could")) and "solution" not in l)[-2500:])
    # the same build from inside the repository, exactly as a student does it (cpp_support found automatically)
    rc3, out3 = run("cmake -S cpp/ros2 -B /tmp/build_ros_student", cwd=lab, timeout=300)
    check("461 Lab 5 C++: configure finds cpp_support from the Lab folder (student layout)", rc3 == 0,
          "" if rc3 == 0 else [l for l in out3.splitlines() if "Error" in l][:1])
    if ok:
        for f in glob.glob(os.path.join(work, "solution*.py")): os.remove(f)   # simulator must run on the untouched starter.py
        pid_run("461 Lab 5 C++ node", work,
                "./build_ros/joint_pid_node --ros-args -p plant:=arm_j2 -p wn:=20.0 -p reference:=step "
                "-p q_target:=-0.3 -p t_step:=1.0 -p dt:=0.005")


def course_46X():
    rc, out = run("ros2 pkg prefix nav2_bringup && ros2 pkg prefix nav2_minimal_tb3_sim", timeout=30)
    check("46X Lab 11: Nav2 + TurtleBot3 simulation packages installed in the course container", rc == 0, out.strip().replace("\n", " "))
    launch = os.path.exists("/opt/ros/jazzy/share/nav2_bringup/launch/tb3_simulation_launch.py")
    check("46X Lab 11: handout launch file nav2_bringup/tb3_simulation_launch.py exists", launch)
    if not (rc == 0 and launch):
        return
    start("ros2 launch nav2_bringup tb3_simulation_launch.py headless:=True use_rviz:=False", log="/tmp/nav2.log")
    t0 = time.time(); topics = ""
    while time.time() - t0 < 240:
        rc, topics = run("ros2 topic list", timeout=20)
        if "/odom" in topics and "/map" in topics and "/amcl_pose" in topics: break
        time.sleep(5)
    check("46X Lab 11: Nav2 + Gazebo bring-up (/odom, /map, /amcl_pose)", "/odom" in topics and "/map" in topics,
          f"{time.time() - t0:.0f} s")
    rc, out = run("timeout 10 ros2 topic echo /odom --once", timeout=20)
    check("46X Lab 11: 'ros2 topic echo /odom' (wheel odometry)", rc == 0 and "pose" in out)
    # 2D Pose Estimate equivalent: TB3 world spawn pose
    pose = ("{header: {frame_id: map}, pose: {pose: {position: {x: -2.0, y: -0.5, z: 0.0}, orientation: {w: 1.0}}, "
            "covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}")
    amcl = ""
    for _ in range(6):
        run(f"ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \"{pose}\"", timeout=30)
        rc, amcl = run("timeout 10 ros2 topic echo /amcl_pose --once", timeout=20)
        if "position" in amcl: break
        time.sleep(5)
    check("46X Lab 11: initial pose -> AMCL publishes /amcl_pose", "position" in amcl)
    # wait until Nav2's lifecycle nodes are active (in the handout the student sends the goal from RViz afterwards)
    t0, state = time.time(), ""
    while time.time() - t0 < 180:
        rc, state = run("ros2 lifecycle get /bt_navigator", timeout=20)
        if state.strip().startswith("active"): break
        time.sleep(5)
    check("46X Lab 11: Nav2 lifecycle active (bt_navigator)", state.strip().startswith("active"), f"{time.time() - t0:.0f} s")
    goal = "{pose: {header: {frame_id: map}, pose: {position: {x: 0.5, y: -0.5, z: 0.0}, orientation: {w: 1.0}}}}"
    out = ""
    for _ in range(3):
        rc, out = run(f"timeout 240 ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \"{goal}\"", timeout=260)
        if "rejected" not in out: break
        time.sleep(10)
    check("46X Lab 11: Nav2 goal reached (NavigateToPose SUCCEEDED)", "SUCCEEDED" in out, out.strip().splitlines()[-1][:80] if out.strip() else "")
    stop_all()


try:
    {"361": course_361, "461": course_461, "46X": course_46X}[course]()
finally:
    stop_all()
bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} ROS 2 checks passed")
sys.exit(1 if bad else 0)
