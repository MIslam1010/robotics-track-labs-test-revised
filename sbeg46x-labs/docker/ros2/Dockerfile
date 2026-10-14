# ROS 2 Jazzy desktop (RViz) + the course Python packages, for the labs with ROS 2 parts.
#   docker build -t sbeg46x-labs-ros2 -f docker/ros2/Dockerfile .
#   Linux:   docker run -it --rm --net=host -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD":/work sbeg46x-labs-ros2
#   macOS/Windows: use VS Code "Reopen in Container" with .devcontainer/ros2, or the lab PCs for RViz windows.
FROM osrf/ros:jazzy-desktop
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv build-essential cmake libeigen3-dev nlohmann-json3-dev git ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb3-sim && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
# Course packages in a virtual environment ON TOP OF the ROS 2 Python (--system-site-packages): rclpy, tf2_ros and the
# message packages stay visible, and NumPy stays Ubuntu's 1.26 that ROS 2 Jazzy is built against (NumPy 2 breaks some
# ROS message types). All course packages support NumPy 1.26. toppra (461 Lab 6 only) has no wheel for Python 3.12;
# if it cannot be built, do Lab 6 in your normal course environment.
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv --system-site-packages /opt/course-venv \
 && grep -vE '^(numpy|toppra)==' /tmp/requirements.txt > /tmp/requirements-ros.txt \
 && /opt/course-venv/bin/pip install --no-cache-dir -r /tmp/requirements-ros.txt \
 && (! grep -q '^toppra==' /tmp/requirements.txt || /opt/course-venv/bin/pip install --no-cache-dir "$(grep '^toppra==' /tmp/requirements.txt)" \
     || echo "toppra could not be built for Python 3.12 here: do Lab 6 in your normal course environment")
ENV PATH=/opt/course-venv/bin:$PATH
WORKDIR /work
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
CMD ["bash"]
