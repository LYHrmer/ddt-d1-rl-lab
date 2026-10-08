#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/lyh/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Run this script through sudo in a local terminal. It installs only the
# requested ROS2/MuJoCo build dependencies and the downloaded ONNX 1.10.0.
rtk proxy test "$(rtk proxy id -u)" -eq 0
rtk proxy test -f /home/lyh/ddt_ros2_ws/downloads/onnxruntime-linux-x64-1.10.0/include/onnxruntime_cxx_api.h
rtk proxy test -f /home/lyh/ddt_ros2_ws/downloads/onnxruntime-linux-x64-1.10.0/lib/libonnxruntime.so.1.10.0
rtk proxy apt-get update
rtk proxy apt-get install -y --no-upgrade \
  ros-humble-ros2-control ros-humble-ros2-controllers \
  ros-humble-ament-cmake-auto ros-humble-generate-parameter-library \
  ros-humble-xacro ros-humble-robot-state-publisher \
  ros-humble-control-toolbox ros-humble-eigen3-cmake-module \
  python3-colcon-common-extensions libglfw3-dev libglew-dev \
  libeigen3-dev libyaml-cpp-dev libgl1-mesa-dev \
  libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev cmake build-essential
rtk proxy cp -a /home/lyh/ddt_ros2_ws/downloads/onnxruntime-linux-x64-1.10.0/include/. /usr/include/
rtk proxy cp -a /home/lyh/ddt_ros2_ws/downloads/onnxruntime-linux-x64-1.10.0/lib/. /usr/lib/
rtk proxy ldconfig
rtk proxy touch /home/lyh/ddt_ros2_ws/logs/system_dependencies_installed.ok
rtk proxy printf '\nD1 system dependencies installed successfully. You may close this terminal.\n'
