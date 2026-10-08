#!/usr/bin/env bash
set -euo pipefail
rtk proxy env CMAKE_PREFIX_PATH=/opt/ros/humble:/home/lyh/ddt_ros2_ws/install/mujoco cmake -S /home/lyh/ddt_ros2_ws/runtime03 -B /home/lyh/ddt_ros2_ws/runtime03/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
rtk proxy cmake --build /home/lyh/ddt_ros2_ws/runtime03/build --parallel 4
