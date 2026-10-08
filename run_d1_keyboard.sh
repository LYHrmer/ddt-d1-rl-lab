#!/usr/bin/env bash
set -eo pipefail
source /home/lyh/ddt_ros2_ws/ros_env.sh
exec rtk proxy ros2 run keyboard_controller keyboard_controller_node
