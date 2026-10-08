#!/usr/bin/env bash
set -eo pipefail
source /home/lyh/ddt_ros2_ws/ros_env.sh
exec rtk proxy ros2 launch rl_controller sim_mujoco.launch.py robot:=d1 "$@"
