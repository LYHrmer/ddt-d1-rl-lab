#!/bin/bash
set -euo pipefail
exec rtk proxy /home/lyh/ddt_ros2_ws/run_d1_mujoco.sh scene:=scene_training_course_03.xml "$@"
