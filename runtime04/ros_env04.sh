#!/usr/bin/env bash
# Full ROS Humble environment for contract 04 entries. Source only: this file
# exports variables and creates no node, publishes no message and integrates no
# physics step.
#
# The frozen 03 launcher hand-wrote PYTHONPATH with only
# /opt/ros/humble/local/lib/python3.10/dist-packages, which dropped
# /opt/ros/humble/lib/python3.10/site-packages where rpyutils is installed, so
# `import rclpy` died with ModuleNotFoundError before the native runner started.
# Sourcing setup.bash is what puts both entries on the path; never override
# PYTHONPATH wholesale after this point.

D1_WS04=/home/lyh/ddt_ros2_ws

# setup.bash is not written for `set -u`/`set -e` callers, so relax while sourcing.
__d1_env04_opts="$(set +o)"
set +eu
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [ -f "$D1_WS04/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$D1_WS04/install/setup.bash"
fi
eval "$__d1_env04_opts"
unset __d1_env04_opts

# Dedicated domain, loopback only: no traffic leaves this host.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-84}"
export ROS_LOCALHOST_ONLY=1

# MuJoCo 3.3.0 shared objects for the frozen native runner, appended after the
# ROS libraries exactly as the 03 deployment commands did.
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$D1_WS04/install/mujoco/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Single-threaded BLAS keeps native step timing comparable with the 03 receipts.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
