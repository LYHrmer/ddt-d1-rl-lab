#!/usr/bin/env bash
# The single launch entry for every contract 04 process that needs ROS. It
# loads the full ROS Humble environment and then execs the given command.
# It starts nothing by itself: no node, no message, no physics step.
#
# Usage:
#   runtime04/launch_env04.sh <command> [args...]
#   runtime04/launch_env04.sh --print-env     # resolved identity, runs nothing
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/ros_env04.sh"

if [ "$#" -eq 0 ]; then
  echo "usage: launch_env04.sh <command> [args...] | --print-env" >&2
  exit 2
fi

if [ "$1" = "--print-env" ]; then
  printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-}"
  printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-}"
  printf 'ROS_LOCALHOST_ONLY=%s\n' "${ROS_LOCALHOST_ONLY:-}"
  printf 'RMW_IMPLEMENTATION=%s\n' "${RMW_IMPLEMENTATION:-unset}"
  printf 'AMENT_PREFIX_PATH=%s\n' "${AMENT_PREFIX_PATH:-}"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH:-}"
  printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
  exit 0
fi

exec "$@"
