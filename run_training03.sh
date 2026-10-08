#!/bin/bash
set -euo pipefail
task_root=/home/lyh/ddt_ros2_ws
exec rtk proxy env PYTHONPATH="$task_root/.python-deps:$task_root/training" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$@"
