#ifndef MUJOCO_ROS2_CONTROL_STEP_EVIDENCE_HPP
#define MUJOCO_ROS2_CONTROL_STEP_EVIDENCE_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "mujoco/mujoco.h"

namespace mujoco_ros2_control {
// Opt-in observation only. Record after the solver/integrator and before the
// next forward pass: derived geometry and contact forces share pre-step time.
class StepEvidence {
 public:
  StepEvidence(const std::string &path, const mjModel *model, const mjData *data,
               uint64_t max_native_steps);
  ~StepEvidence();
  StepEvidence(const StepEvidence &) = delete;
  StepEvidence &operator=(const StepEvidence &) = delete;
  void Record(const mjModel *model, const mjData *data,
              uint64_t control_count, bool split_step);

 private:
  void Write(const std::string &line, bool flush = false);
  FILE *file_ = nullptr;
  uint64_t count_ = 0;
  uint64_t max_native_steps_ = 0;
  uint64_t bytes_written_ = 0;
  double last_post_time_ = 0;
  std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
  int base_id_ = -1;
  int velocity_adr_ = -1;
  int floor_id_ = -1;
  int step_id_ = -1;
  std::array<int, 4> wheel_ids_;
  std::vector<int> robot_geoms_;
  std::vector<bool> is_robot_;
  std::vector<bool> is_wheel_;
};
}  // namespace mujoco_ros2_control
#endif
