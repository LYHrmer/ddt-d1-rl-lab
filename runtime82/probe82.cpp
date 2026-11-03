// Explicit synthetic-input CLI. It never starts ROS, ORT or hardware.
#include "adapter82.hpp"
#include <iostream>
#include <string>

using namespace d1_adapter82;
template <class T, std::size_t N>
void array_json(const std::array<T, N>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "probe82 --scenario nominal|late|stale_command|reset (synthetic; no ROS/ORT/physics)\n";
    return 0;
  }
  if (argc != 3 || std::string(argv[1]) != "--scenario") return 2;
  const std::string scenario = argv[2];
  if (scenario != "nominal" && scenario != "late" &&
      scenario != "stale_command" && scenario != "reset") return 2;
  Adapter adapter;
  Sample sample;
  for (std::size_t i = 0; i < 16; ++i) sample.q[i] = q0(i);
  sample.gyro = {.1, -.2, .3};
  adapter.reset(0);
  adapter.receive_command({.1, 0., .2}, 0);
  const auto request = adapter.begin_tick(sample, 0);
  if (!request) return 3;
  Action action{};
  for (std::size_t i = 0; i < 16; ++i) action[i] = wheel(i) ? 2.f : .25f;
  const auto finish = scenario == "late" ? kDeadlineNs + 1 : 1000000;
  const bool accepted = adapter.complete_tick(request->token, action, finish);
  auto output_time = finish;
  if (scenario == "reset") adapter.reset(finish);
  if (scenario == "stale_command") output_time = kCommandAgeNs + 1;
  const auto out = adapter.torque_output(sample.q, sample.dq, output_time);
  std::cout.precision(17);
  std::cout << "{\"schema\":\"d1-adapter82-synthetic-probe-v1\",\"scenario\":\"" << scenario
            << "\",\"synthetic_actions\":true,\"ros_connected\":false,\"physics_steps\":0,"
            << "\"accepted_response\":" << (accepted ? "true" : "false")
            << ",\"authorized\":" << (out.authorized ? "true" : "false")
            << ",\"has_policy_targets\":" << (out.targets ? "true" : "false")
            << ",\"reason\":\"" << out.reason << "\",\"raw_actor627\":";
  array_json(request->actor);
  std::cout << ",\"effort16\":";
  array_json(out.effort);
  if (out.targets) {
    std::cout << ",\"clipped_action16\":"; array_json(out.targets->action);
    std::cout << ",\"target_q16\":"; array_json(out.targets->position);
    std::cout << ",\"target_dq16\":"; array_json(out.targets->velocity);
  }
  std::cout << "}\n";
}
