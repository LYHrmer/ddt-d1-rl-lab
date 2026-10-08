// Offline harness: links and executes the unmodified deployed FSMState_RL.cpp.
// There is no MuJoCo/ROS link and no physics. Only the background scheduler is
// replaced by a no-op; each real update_forward loop is stopped after one ONNX call.
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl_controller/fsm/FSMState_RL.h"
#include "rl_controller/inferrer/onnx_inferrer.hpp"
#include "interface_config03.hpp"

class OneShot03 final : public InferrerBase {
 public:
  ONNXInferrer* real;
  bool* running;
  std::vector<std::vector<float>> captured;
  std::vector<float> output;
  OneShot03(ONNXInferrer* r, bool* flag) : real(r), running(flag) {}
  bool loadModel(const std::string& p, bool v = false) override {
    return real->loadModel(p, v);
  }
  bool setOutput(const std::string& p, size_t n) override {
    return real->setOutput(p, n);
  }
  std::vector<float> computeActions(const std::vector<std::vector<float>>& input) override {
    captured = input;
    output = real->computeActions(input);  // Actual shipped ONNXInferrer.cpp / ORT 1.10.
    *running = false;                    // Bound the real FSM loop to one iteration.
    return output;
  }
};

class AuditedFSM03 final : public FSMState_RL {
 public:
  std::unique_ptr<ONNXInferrer> real;
  OneShot03* proxy;
  AuditedFSM03(std::shared_ptr<ControlFSMData> data, RLParameters* p)
      : FSMState_RL(data, p, "interface_audit_flat_lab") {
    // The original InferrerBase has no virtual destructor. Keep ownership typed.
    real.reset(static_cast<ONNXInferrer*>(inferrer_.release()));
    proxy = new OneShot03(real.get(), &threadRunning);
    inferrer_.reset(proxy);
  }
  ~AuditedFSM03() override {
    if (forward_thread.joinable()) forward_thread.join();
    inferrer_.release();
    delete proxy;
  }
  void update_forward() override {}  // Suppress only the asynchronous worker.
  void enter_bounded() {
    FSMState_RL::enter();            // Genuine reset, observation and history fill.
    if (forward_thread.joinable()) forward_thread.join();
  }
  void forward_bounded() {
    threadRunning = true;
    FSMState_RL::update_forward();  // Genuine observation / ONNX / history path.
  }
  void inject_action(const std::vector<float>& a) { action_vec_ = vectorToEigen(a); }
  const DVec<float>& obs() const { return obs_vec_; }
  const DVec<float>& history() const { return obs_history_vec_; }
  const DVec<float>& action() const { return action_vec_; }
};

template <typename V> void vec(const V& x) {
  std::cout << '[';
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    if (i) std::cout << ',';
    std::cout << x[i];
  }
  std::cout << ']';
}

int main() {
  std::cout << std::setprecision(17);
  RLParameters params;
  auto data = std::make_shared<ControlFSMData>(16);
  configure_interface03(params, *data);
  // RemoteControlData::zero uses .Zero() expressions without assigning three
  // vectors. All fixture state is therefore explicitly initialized here.
  data->rc_data->twist_linear.setZero();
  data->rc_data->twist_angular.setZero();
  data->rc_data->pose_position.setZero();
  AuditedFSM03 fsm(data, &params);
  std::cout << "{\"kind\":\"metadata\",\"onnxruntime\":\""
            << OrtGetApiBase()->GetVersionString() << "\",\"commands_scale\":";
  vec(params.commands_scale);
  std::cout << ",\"commands_gain\":"; vec(params.commands_gain);
  std::cout << ",\"commands_comp\":"; vec(params.commands_comp);
  std::cout << ",\"time_interval\":" << params.time_interval << "}\n";
  int count;
  if (!(std::cin >> count) || count < 1 || count > 40) return 2;
  for (int record = 0; record < count; ++record) {
    std::string op, label;
    std::cin >> op >> label;
    for (int i = 0; i < 3; ++i) std::cin >> data->low_state->gyro[i];
    for (int i = 0; i < 4; ++i) std::cin >> data->low_state->quat[i];
    std::cin >> data->rc_data->twist_linear[0]
             >> data->rc_data->twist_linear[1]
             >> data->rc_data->twist_angular[2];
    for (int i = 0; i < 16; ++i) std::cin >> data->low_state->q[i];
    for (int i = 0; i < 16; ++i) std::cin >> data->low_state->dq[i];
    fsm.proxy->captured.clear();
    fsm.proxy->output.clear();
    if (op == "enter") fsm.enter_bounded();
    else if (op == "forward") fsm.forward_bounded();
    else if (op == "injected_action") {
      std::vector<float> action(16);
      for (float& value : action) std::cin >> value;
      fsm.inject_action(action);
    } else return 3;
    if (!std::cin) return 4;
    fsm.run();  // Genuine deployed action -> LowlevelCmd mapping.
    std::cout << "{\"kind\":\"case\",\"op\":\"" << op
              << "\",\"label\":\"" << label << "\",\"obs_after\":";
    vec(fsm.obs());
    std::cout << ",\"history_after\":"; vec(fsm.history());
    std::cout << ",\"action_after\":"; vec(fsm.action());
    std::cout << ",\"onnx_output\":"; vec(fsm.proxy->output);
    std::cout << ",\"inputs\":[";
    for (int i = 0; i < static_cast<int>(fsm.proxy->captured.size()); ++i) {
      if (i) std::cout << ',';
      vec(fsm.proxy->captured[i]);
    }
    std::cout << "],\"qd\":"; vec(data->low_cmd->qd);
    std::cout << ",\"qd_dot\":"; vec(data->low_cmd->qd_dot);
    std::cout << ",\"kp\":"; vec(data->low_cmd->kp);
    std::cout << ",\"kd\":"; vec(data->low_cmd->kd);
    std::cout << ",\"tau_ff\":"; vec(data->low_cmd->tau_cmd);
    std::cout << "}\n";
  }
  return 0;
}
