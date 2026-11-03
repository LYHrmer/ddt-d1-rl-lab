#pragma once
// Independent, effort-only 25/34 interface core. No ROS, actuator or ORT calls.
// One serialized owner must supply steady-clock nanoseconds and coherent samples.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace d1_adapter82 {
using Vector3 = std::array<double, 3>;
using Joints = std::array<double, 16>;
using Action = std::array<float, 16>;
using Frame = std::array<float, 57>;
using ActorInput = std::array<float, 627>;
constexpr std::int64_t kDeadlineNs = 20000000;
constexpr std::int64_t kCommandAgeNs = 500000000;  // New adapter82 policy.
constexpr bool wheel(std::size_t i) { return i % 4 == 3; }
constexpr double q0(std::size_t i) {
  return i % 4 == 1 ? .8 : (i % 4 == 2 ? -1.5 : 0.);
}
constexpr double action_scale(std::size_t i) { return wheel(i) ? 5. : .5; }
constexpr double kp(std::size_t i) { return wheel(i) ? 0. : 60.; }
constexpr double kd(std::size_t i) { return wheel(i) ? .5 : 1.5; }
constexpr double torque_limit(std::size_t i) { return wheel(i) ? 12. : 80.; }

template <class T, std::size_t N>
bool finite(const std::array<T, N>& values) {
  return std::all_of(values.begin(), values.end(), [](T v) { return std::isfinite(v); });
}

// Check before conversion: a finite double outside float's range must not be
// converted and only inspected afterwards. This mirrors server34's wire guard.
inline float checked_float(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("NONFINITE_FLOAT32_FRAME");
  constexpr double maximum = static_cast<double>(std::numeric_limits<float>::max());
  if (value < -maximum || value > maximum)
    throw std::invalid_argument("FLOAT32_FRAME_OUT_OF_RANGE");
  return static_cast<float>(value);
}

struct Sample {
  Vector3 gyro{};
  std::array<double, 4> quaternion_wxyz{1., 0., 0., 0.};
  Joints q{}, dq{};
};

// Same normalized wxyz inverse-rotation convention as training/d1_rl/interface.py.
inline Vector3 gravity_body(const std::array<double, 4>& quaternion) {
  if (!finite(quaternion)) throw std::invalid_argument("NONFINITE_QUATERNION");
  double norm = 0.;
  for (double value : quaternion) norm += value * value;
  norm = std::sqrt(norm);
  if (!std::isfinite(norm) || norm < 1e-12)
    throw std::invalid_argument("INVALID_QUATERNION_NORM");
  const double w = quaternion[0] / norm, x = quaternion[1] / norm;
  const double y = quaternion[2] / norm, z = quaternion[3] / norm;
  const Vector3 cross{-y, x, 0.};
  const Vector3 second{y * cross[2] - z * cross[1],
                       z * cross[0] - x * cross[2],
                       x * cross[1] - y * cross[0]};
  Vector3 result{};
  for (std::size_t i = 0; i < 3; ++i)
    result[i] = (i == 2 ? -1. : 0.) - 2. * w * cross[i] + 2. * second[i];
  return result;
}

struct Token { std::uint64_t generation = 0, sequence = 0; };
struct Request { Token token; ActorInput actor{}; };
struct Targets {
  Action action{};
  Joints position{}, velocity{}, proportional{}, derivative{};
  std::array<bool, 16> saturated{};
};
struct Output {
  bool authorized = false;
  Joints effort{};  // Unauthorized always means zero effort, NOT q0 position PD.
  std::optional<Targets> targets;
  std::string reason;
};

class Adapter {
 public:
  // A reset also requires a NEW actor_server34 process/session. Tokens alone do
  // not clear its private reference-shaper state. No wrapper is connected here.
  bool reset(std::int64_t now) {
    if (!clock(now)) return false;
    ++generation_;
    sequence_ = 0;
    filtered_ = {};
    last_action_ = {};
    history_ = {};
    current_ = {};
    command_ = {};
    pending_.reset();
    command_received_ = initialized_ = has_action_ = false;
    fault_.clear();
    reset_done_ = true;
    return true;
  }

  bool receive_command(Vector3 value, std::int64_t now) {
    if (!clock(now) || !ready()) return false;
    if (!finite(value)) return fail("NONFINITE_COMMAND");
    for (double& v : value) v = std::clamp(v, -1., 1.);
    command_ = value;
    command_time_ = now;
    command_received_ = true;
    return true;
  }

  std::optional<Request> begin_tick(const Sample& sample, std::int64_t now) {
    if (!clock(now) || !ready() || !fresh_command(now)) return std::nullopt;
    if (pending_) { fail("REQUEST_ALREADY_PENDING"); return std::nullopt; }
    if (!finite(sample.gyro) || !finite(sample.q) || !finite(sample.dq)) {
      fail("NONFINITE_SENSOR"); return std::nullopt;
    }
    try {
      const auto gravity = gravity_body(sample.quaternion_wxyz);
      Frame frame{};
      for (std::size_t i = 0; i < 3; ++i) {
        filtered_[i] = .97 * sample.gyro[i] + .03 * filtered_[i];
        frame[i] = checked_float(filtered_[i] * .25);
        frame[i + 3] = checked_float(gravity[i]);
        frame[i + 6] = checked_float(command_[i] * (i == 2 ? .25 : 2.));
      }
      for (std::size_t i = 0; i < 16; ++i) {
        frame[9 + i] = checked_float(wheel(i) ? 0. : sample.q[i] - q0(i));
        frame[25 + i] = checked_float(sample.dq[i] * .05);
        frame[41 + i] = last_action_[i];
      }
      if (!finite(frame)) throw std::invalid_argument("NONFINITE_FLOAT32_FRAME");
      if (!initialized_) { history_.fill(frame); initialized_ = true; }
      current_ = frame;
      Request request{{generation_, sequence_}, {}};
      std::copy(frame.begin(), frame.end(), request.actor.begin());
      for (std::size_t i = 0; i < 10; ++i)
        std::copy(history_[i].begin(), history_[i].end(), request.actor.begin() + (i + 1) * 57);
      pending_ = request.token;
      request_time_ = now;
      return request;
    } catch (const std::exception& error) {
      fail(error.what()); return std::nullopt;
    }
  }

  bool complete_tick(Token token, Action raw, std::int64_t now) {
    if (!clock(now) || !ready()) return false;
    if (!pending_ || token.generation != generation_ ||
        token.sequence != pending_->sequence || token.generation != pending_->generation)
      return fail("RESPONSE_TOKEN_MISMATCH");
    if (now - request_time_ > kDeadlineNs) return fail("INFERENCE_DEADLINE");
    if (!fresh_command(now)) return false;
    if (!finite(raw)) return fail("NONFINITE_ACTION");
    for (float& value : raw) value = std::clamp(value, -1.f, 1.f);
    last_action_ = raw;
    for (std::size_t i = 0; i < 9; ++i) history_[i] = history_[i + 1];
    history_[9] = current_;  // Append only after this frame's inference.
    pending_.reset();
    ++sequence_;
    has_action_ = true;
    return true;
  }

  Output torque_output(const Joints& q, const Joints& dq, std::int64_t now) {
    Output out;
    if (!clock(now) || !ready() || !fresh_command(now)) return stopped();
    if (pending_ && now - request_time_ > kDeadlineNs) fail("INFERENCE_DEADLINE");
    if (!fault_.empty()) return stopped();
    // No old-action hold while awaiting a response. This is an explicit core
    // contract, not an already-integrated ROS scheduling/physical-stop claim.
    if (pending_) { out.reason = "AWAITING_RESPONSE"; return out; }
    if (!has_action_) { out.reason = "NO_ACCEPTED_ACTION"; return out; }
    if (now - request_time_ > kDeadlineNs) { fail("ACTION_DEADLINE"); return stopped(); }
    if (!finite(q) || !finite(dq)) { fail("NONFINITE_PD_SENSOR"); return stopped(); }
    Targets target;
    target.action = last_action_;
    for (std::size_t i = 0; i < 16; ++i) {
      target.position[i] = wheel(i) ? 0. : q0(i) + action_scale(i) * last_action_[i];
      target.velocity[i] = wheel(i) ? action_scale(i) * last_action_[i] : 0.;
      target.proportional[i] = kp(i);
      target.derivative[i] = kd(i);
      const double raw = kp(i) * (target.position[i] - q[i]) +
                         kd(i) * (target.velocity[i] - dq[i]);
      if (!std::isfinite(raw)) { fail("NONFINITE_PD_TORQUE"); return stopped(); }
      target.saturated[i] = std::abs(raw) >= torque_limit(i);
      out.effort[i] = std::clamp(raw, -torque_limit(i), torque_limit(i));
    }
    out.authorized = true;
    out.targets = target;
    out.reason = "AUTHORIZED_EFFORT_ONLY";
    return out;
  }

  const std::string& fault() const { return fault_; }
  std::uint64_t generation() const { return generation_; }

 private:
  bool fail(const std::string& reason) {
    if (fault_.empty()) fault_ = reason;
    last_action_ = {};
    has_action_ = false;
    pending_.reset();
    return false;
  }
  bool clock(std::int64_t now) {
    if (now < 0 || (time_seen_ && now < last_time_)) return fail("CLOCK_REGRESSION");
    time_seen_ = true;
    last_time_ = now;
    return true;
  }
  bool ready() {
    if (!reset_done_) return fail("RESET_REQUIRED");
    return fault_.empty();
  }
  bool fresh_command(std::int64_t now) {
    if (!command_received_) return fail("COMMAND_REQUIRED");
    if (now - command_time_ > kCommandAgeNs) return fail("COMMAND_STALE");
    return true;
  }
  Output stopped() const { Output out; out.reason = fault_; return out; }
  bool reset_done_ = false, time_seen_ = false, command_received_ = false;
  bool initialized_ = false, has_action_ = false;
  std::uint64_t generation_ = 0, sequence_ = 0;
  std::int64_t last_time_ = 0, command_time_ = 0, request_time_ = 0;
  Vector3 command_{}, filtered_{};
  Action last_action_{};
  Frame current_{};
  std::array<Frame, 10> history_{};
  std::optional<Token> pending_;
  std::string fault_;
};
}  // namespace d1_adapter82
