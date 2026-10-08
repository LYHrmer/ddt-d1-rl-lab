#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace d1rt03 {
using Json = nlohmann::json;
using V3 = std::array<double, 3>;
using Q4 = std::array<double, 4>;
using J16 = std::array<double, 16>;
using A16 = std::array<float, 16>;
using Obs = std::array<float, 57>;
using History = std::array<float, 570>;
inline constexpr std::array<const char*, 16> joint_names = {
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint"};
template<class T> inline void require_finite(const T& values, const char* context)
{
  for (auto v : values)
    if (!std::isfinite(v))
      throw std::runtime_error(std::string(context) + ": nonfinite");
}
template<size_t N, class T = double> std::array<T, N> fixed(const Json& j, const char* context)
{
  if (!j.is_array() || j.size() != N)
    throw std::runtime_error(std::string(context) + ": wrong shape");
  std::array<T, N> a{};
  for (size_t i = 0; i < N; ++i)
    a[i] = j.at(i).get<T>();
  require_finite(a, context);
  return a;
}
inline bool wheel(size_t i)
{
  return i % 4 == 3;
}
struct Config {
  J16 q0{}, kp{}, kd{}, scale{}, torque_limit{};
  double native_dt = .002;
  int pd_decimation = 2, policy_decimation = 10;
  double watchdog_seconds = .5;
  double command_vx_limit = .3, command_wz_limit = .3;
  explicit Config(const Json& j)
  {
    if (j.at("interface_id") != "d1-selftrained-pilot03-v1" &&
        j.at("interface_id") != "d1-selftrained-pilot03b-v2")
      throw std::runtime_error("schema interface_id mismatch");
    q0 = fixed<16>(j.at("default_joint_angles"), "default_joint_angles");
    kp = fixed<16>(j.at("kp"), "kp");
    kd = fixed<16>(j.at("kd"), "kd");
    scale = fixed<16>(j.at("action_scales"), "action_scales");
    torque_limit = fixed<16>(j.at("torque_limits"), "torque_limits");
    native_dt = j.at("native_dt").get<double>();
    pd_decimation = j.at("pd_decimation").get<int>();
    policy_decimation = j.at("policy_decimation").get<int>();
    watchdog_seconds = j.at("watchdog_seconds").get<double>();
    if (std::abs(native_dt - .002) > 1e-12 || pd_decimation != 2 || policy_decimation != 10 ||
        std::abs(watchdog_seconds - .5) > 1e-12)
      throw std::runtime_error("pilot03 frozen timing mismatch");
    if (j.at("joint_order").size() != 16)
      throw std::runtime_error("joint order shape");
    for (size_t i = 0; i < 16; ++i)
    {
      if (j.at("joint_order").at(i) != joint_names[i])
        throw std::runtime_error("joint order mismatch");
      if (kp[i] < 0 || kd[i] < 0 || scale[i] <= 0 || torque_limit[i] != (wheel(i)?12.:80.))
        throw std::runtime_error("invalid gains/scale/physical torque limit");
    }
  }
};
inline V3 cross(const V3&a, const V3&b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline V3 inverse_rotate(Q4 q, const V3&v)
{
  require_finite(q, "quaternion");
  require_finite(v, "vector");
  double norm = 0;
  for (double x:q)
    norm += x * x;
  norm = std::sqrt(norm);
  if (norm < 1e-12)
    throw std::runtime_error("zero quaternion");
  for (double&x:q)
    x /= norm;
  V3 xyz{q[1], q[2], q[3]}, c = cross(xyz, v), cc = cross(xyz, c), out{};
  for (int i = 0; i < 3; ++i)
    out[i] = v[i] - 2 * q[0] * c[i] + 2 * cc[i];
  return out;
}
struct Sample { V3 gyro{}, command{};
                Q4 quat{1, 0, 0, 0};
                J16 q{}, dq{};
};
inline Sample sample_json(const Json&j)
{
  Sample s;
  s.gyro = fixed<3>(j.at("gyro"), "gyro");
  s.quat = fixed<4>(j.at("quat_wxyz"), "quat_wxyz");
  s.command = fixed<3>(j.at("command"), "command");
  s.q = fixed<16>(j.at("q"), "q");
  s.dq = fixed<16>(j.at("dq"), "dq");
  return s;
}
inline A16 clip_action(const std::vector<float>& raw)
{
  if (raw.size() != 16)
    throw std::runtime_error("action shape must be 16");
  require_finite(raw, "policy action");
  A16 a{};
  for (size_t i = 0; i < 16; ++i)
    a[i] = std::clamp(raw[i], -1.f, 1.f);
  return a;
}
struct Torque { J16 value{}, raw{};
                std::array<bool, 16> saturated{};
};
inline Torque pd(const Config&c, const A16&a, const J16&q, const J16&dq)
{
  require_finite(a, "executed action");
  require_finite(q, "q");
  require_finite(dq, "dq");
  Torque t;
  for (size_t i = 0; i < 16; ++i)
  {
    if (std::abs(a[i]) > 1.0000001f)
      throw std::runtime_error("PD expects clipped action");
    double target_q = c.q0[i] + c.scale[i] * static_cast<double>(a[i]);
    double target_dq = wheel(i)?c.scale[i] * static_cast<double>(a[i]):0.;
    t.raw[i] = c.kp[i] * (target_q - q[i]) + c.kd[i] * (target_dq - dq[i]);
    t.saturated[i] = std::abs(t.raw[i]) >= c.torque_limit[i];
    t.value[i] = std::clamp(t.raw[i], -c.torque_limit[i], c.torque_limit[i]);
  }
  return t;
}
struct Interface {
  const Config& cfg;
  V3 filtered{};
  A16 last_issued{}, executed{};
  Obs current{};
  History history{};
  std::deque<A16> queue;
  int delay_ticks = 0;
  explicit Interface(const Config&c) : cfg(c)
  {
  }
  Obs observe(const Sample&s)
  {
    require_finite(s.gyro, "gyro");
    require_finite(s.command, "command");
    require_finite(s.q, "q");
    require_finite(s.dq, "dq");
    V3 gravity = inverse_rotate(s.quat, {0, 0, -1});
    for (int i = 0; i < 3; ++i)
    {
      filtered[i] = .97 * s.gyro[i] + .03 * filtered[i];
      current[i] = float(filtered[i] * .25);
      current[3 + i] = float(gravity[i]);
    }
    current[6] = float(std::clamp(s.command[0], -1., 1.) * 2);
    current[7] = float(std::clamp(s.command[1], -1., 1.) * 2);
    current[8] = float(std::clamp(s.command[2], -1., 1.) * .25);
    for (size_t i = 0; i < 16; ++i)
    {
      current[9 + i] = wheel(i)?0.f:float(s.q[i] - cfg.q0[i]);
      current[25 + i] = float(s.dq[i] * .05);
      current[41 + i] = last_issued[i];
    }
    require_finite(current, "observation");
    return current;
  }
  void reset(const Sample&s, int delay)
  {
    if (delay < 0 || delay > 3)
      throw std::runtime_error("delay_ticks must be 0..3");
    delay_ticks = delay;
    filtered.fill(0);
    last_issued.fill(0);
    executed.fill(0);
    queue.assign(delay, A16{});
    observe(s);
    for (size_t i = 0; i < 10; ++i)
      std::copy(current.begin(), current.end(), history.begin() + i * 57);
  }
  A16 accept(const std::vector<float>&raw)
  {
    last_issued = clip_action(raw);
    queue.push_back(last_issued);
    executed = queue.front();
    queue.pop_front();
    std::move(history.begin() + 57, history.end(), history.begin());
    std::copy(current.begin(), current.end(), history.end() - 57);
    return executed;
  }
  void fallback()
  {
    last_issued.fill(0);
    executed.fill(0);
    queue.assign(delay_ticks, A16{});
  }
};
struct Watchdog {
  V3 requested{};
  double last_received = 0;
  bool received = false;
  void receive(const V3&v, double now)
  {
    require_finite(v, "command");
    require_finite(V3{now, 0, 0}, "clock");
    requested = v;
    last_received = now;
    received = true;
  }
  V3 value(double now, double timeout, bool*stale = nullptr)const
  {
    bool expired = !received || now - last_received > timeout || now < last_received;
    if (stale)
      *stale = expired;
    return expired?V3{}:requested;
  }
};
} // namespace d1rt03
