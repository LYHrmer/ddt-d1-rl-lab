#include "runtime82/adapter82.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace d1_adapter82;
int checks = 0;
void expect(bool ok, const char* message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
  expect(std::abs(actual - expected) < 1e-7, message);
}
Sample sample() {
  Sample s;
  for (std::size_t i = 0; i < 16; ++i) s.q[i] = q0(i);
  return s;
}
Request start(Adapter& a, const Sample& s, std::int64_t now = 0) {
  expect(a.reset(now), "reset");
  expect(a.receive_command({.1, -.2, .3}, now), "command");
  auto request = a.begin_tick(s, now);
  expect(request.has_value(), "request");
  return *request;
}
void stopped(const Output& out) {
  expect(!out.authorized && !out.targets, "no authorized pose after fault/reset");
  for (double torque : out.effort) expect(torque == 0., "no old effort");
}

void observation_and_mapping() {
  Adapter a;
  auto s = sample();
  s.gyro = {1., 2., 3.};
  s.quaternion_wxyz = {2., 0., 0., 0.};
  s.q[3] = 99.;
  s.dq[2] = 4.;
  auto request = start(a, s);
  close(request.actor[0], .2425, "gyro filtered once");
  close(request.actor[5], -1., "normalized gravity");
  close(request.actor[6], .2, "vx command scale");
  close(request.actor[8], .075, "raw yaw, no double reference shaping");
  close(request.actor[12], 0., "wheel phase masked");
  close(request.actor[27], .2, "joint velocity scale");
  for (std::size_t row = 1; row <= 10; ++row)
    for (std::size_t col = 0; col < 57; ++col)
      expect(request.actor[row * 57 + col] == request.actor[col], "reset constant prehistory");
  Action actions;
  actions.fill(2.f);
  expect(a.complete_tick(request.token, actions, 1000), "accept");
  const auto out = a.torque_output(s.q, s.dq, 1000);
  expect(out.authorized && out.targets.has_value(), "authorized output");
  close(out.targets->position[0], .5, "25/34 leg multiplier, not legacy .25");
  close(out.targets->position[1], 1.3, "default joint pose plus offset");
  close(out.targets->velocity[3], 5., "wheel multiplier");
  close(out.effort[0], 30., "leg PD");
  close(out.effort[3], 2.5, "wheel PD ignores wheel position");
  auto second = a.begin_tick(s, 20000000);
  expect(second.has_value(), "second request");
  close(second->actor[41], 1., "previous action is clipped");
  close(second->actor[57 + 41], 0., "history excludes current");
  close(second->actor[0], .249775, "gyro filter carries once");
  stopped(a.torque_output(s.q, s.dq, 20000000));
}

void nontrivial_history() {
  Adapter a;
  expect(a.reset(0), "history reset");
  std::vector<Frame> expected_frames;
  Action previous{};
  // Twelve distinct frames exercise both initial padding and a full FIFO wrap.
  // Expected gyro uses a closed-form ramp response, not adapter filter state.
  for (int tick = 0; tick < 12; ++tick) {
    const std::int64_t now = static_cast<std::int64_t>(tick) * 20000000;
    auto s = sample();
    Vector3 command{.02 * tick, -.03 * tick, (tick % 2 ? -.2 : .3)};
    expect(a.receive_command(command, now), "history command");
    Frame expected{};
    for (std::size_t i = 0; i < 3; ++i) {
      const double increment = .1 * (i + 1);
      s.gyro[i] = increment * (tick + 1);
      const double filtered = increment * ((tick + 1) - .03 / .97 * (1. - std::pow(.03, tick + 1)));
      expected[i] = static_cast<float>(filtered * .25);
      expected[3 + i] = i == 2 ? -1.f : 0.f;
      expected[6 + i] = static_cast<float>(command[i] * (i == 2 ? .25 : 2.));
    }
    for (std::size_t i = 0; i < 16; ++i) {
      const double offset = .001 * (tick + 1) * (i + 1);
      s.q[i] += offset;
      s.dq[i] = .02 * (tick + 1) * (i + 1);
      expected[9 + i] = static_cast<float>(i % 4 == 3 ? 0. : offset);
      expected[25 + i] = static_cast<float>(.001 * (tick + 1) * (i + 1));
      expected[41 + i] = previous[i];
    }
    expected_frames.push_back(expected);
    auto request = a.begin_tick(s, now);
    expect(request.has_value(), "history sequence request");
    for (std::size_t col = 0; col < 57; ++col)
      close(request->actor[col], expected[col], "current frame independently constructed");
    for (int row = 0; row < 10; ++row) {
      const int expected_tick = std::max(0, tick - 10 + row);
      for (std::size_t col = 0; col < 57; ++col)
        close(request->actor[(row + 1) * 57 + col], expected_frames[expected_tick][col],
              "history is ten previous frames oldest first, no current frame");
    }
    Action action{};
    for (std::size_t i = 0; i < 16; ++i) {
      action[i] = static_cast<float>((tick % 2 ? -1. : 1.) * .1 * (i + 1));
      previous[i] = std::clamp(action[i], -1.f, 1.f);
    }
    expect(a.complete_tick(request->token, action, now + 1000), "history accepted action");
  }
}

void deadlines_and_tokens() {
  const auto s = sample();
  Action actions{};
  for (auto elapsed : {kDeadlineNs, kDeadlineNs + 1}) {
    Adapter a;
    const auto request = start(a, s);
    const bool accepted = a.complete_tick(request.token, actions, elapsed);
    expect(accepted == (elapsed == kDeadlineNs), "inclusive 20ms boundary");
    const auto out = a.torque_output(s.q, s.dq, elapsed);
    expect(out.authorized == accepted, "no late output");
    if (!accepted) stopped(out);
  }
  Adapter clock_bad;
  auto request = start(clock_bad, s, 100);
  expect(!clock_bad.complete_tick(request.token, actions, 99), "clock regression refused");
  stopped(clock_bad.torque_output(s.q, s.dq, 100));
  Adapter old;
  auto before = start(old, s);
  expect(old.reset(1), "reset while pending");
  expect(old.receive_command({0., 0., 0.}, 1), "post reset command");
  const auto after = old.begin_tick(s, 1);
  expect(after && after->token.generation != before.token.generation, "new session generation");
  expect(!old.complete_tick(before.token, actions, 2), "stale response cross reset");
  stopped(old.torque_output(s.q, s.dq, 2));
  Adapter duplicate;
  request = start(duplicate, s);
  expect(duplicate.complete_tick(request.token, actions, 1), "first response");
  expect(!duplicate.complete_tick(request.token, actions, 2), "duplicate response refused");
  stopped(duplicate.torque_output(s.q, s.dq, 2));
  Adapter expired;
  request = start(expired, s);
  actions.fill(1.f);
  expect(expired.complete_tick(request.token, actions, 1), "action accepted");
  stopped(expired.torque_output(s.q, s.dq, kDeadlineNs + 1));
  expect(expired.fault() == "ACTION_DEADLINE", "accepted output cannot live forever");
}

void command_age_and_reset() {
  const auto s = sample();
  for (auto age : {kCommandAgeNs, kCommandAgeNs + 1}) {
    Adapter a;
    expect(a.reset(0), "reset command age");
    expect(a.receive_command({0., 0., 0.}, 0), "receive age");
    expect(a.begin_tick(s, age).has_value() == (age == kCommandAgeNs), "500ms boundary");
  }
  Adapter a;
  auto moving = s;
  moving.gyro = {3., 2., 1.};
  const auto first = start(a, moving);
  Action actions;
  actions.fill(1.f);
  expect(a.complete_tick(first.token, actions, 1), "action before hot reset");
  expect(a.reset(2), "hot reset");
  expect(a.receive_command({0., 0., 0.}, 2), "new command after reset");
  stopped(a.torque_output(s.q, s.dq, 2));
  const auto next = a.begin_tick(s, 3);
  expect(next.has_value(), "request after hot reset");
  for (std::size_t row = 0; row < 11; ++row) {
    for (std::size_t col = 0; col < 3; ++col)
      expect(next->actor[row * 57 + col] == 0.f, "filter/history cleared");
    for (std::size_t col = 41; col < 57; ++col)
      expect(next->actor[row * 57 + col] == 0.f, "last action/history cleared");
  }
}

void invalid_values_and_limits() {
  const auto s = sample();
  const auto pitched = gravity_body({2. * std::cos(.3), 0., 2. * std::sin(.3), 0.});
  close(pitched[0], std::sin(.6), "normalized inverse rotation under pitch");
  close(pitched[2], -std::cos(.6), "pitched gravity vertical");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double value : {nan, inf, -inf}) {
    for (int field = 0; field < 39; ++field) {
      Adapter a;
      auto bad = s;
      if (field < 4) bad.quaternion_wxyz[field] = value;
      else if (field < 7) bad.gyro[field - 4] = value;
      else if (field < 23) bad.q[field - 7] = value;
      else bad.dq[field - 23] = value;
      expect(a.reset(0) && a.receive_command({0., 0., 0.}, 0), "bad sensor setup");
      expect(!a.begin_tick(bad, 0), "nonfinite sensor refused");
      stopped(a.torque_output(s.q, s.dq, 0));
    }
    Adapter action;
    const auto request = start(action, s);
    Action bad_action{};
    bad_action[3] = static_cast<float>(value);
    expect(!action.complete_tick(request.token, bad_action, 1), "nonfinite action refused");
    stopped(action.torque_output(s.q, s.dq, 1));
    Adapter command;
    expect(command.reset(0), "command reset");
    expect(!command.receive_command({value, 0., 0.}, 0), "nonfinite command refused");
    Adapter pd;
    const auto pd_request = start(pd, s);
    expect(pd.complete_tick(pd_request.token, Action{}, 1), "PD invalid sensor setup");
    auto bad_q = s.q;
    bad_q[0] = value;
    stopped(pd.torque_output(bad_q, s.dq, 1));
  }
  Adapter quaternion;
  auto zero_q = s;
  zero_q.quaternion_wxyz = {};
  expect(quaternion.reset(0) && quaternion.receive_command({0., 0., 0.}, 0), "zero quat setup");
  expect(!quaternion.begin_tick(zero_q, 0), "zero quaternion refused");
  for (double sign : {-1., 1.}) {
    Adapter a;
    const auto request = start(a, s);
    Action actions{};
    expect(a.complete_tick(request.token, actions, 1), "limit setup");
    auto joints = s.q, velocities = s.dq;
    for (std::size_t i = 0; i < 16; ++i) {
      joints[i] = -sign * 100.;
      velocities[i] = -sign * 100.;
    }
    const auto out = a.torque_output(joints, velocities, 1);
    expect(out.authorized && out.targets.has_value(), "limited output valid");
    for (std::size_t i = 0; i < 16; ++i) {
      expect(out.effort[i] == sign * (wheel(i) ? 12. : 80.), "both torque bounds assigned");
      expect(out.targets->saturated[i], "saturation recorded");
    }
  }
}

void finite_conversion_boundaries() {
  const double maximum = static_cast<double>(std::numeric_limits<float>::max());
  const double just_outside = std::nextafter(maximum, std::numeric_limits<double>::infinity());
  for (double direction : {-1., 1.}) {
    expect(checked_float(direction * maximum) == direction * maximum,
           "float maximum itself is representable");
    bool rejected = false;
    try { (void)checked_float(direction * just_outside); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "finite double just outside float range rejected before cast");
    for (int field = 0; field < 3; ++field) {
      for (bool extreme : {false, true}) {
        Adapter a;
        auto bad = sample();
        // Each finite sensor passes double validation but its scaled frame
        // value exceeds float max. No out-of-range float conversion occurs.
        const double multiplier = field == 0 ? 8. : (field == 1 ? 2. : 32.);
        const double value = direction * (extreme ? std::numeric_limits<double>::max()
                                                  : multiplier * maximum);
        expect(std::isfinite(value), "overflow fixture remains finite double");
        if (field == 0) bad.gyro[0] = value;
        if (field == 1) bad.q[0] = value;
        if (field == 2) bad.dq[0] = value;
        expect(a.reset(0) && a.receive_command({0., 0., 0.}, 0), "finite conversion setup");
        expect(!a.begin_tick(bad, 0), "finite oversized frame refused");
        expect(a.fault() == "FLOAT32_FRAME_OUT_OF_RANGE", "range failure explicit");
        stopped(a.torque_output(sample().q, sample().dq, 0));
      }
    }
    Adapter pd;
    const auto s = sample();
    const auto request = start(pd, s);
    expect(pd.complete_tick(request.token, Action{}, 1), "finite PD overflow setup");
    auto joints = s.q;
    joints[0] = direction * std::numeric_limits<double>::max();
    stopped(pd.torque_output(joints, s.dq, 1));
    expect(pd.fault() == "NONFINITE_PD_TORQUE", "finite sensors with overflowed PD torque refused");
  }
}

int main() {
  try {
    observation_and_mapping();
    nontrivial_history();
    deadlines_and_tokens();
    command_age_and_reset();
    invalid_values_and_limits();
    finite_conversion_boundaries();
    std::cout << "{\"status\":\"PASS\",\"assertions\":" << checks
              << ",\"physics_steps\":0,\"model_calls\":0,\"ros_nodes\":0}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
