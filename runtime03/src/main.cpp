#include "runtime03/interface.hpp"
#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>
#include <openssl/evp.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace d1rt03;
using Clock = std::chrono::steady_clock;
static double steady_seconds()
{
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
static Json read_json(const fs::path&p)
{
  std::ifstream s(p);
  if (!s)
    throw std::runtime_error("Cannot read " + p.string());
  Json j;
  s >> j;
  return j;
}
static void write_json(const fs::path&p, const Json&j)
{
  std::ofstream s(p);
  if (!s)
    throw std::runtime_error("Cannot write " + p.string());
  s << j.dump(2) << '\n';
  s.flush();
  if (!s)
    throw std::runtime_error("Write failed " + p.string());
}
static std::string sha256(const fs::path&p)
{
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot hash " + p.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA init");
  std::array<char, 65536>b{};
  while (f)
  {
    f.read(b.data(), b.size());
    if (f.gcount() && EVP_DigestUpdate(ctx.get(), b.data(), f.gcount()) != 1)
      throw std::runtime_error("SHA update");
  }
  unsigned char md[EVP_MAX_MD_SIZE];
  unsigned int n = 0;
  if (EVP_DigestFinal_ex(ctx.get(), md, &n) != 1)
    throw std::runtime_error("SHA final");
  std::ostringstream o;
  for (unsigned i = 0; i < n; ++i)
    o << std::hex << std::setw(2) << std::setfill('0') << int(md[i]);
  return o.str();
}
static void expected_hash(const std::string&actual, const std::string&expected, const char*what)
{
  if (expected.size() != 64 || actual != expected)
    throw std::runtime_error(std::string(what) + " SHA256 mismatch or expected hash missing");
}
static void signature(const Json&j)
{
  Json expected = {{"obs", {1, 57}}, {"history", {1, 10, 57}}};
  if (j.at("inputs") != expected || j.at("outputs") != Json{{"actions", {1, 16}}})
    throw std::runtime_error("ONNX signature requires float obs[1,57],history[1,10,57] -> actions[1,16]");
}
class Policy {
Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "d1_runtime03"};
std::unique_ptr<Ort::Session> session_;
public:
Json model_signature;
explicit Policy(const fs::path&p)
{
  if (std::string(OrtGetApiBase()->GetVersionString()) != "1.10.0")
    throw std::runtime_error("ORT must be exactly 1.10.0");
  Ort::SessionOptions opt;
  opt.SetIntraOpNumThreads(1);
  opt.SetInterOpNumThreads(1);
  session_ = std::make_unique<Ort::Session>(env_, p.c_str(), opt);
  Ort::AllocatorWithDefaultOptions alloc;
  model_signature = {{"inputs", Json::object()}, {"outputs", Json::object()}};
  if (session_->GetInputCount() != 2 || session_->GetOutputCount() != 1)
    throw std::runtime_error("ONNX input/output count");
  for (size_t i = 0; i < 2; ++i)
  {
    char*n = session_->GetInputName(i, alloc);
    // ORT 1.10 returns an unowned tensor view; retain the owning TypeInfo.
    auto type_info = session_->GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("ONNX input dtype");
    model_signature["inputs"][n] = info.GetShape();
    alloc.Free(n);
  }
  char*n = session_->GetOutputName(0, alloc);
  auto output_type_info = session_->GetOutputTypeInfo(0);
  auto info = output_type_info.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("ONNX output dtype");
  model_signature["outputs"][n] = info.GetShape();
  alloc.Free(n);
  signature(model_signature);
}
std::vector<float> infer(const Obs&o, const History&h)
{
  require_finite(o, "obs");
  require_finite(h, "history");
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2>oshape{1, 57};
  std::array<int64_t, 3>hshape{1, 10, 57};
  std::vector<Ort::Value>inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(o.data()), o.size(), oshape.data(), 2));
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(h.data()), h.size(), hshape.data(), 3));
  const char*names[] = {"obs", "history"};
  const char*outnames[] = {"actions"};
  auto out = session_->Run(Ort::RunOptions{}, names, inputs.data(), 2, outnames, 1);
  if (out.size() != 1 || out[0].GetTensorTypeAndShapeInfo().GetElementCount() != 16)
    throw std::runtime_error("ONNX returned wrong output shape");
  const float*x = out[0].GetTensorData<float>();
  std::vector<float>v(x, x + 16);
  require_finite(v, "ONNX output");
  return v;
}
};
struct Options {
  std::string model, policy, output, mode = "replay", sequence = "forward_stop", initial, actions, golden, onnx_golden;
  std::string schema = "/home/lyh/ddt_ros2_ws/runtime03/schema_pilot03.json", model_sha, policy_sha, schema_sha;
  double seconds = 12;
  int delay = 0, max_native = 15000;
  uint64_t seed = 101;
  bool self_test = false, validate_only = false;
};
static Options options(int argc, char**argv)
{
  Options o;
  for (int i = 1; i < argc; ++i)
  {
    std::string k = argv[i];
    if (k == "--self-test")
    {
      o.self_test = true;
      continue;
    }
    if (k == "--validate-only")
    {
      o.validate_only = true;
      continue;
    }
    if (k == "--help")
    {
      std::cout << "d1_runtime03 --schema JSON --model XML --policy ONNX --expected-model-sha256 SHA --expected-policy-sha256 SHA --expected-schema-sha256 SHA --output NEW_DIR [--seconds 12 --mode replay|live --sequence forward_stop|reverse_stop|turn_stop|JSON --delay-ticks 0..3 --seed 101 --max-native 15000 --initial-state JSON --actions JSON --validate-only]\nOffline: --self-test or --offline-golden JSON, --output NEW_DIR\nNative ONNX: --onnx-golden JSON --policy ONNX --expected-policy-sha256 SHA --output NEW_DIR\n";
      std::exit(0);
    }
    if (i + 1 >= argc)
      throw std::runtime_error("Missing value for " + k);
    std::string v = argv[++i];
    if (k == "--model")
      o.model = v;
    else if (k == "--policy")
      o.policy = v;
    else if (k == "--output")
      o.output = v;
    else if (k == "--mode")
      o.mode = v;
    else if (k == "--sequence")
      o.sequence = v;
    else if (k == "--initial-state")
      o.initial = v;
    else if (k == "--actions")
      o.actions = v;
    else if (k == "--offline-golden")
      o.golden = v;
    else if (k == "--onnx-golden")
      o.onnx_golden = v;
    else if (k == "--schema")
      o.schema = v;
    else if (k == "--expected-model-sha256")
      o.model_sha = v;
    else if (k == "--expected-policy-sha256")
      o.policy_sha = v;
    else if (k == "--expected-schema-sha256")
      o.schema_sha = v;
    else if (k == "--seconds")
      o.seconds = std::stod(v);
    else if (k == "--delay-ticks")
      o.delay = std::stoi(v);
    else if (k == "--seed")
      o.seed = std::stoull(v);
    else if (k == "--max-native")
      o.max_native = std::stoi(v);
    else
      throw std::runtime_error("Unknown option " + k);
  }
  if (o.output.empty())
    throw std::runtime_error("--output NEW_DIR required");
  if (!std::isfinite(o.seconds) || o.seconds <= 0 || o.seconds > 30 || o.max_native < 1 || o.max_native > 15000 || o.delay < 0 || o.delay > 3)
    throw std::runtime_error("Bound violation: seconds(0,30], native[1,15000],delay[0,3]");
  if (o.mode != "replay" && o.mode != "live")
    throw std::runtime_error("mode must be replay/live");
  return o;
}
static Json self_tests(const Config&cfg)
{
  Json checks = Json::object();
  auto rejects = [&](const char*n, auto fn){
                   try{fn();
                       checks[n] = false;
                   }catch (const std::exception&) {checks[n] = true;
                   }
                 };
  Watchdog w;
  bool stale = false;
  checks["watchdog_no_command_zero"] = w.value(1, .5) == V3{};
  w.receive({.2, 0, .1}, 10);
  checks["watchdog_fresh"] = w.value(10.4, .5) == V3({.2, 0, .1});
  checks["watchdog_boundary_fresh"] = w.value(10.5, .5) == V3({.2, 0, .1});
  checks["watchdog_stale_zero"] = w.value(10.5001, .5, &stale) == V3{} && stale;
  checks["clock_regression_zero"] = w.value(9, .5) == V3{};
  rejects("nan_action_rejected", [] {
    auto a = std::vector<float>(16, 0);
    a[3] = NAN;
    clip_action(a);
  });
  rejects("action_shape_rejected", [] {
    clip_action(std::vector<float>(15));
  });
  rejects("wrong_hash_rejected", [] {
    expected_hash(std::string(64, 'a'), std::string(64, 'b'), "test");
  });
  rejects("wrong_onnx_signature_rejected", [] {
    signature(Json{{"inputs", {{"obs", {1, 58}}, {"history", {1, 10, 57}}}}, {"outputs", {{"actions", {1, 16}}}}});
  });
  rejects("zero_quaternion_rejected", [] {
    inverse_rotate({0, 0, 0, 0}, {0, 0, -1});
  });
  Sample s;
  s.q = cfg.q0;
  Interface f(cfg);
  f.reset(s, 2);
  auto a = std::vector<float>(16, 2);
  auto first = f.accept(a);
  auto second = f.accept(a);
  auto third = f.accept(a);
  checks["delay_two_ticks"] = first == A16{} && second == A16{} && third[0] == 1;
  checks["last_action_is_issued_clipped"] = f.last_issued[0] == 1;
  f.reset(s, 2);
  checks["reset_clears_action_filter_queue"] = f.last_issued == A16{} && f.executed == A16{} && f.filtered == V3{} && f.queue.size() == 2 && f.queue.front() == A16{};
  auto t = pd(cfg, A16{}, J16{}, J16{});
  bool bounded = true;
  for (size_t i = 0; i < 16; ++i)
    bounded &= std::abs(t.value[i]) <= cfg.torque_limit[i];
  checks["pd_torque_bound"] = bounded;
  bool pass = true;
  for (auto&x:checks.items())
    pass &= x.value().get<bool>();
  return Json{{"status", pass?"PASS":"FAIL"}, {"physics_steps", 0}, {"checks", checks}, {"scope", "Pure runtime components; no ROS node, MuJoCo model, ONNX session, or dynamics started. Signature fault injection tests the production signature validator, not a synthetic corrupted ONNX file."}};
}
static Json offline_golden(const Config&cfg, const Json&fixture)
{
  Interface f(cfg);
  Json rows = Json::array();
  int delay = fixture.value("delay_ticks", 0);
  for (const auto&j:fixture.at("cases"))
  {
    auto s = sample_json(j);
    std::string op = j.at("op");
    if (op == "reset")
      f.reset(s, delay);
    else if (op == "observe")
      f.observe(s);
    else if (op == "action")
    {
      auto a = fixed<16, float>(j.at("action"), "action");
      f.accept(std::vector<float>(a.begin(), a.end()));
    }
    else
      throw std::runtime_error("offline op reset/observe/action required");
    auto torque = pd(cfg, f.executed, s.q, s.dq);
    rows.push_back(Json{{"label", j.value("label", std::string())}, {"op", op}, {"obs", f.current}, {"history", f.history}, {"filtered_gyro", f.filtered}, {"last_clipped_action", f.last_issued}, {"executed_action", f.executed}, {"torque", torque.value}, {"saturation", torque.saturated}});
  }
  return Json{{"status", "COMPLETED"}, {"physics_steps", 0}, {"cases", rows}};
}
static Json builtin_sequence(const std::string&name)
{
  auto segment = [](double start, double end, V3 command){
                   return Json{{"start", start}, {"end", end}, {"command", command}};
                 };
  Json segments = Json::array({segment(0, 2, {0, 0, 0})});
  if (name == "forward_stop" || name == "reverse_stop")
  {
    segments.push_back(segment(2, 6, {name == "forward_stop"?.2:-.2, 0, 0}));
    segments.push_back(segment(6, 30, {0, 0, 0}));
  }
  else if (name == "turn_stop")
  {
    segments.push_back(segment(2, 5, {0, 0, .2}));
    segments.push_back(segment(5, 6, {0, 0, 0}));
    segments.push_back(segment(6, 9, {0, 0, -.2}));
    segments.push_back(segment(9, 30, {0, 0, 0}));
  }
  else
    return read_json(name);
  return Json{{"name", name}, {"segments", segments}};
}
static void validate_sequence(const Json&sequence, double seconds)
{
  double previous = 0;
  for (const auto&seg:sequence.at("segments"))
  {
    double start = seg.at("start"), end = seg.at("end");
    auto c = fixed<3>(seg.at("command"), "segment command");
    if (!std::isfinite(start) || !std::isfinite(end) || std::abs(start - previous) > 1e-10 || end <= start || std::abs(c[0]) > .30000001 || std::abs(c[1]) > 1e-12 || std::abs(c[2]) > .30000001)
      throw std::runtime_error("sequence requires contiguous intervals and pilot03 vx/wz<=.3,vy=0");
    previous = end;
  }
  if (previous + 1e-10 < seconds)
    throw std::runtime_error("sequence does not cover requested seconds");
}
static V3 command_at(const Json&seq, double t)
{
  for (const auto&seg:seq.at("segments"))
    if (t + 1e-10 >= double(seg.at("start")) && t < double(seg.at("end")) - 1e-10)
      return fixed<3>(seg.at("command"), "command");
  return {};
}
struct Robot {
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)>model{nullptr, mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)>data{nullptr, mj_deleteData};
  std::array<int, 16>qadr{}, vadr{};
  std::array<int, 4>wheel_geoms{};
  int gyro_adr = -1, quat_adr = -1, vel_adr = -1, pos_adr = -1, robot_root = -1;
  double h_nom = 0;
  explicit Robot(const fs::path&path, const Config&cfg)
  {
    if (std::string(mj_versionString()) != "3.3.0")
      throw std::runtime_error("MuJoCo must be exactly 3.3.0");
    char error[2048]{};
    model.reset(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
    if (!model)
      throw std::runtime_error(std::string("MJCF load: ") + error);
    data.reset(mj_makeData(model.get()));
    if (!data)
      throw std::runtime_error("mj_makeData failed");
    if (model->nq != 23 || model->nv != 22 || model->nu != 16 || std::abs(model->opt.timestep - cfg.native_dt) > 1e-12)
      throw std::runtime_error("Expected pure D1 nq23/nv22/nu16 and .002s timestep");
    int free = mj_name2id(model.get(), mjOBJ_JOINT, "floating_base_joint");
    if (free < 0 || model->jnt_type[free] != mjJNT_FREE || model->jnt_qposadr[free] != 0 || model->jnt_dofadr[free] != 0)
      throw std::runtime_error("free base identity mismatch");
    robot_root = model->jnt_bodyid[free];
    for (int i = 0; i < 16; ++i)
    {
      int j = mj_name2id(model.get(), mjOBJ_JOINT, joint_names[i]);
      if (j < 0)
        throw std::runtime_error("missing joint");
      qadr[i] = model->jnt_qposadr[j];
      vadr[i] = model->jnt_dofadr[j];
      if (qadr[i] < 7 || vadr[i] < 6)
        throw std::runtime_error("joint/freebase mapping mismatch");
      if (wheel(i))
      {
        int body = model->jnt_bodyid[j], found = -1;
        for (int g = 0; g < model->ngeom; ++g)
          if (model->geom_bodyid[g] == body && model->geom_type[g] == mjGEOM_CYLINDER && (model->geom_contype[g] || model->geom_conaffinity[g]))
          {
            if (found >= 0)
              throw std::runtime_error("multiple wheel collision cylinders");
            found = g;
          }
        if (found < 0)
          throw std::runtime_error("wheel collision cylinder missing");
        wheel_geoms[i / 4] = found;
      }
    }
    auto sensor = [&](const char*n, int dim){
                    int id = mj_name2id(model.get(), mjOBJ_SENSOR, n);
                    if (id < 0 || model->sensor_dim[id] != dim)
                      throw std::runtime_error(std::string("sensor missing/mismatch ") + n);
                    return model->sensor_adr[id];
                  };
    gyro_adr = sensor("trunk_gyro", 3);
    quat_adr = sensor("trunk_quat", 4);
    vel_adr = sensor("frame_vel", 3);
    pos_adr = sensor("frame_pos", 3);
    mj_resetData(model.get(), data.get());
    data->qpos[2] = 0;
    data->qpos[3] = 1;
    data->qpos[4] = data->qpos[5] = data->qpos[6] = 0;
    for (int i = 0; i < 16; ++i)
      data->qpos[qadr[i]] = cfg.q0[i];
    refresh();
    h_nom = -minimum_wheel_bottom();
  }
  void refresh()
  {
    mj_forward(model.get(), data.get());
  }
  double minimum_wheel_bottom()const
  {
    double bottom = 1e9;
    for (int g:wheel_geoms)
    {
      const auto*R = data->geom_xmat + 9 * g;
      double radius = model->geom_size[3 * g], half = model->geom_size[3 * g + 1];
      double extent = radius * std::sqrt(R[6] * R[6] + R[7] * R[7]) + half * std::abs(R[8]);
      bottom = std::min(bottom, data->geom_xpos[3 * g + 2] - extent);
    }
    return bottom;
  }
  void reset(const Config&cfg, uint64_t seed, const std::string&initial)
  {
    mj_resetData(model.get(), data.get());
    std::vector<double> warmstart;
    if (!initial.empty())
    {
      auto j = read_json(initial);
      auto q = j.at("qpos").get<std::vector<double> >(), v = j.at("qvel").get<std::vector<double> >();
      if (q.size() != size_t(model->nq) || v.size() != size_t(model->nv))
        throw std::runtime_error("initial-state shape");
      require_finite(q, "initial qpos");
      require_finite(v, "initial qvel");
      std::copy(q.begin(), q.end(), data->qpos);
      std::copy(v.begin(), v.end(), data->qvel);
      Q4 quat{q[3], q[4], q[5], q[6]};
      double n = 0;
      for (auto x:quat)
        n += x * x;
      if (std::abs(n - 1) > 1e-8)
        throw std::runtime_error("initial quaternion must be unit");
      if (j.contains("qacc_warmstart"))
      {
        warmstart = j.at("qacc_warmstart").get<std::vector<double>>();
        if (warmstart.size() != size_t(model->nv)) throw std::runtime_error("initial warmstart shape");
        require_finite(warmstart, "initial qacc_warmstart");
      }
    }
    else
    {
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double>joint_noise(-.02, .02), yaw_noise(-.05, .05);
      double yaw = yaw_noise(rng);
      data->qpos[0] = data->qpos[1] = data->qpos[2] = 0;
      data->qpos[3] = std::cos(yaw / 2);
      data->qpos[4] = data->qpos[5] = 0;
      data->qpos[6] = std::sin(yaw / 2);
      for (size_t i = 0; i < 16; ++i)
        data->qpos[qadr[i]] = cfg.q0[i] + (wheel(i)?0:joint_noise(rng));
      refresh();
      data->qpos[2] = .002 - minimum_wheel_bottom();
    }
    apply(J16{});
    refresh();
    // mj_forward updates this solver state. Restore the recorded Python value
    // after the final static forward so the first integrator sees the same input.
    if (!warmstart.empty()) std::copy(warmstart.begin(), warmstart.end(), data->qacc_warmstart);
  }
  void apply(const J16&t)
  {
    mju_zero(data->qfrc_applied, model->nv);
    mju_zero(data->ctrl, model->nu);
    for (size_t i = 0; i < 16; ++i)
      data->qfrc_applied[vadr[i]] = t[i];
  }
  Sample sample(V3 cmd)const
  {
    Sample s;
    s.command = cmd;
    for (int i = 0; i < 3; ++i)
      s.gyro[i] = data->sensordata[gyro_adr + i];
    for (int i = 0; i < 4; ++i)
      s.quat[i] = data->sensordata[quat_adr + i];
    for (int i = 0; i < 16; ++i)
    {
      s.q[i] = data->qpos[qadr[i]];
      s.dq[i] = data->qvel[vadr[i]];
    }
    return s;
  }
  Json truth()const
  {
    // Current integration state, not the pre-integration sensordata left by
    // mj_step. This avoids extra mj_forward calls that could change warmstart.
    // The free joint is at the body origin; translation velocity is in world
    // axes and rotational qvel is in body axes. trunk_imu is colocated/aligned.
    V3 world_v{data->qvel[0], data->qvel[1], data->qvel[2]};
    V3 position{data->qpos[0], data->qpos[1], data->qpos[2]};
    Q4 q{data->qpos[3], data->qpos[4], data->qpos[5], data->qpos[6]};
    V3 bv = inverse_rotate(q, world_v), gravity = inverse_rotate(q, {0, 0, -1});
    double roll = std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
    double pitch = std::asin(std::clamp(2 * (q[0] * q[2] - q[3] * q[1]), -1., 1.));
    return Json{{"sim_time", data->time}, {"position", position}, {"quat_wxyz", q}, {"body_velocity", bv}, {"body_vx", bv[0]}, {"body_wz", data->qvel[5]}, {"roll", roll}, {"pitch", pitch}, {"tilt", std::acos(std::clamp(-gravity[2], -1., 1.))}, {"base_z", data->qpos[2]}, {"qpos", std::vector<double>(data->qpos, data->qpos + model->nq)}, {"qvel", std::vector<double>(data->qvel, data->qvel + model->nv)}};
  }
  void check_finite() const {
    for (int i=0;i<model->nq;++i) if(!std::isfinite(data->qpos[i])) throw std::runtime_error("nonfinite physics qpos");
    for (int i=0;i<model->nv;++i) if(!std::isfinite(data->qvel[i])||!std::isfinite(data->qacc[i])) throw std::runtime_error("nonfinite physics qvel/qacc");
    for (int i=0;i<mjNWARNING;++i) if(data->warning[i].number>0) throw std::runtime_error("MuJoCo numerical warning; trial stopped");
  }
  bool illegal_support_contact()const
  {
    auto robot_body = [this](int body) {
      for (int b = body; b > 0; b = model->body_parentid[b])
        if (b == robot_root) return true;
      return false;
    };
    for (int c = 0; c < data->ncon; ++c)
    {
      const auto&con = data->contact[c];
      bool w1 = std::find(wheel_geoms.begin(), wheel_geoms.end(), con.geom1) != wheel_geoms.end(), w2 = std::find(wheel_geoms.begin(), wheel_geoms.end(), con.geom2) != wheel_geoms.end();
      int b1 = model->geom_bodyid[con.geom1], b2 = model->geom_bodyid[con.geom2];
      bool robot1 = robot_body(b1), robot2 = robot_body(b2);
      if (robot1 == robot2)
        continue;
      int rg = robot1?con.geom1:con.geom2, terrain = robot1?con.geom2:con.geom1;
      if ((robot1?w1:w2) || (model->geom_contype[rg] == 0 && model->geom_conaffinity[rg] == 0))
        continue;
      int type = model->geom_type[terrain];
      if (type != mjGEOM_PLANE && type != mjGEOM_BOX && type != mjGEOM_HFIELD)
        continue;
      mjtNum force[6]{};
      mj_contactForce(model.get(), data.get(), c, force);
      if (force[0] > 1.)
        return true;
    }
    return false;
  }
};
class LiveNode : public rclcpp::Node {
rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_;
rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detail_;
public: Watchdog watchdog;
uint64_t invalid_commands = 0;
LiveNode() : Node("d1_selftrained_runtime03")
{
  sub_ = create_subscription<geometry_msgs::msg::Twist>("/d1_selftrained/cmd_vel", 10, [this](geometry_msgs::msg::Twist::ConstSharedPtr m){
      V3 c{m->linear.x, m->linear.y, m->angular.z};
      try{require_finite(c, "Twist");
          c[0] = std::clamp(c[0], -.3, .3);
          c[1] = 0;
          c[2] = std::clamp(c[2], -.3, .3);
          watchdog.receive(c, steady_seconds());
      }catch (const std::exception&) {++invalid_commands;
                                      watchdog.receive({}, steady_seconds());
      }
    });
  odom_ = create_publisher<nav_msgs::msg::Odometry>("/d1_selftrained/odometry", 10);
  detail_ = create_publisher<std_msgs::msg::String>("/d1_selftrained/truth", 10);
}
void publish_truth(const Json&t)
{
  nav_msgs::msg::Odometry o;
  o.header.stamp = rclcpp::Time(int64_t(double(t.at("sim_time")) * 1e9));
  o.header.frame_id = "world";
  o.child_frame_id = "base_link";
  auto p = t.at("position").get<V3>();
  auto q = t.at("quat_wxyz").get<Q4>();
  auto v = t.at("body_velocity").get<V3>();
  o.pose.pose.position.x = p[0];
  o.pose.pose.position.y = p[1];
  o.pose.pose.position.z = p[2];
  o.pose.pose.orientation.w = q[0];
  o.pose.pose.orientation.x = q[1];
  o.pose.pose.orientation.y = q[2];
  o.pose.pose.orientation.z = q[3];
  o.twist.twist.linear.x = v[0];
  o.twist.twist.linear.y = v[1];
  o.twist.twist.linear.z = v[2];
  o.twist.twist.angular.z = t.at("body_wz");
  odom_->publish(o);
  std_msgs::msg::String s;
  s.data = t.dump();
  detail_->publish(s);
}
};
static Json latency_stats(std::vector<double>x)
{
  if (x.empty())
    return Json{{"count", 0}}
  ;
  std::sort(x.begin(), x.end());
  auto p = [&](double q){
             return x[size_t(std::ceil(q * (x.size() - 1)))];
           };
  return Json{{"count", x.size()}, {"unit", "milliseconds"}, {"p50", p(.5)}, {"p95", p(.95)}, {"p99", p(.99)}, {"max", x.back()}};
}
int main(int argc, char**argv)
{
  Json report = {{"status", "STARTING"}, {"native_steps_attempted", 0}, {"native_steps_completed", 0}, {"pd_updates", 0}, {"policy_updates", 0}};
  Options o;
  fs::path output;
  std::unique_ptr<Robot>robot;
  std::unique_ptr<Interface>iface;
  int counter_fd = -1;
  bool ros_started = false, owns_output = false;
  uint64_t attempted = 0, completed = 0, pd_count = 0, policy_count = 0, policy_attempts = 0, onnx_attempts = 0, watchdog_stale_ticks = 0;
  double started = steady_seconds();
  try{
    o = options(argc, argv);
    output = fs::absolute(o.output);
    if (!fs::create_directory(output))
      throw std::runtime_error("Output directory exists or parent missing; refusing overwrite");
    owns_output = true;
    auto schema = read_json(o.schema);
    Config cfg(schema);
    report["schema_sha256"] = sha256(o.schema);
    report["runtime_binary_sha256"] = sha256("/proc/self/exe");
    report["schema"] = schema;
    report["mujoco_version"] = mj_versionString();
    report["onnxruntime_version"] = OrtGetApiBase()->GetVersionString();
    report["physics_authorization_note"] = "The executable itself does not authorize a trial; the root agent must release the bounded trial and account its steps.";
    if (o.self_test)
    {
      report.update(self_tests(cfg));
      write_json(output / "report.json", report);
      std::cout << report.dump(2) << '\n';
      return report["status"] == "PASS"?0:1;
    }
    if (!o.golden.empty())
    {
      report.update(offline_golden(cfg, read_json(o.golden)));
      report["fixture_sha256"] = sha256(o.golden);
      write_json(output / "report.json", report);
      std::cout << "Offline golden complete; physics steps 0\n";
      return 0;
    }
    if (!o.onnx_golden.empty()) {
      if (o.policy.empty()) throw std::runtime_error("--onnx-golden requires --policy");
      expected_hash(sha256(o.policy), o.policy_sha, "policy");
      Policy policy(o.policy);
      const auto fixtures = read_json(o.onnx_golden);
      Json comparisons = Json::array();
      double max_error = 0;
      for (const auto& item : fixtures.at("fixtures")) {
        Obs obs = fixed<57, float>(item.at("obs"), "golden obs");
        const auto rows = item.at("history_oldest_to_newest");
        if (!rows.is_array() || rows.size() != 10) throw std::runtime_error("golden history shape");
        History history{};
        for (size_t i = 0; i < 10; ++i) {
          auto row = fixed<57, float>(rows.at(i), "golden history row");
          std::copy(row.begin(), row.end(), history.begin() + i * 57);
        }
        auto expected = fixed<16, float>(item.at("torch_actions"), "golden Torch actions");
        auto actual = policy.infer(obs, history);
        double err = 0;
        for (size_t i = 0; i < 16; ++i) err = std::max(err, std::abs(double(actual[i]) - expected[i]));
        max_error = std::max(max_error, err);
        comparisons.push_back(Json{{"index", item.at("index")}, {"actions", actual}, {"max_absolute_error", err}, {"passed", err <= 1e-5}});
      }
      if (comparisons.empty()) throw std::runtime_error("empty ONNX golden fixtures");
      report["status"] = max_error <= 1e-5 ? "PASS" : "FAIL";
      report["policy_sha256"] = sha256(o.policy);
      report["fixture_sha256"] = sha256(o.onnx_golden);
      report["physics_steps"] = 0;
      report["onnx_signature"] = policy.model_signature;
      report["onnx_inference_calls"] = comparisons.size();
      report["comparisons"] = comparisons;
      report["max_absolute_error"] = max_error;
      report["absolute_tolerance"] = 1e-5;
      write_json(output / "report.json", report);
      std::cout << "Native ORT1.10 golden parity: " << report["status"] << ", max abs error " << max_error << ", physics 0\n";
      return max_error <= 1e-5 ? 0 : 1;
    }
    expected_hash(report["schema_sha256"], o.schema_sha, "schema");
    expected_hash(sha256(schema.at("audited_interface_schema").get<std::string>()),
                  schema.at("audited_interface_schema_sha256"), "audited interface schema");
    expected_hash(sha256(schema.at("training_config").get<std::string>()),
                  schema.at("training_config_sha256"), "training config");
    if (schema.contains("training_interface_schema"))
      expected_hash(sha256(schema.at("training_interface_schema").get<std::string>()),
                    schema.at("training_interface_schema_sha256"), "training interface schema");
    if (o.model.empty())
      throw std::runtime_error("--model required");
    report["model_sha256"] = sha256(o.model);
    expected_hash(report["model_sha256"], o.model_sha, "model");
    report["robot_xml_sha256"] = sha256(schema.at("robot_xml").get<std::string>());
    expected_hash(report["robot_xml_sha256"], schema.at("robot_xml_sha256"), "robot XML");
    Json sequence = builtin_sequence(o.sequence);
    validate_sequence(sequence, o.seconds);
    write_json(output / "sequence.json", sequence);
    std::unique_ptr<Policy>policy;
    Json replay_actions;
    if (!o.actions.empty())
    {
      if (o.mode != "replay" || !o.policy.empty())
        throw std::runtime_error("--actions requires replay and no --policy");
      replay_actions = read_json(o.actions).at("actions");
      if (!replay_actions.is_array() || replay_actions.empty())
        throw std::runtime_error("empty action fixture");
      for (const auto&a:replay_actions)
        fixed<16, float>(a, "action fixture");
      report["action_fixture_sha256"] = sha256(o.actions);
    }
    else
    {
      if (o.policy.empty())
        throw std::runtime_error("--policy required");
      report["policy_sha256"] = sha256(o.policy);
      expected_hash(report["policy_sha256"], o.policy_sha, "policy");
      policy = std::make_unique<Policy>(o.policy);
      report["onnx_signature"] = policy->model_signature;
    }
    if (!o.initial.empty()) {
      const auto supplied = read_json(o.initial);
      if (supplied.contains("scene_xml_sha256")) expected_hash(o.model_sha, supplied.at("scene_xml_sha256"), "initial scene");
      if (supplied.contains("robot_xml_sha256")) expected_hash(report["robot_xml_sha256"], supplied.at("robot_xml_sha256"), "initial robot");
      if (supplied.contains("interface_schema_sha256"))
        expected_hash(schema.at(schema.contains("training_interface_schema_sha256") ? "training_interface_schema_sha256" : "audited_interface_schema_sha256"),
                      supplied.at("interface_schema_sha256"), "initial interface");
      if (supplied.value("episode_policy_tick", 0) != 0 || std::abs(supplied.value("simulation_time_s", 0.)) > 1e-12)
        throw std::runtime_error("initial-state supports fresh-reset qpos/qvel, not mid-episode restoration");
      if (supplied.value("delay_ticks", o.delay) != o.delay) throw std::runtime_error("initial-state delay differs from CLI");
    }
    robot = std::make_unique<Robot>(o.model, cfg);
    robot->reset(cfg, o.seed, o.initial);
    report["seed"] = o.seed;
    report["reset_rng"] = "std::mt19937_64: matching Python seed alone does not imply matching state; use --initial-state for cross-language equality";
    report["initial_state_override"] = !o.initial.empty();
    if (!o.initial.empty())
      report["initial_state_sha256"] = sha256(o.initial);
    report["initial_qacc_warmstart_restored"] = !o.initial.empty() && read_json(o.initial).contains("qacc_warmstart");
    Json initial = {{"qpos", std::vector<double>(robot->data->qpos, robot->data->qpos + robot->model->nq)}, {"qvel", std::vector<double>(robot->data->qvel, robot->data->qvel + robot->model->nv)}, {"command", command_at(sequence, 0)}, {"delay_ticks", o.delay}, {"model_sha256", report["model_sha256"]}, {"schema_sha256", report["schema_sha256"]}, {"h_nom", robot->h_nom}};
    initial["qacc_warmstart"] = std::vector<double>(robot->data->qacc_warmstart, robot->data->qacc_warmstart + robot->model->nv);
    report["h_nom"] = robot->h_nom;
    iface = std::make_unique<Interface>(cfg);
    iface->reset(robot->sample(o.mode == "live"?V3{}:command_at(sequence, 0)), o.delay);
    initial["current_observation_57"] = iface->current;
    initial["history_oldest_to_newest_flat570"] = iface->history;
    initial["filtered_gyro_body"] = iface->filtered;
    initial["last_clipped_policy_action"] = iface->last_issued;
    initial["executed_action"] = iface->executed;
    write_json(output / "initial_state.json", initial);
    if (o.validate_only)
    {
      report["status"] = "VALIDATED_ZERO_PHYSICS";
      report["onnx_inference_calls"] = 0;
      write_json(output / "report.json", report);
      std::cout << "Identity, model and ONNX shapes validated; physics steps 0\n";
      return 0;
    }
    uint64_t planned = uint64_t(std::floor(o.seconds / cfg.native_dt + 1e-9));
    if (planned < 1)
      throw std::runtime_error("seconds shorter than one native step");
    uint64_t budget = std::min(planned, uint64_t(o.max_native));
    report["planned_native_cap"] = budget;
    report["seconds_requested"] = o.seconds;
    report["mode"] = o.mode;
    report["delay_ticks"] = o.delay;
    report["inference_warmup_calls"] = 0;
    if (policy)
    {
      for (int i = 0; i < 3; ++i)
      {
        clip_action(policy->infer(iface->current, iface->history));
        report["inference_warmup_calls"] = i + 1;
      }
    }
    std::shared_ptr<LiveNode>node;
    if (o.mode == "live")
    {
      int ros_argc = 1;
      char*ros_argv[] = {argv[0]};
      rclcpp::init(ros_argc, ros_argv);
      ros_started = true;
      node = std::make_shared<LiveNode>();
    }
    std::ofstream trace(output / "trace.jsonl");
    if (!trace)
      throw std::runtime_error("trace open failed");
    counter_fd = ::open((output / "native_step_counter.bin").c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (counter_fd < 0)
      throw std::runtime_error("counter open failed");
    std::array<uint64_t, 2>counts{0, 0};
    if (::pwrite(counter_fd, counts.data(), sizeof(counts), 0) != ssize_t(sizeof(counts)))
      throw std::runtime_error("counter initialize failed");
    std::vector<double>latencies, pd_times, policy_times;
    std::vector<double>live_lateness;
    Torque held;
    V3 command{};
    bool illegal_since_policy = false;
    std::string terminal = "TIME_LIMIT";
    auto wall_start = Clock::now();
    double last_latency = 0;
    write_json(output / "report.json", report);
    for (uint64_t tick = 0; tick < budget; ++tick)
    {
      if (node)
      {
        rclcpp::spin_some(node);
        if (!rclcpp::ok())
        {
          terminal = "ROS_SHUTDOWN";
          break;
        }
        auto target = wall_start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(tick * cfg.native_dt));
        std::this_thread::sleep_until(target);
        live_lateness.push_back(std::max(0., std::chrono::duration<double, std::milli>(Clock::now() - target).count()));
      }
      if (tick % cfg.policy_decimation == 0)
      {
        // D1Env._observe performs one forward at each policy boundary; reset
        // already supplied tick 0. Match that sequence exactly, including the
        // final post-transition forward, rather than forwarding per native step.
        if (tick > 0) robot->refresh();
        robot->check_finite();
        illegal_since_policy |= robot->illegal_support_contact();
        auto truth = robot->truth();
        if (tick > 0 && (double(truth["tilt"]) > .6 || double(truth["base_z"]) < robot->h_nom - .18 || illegal_since_policy))
        {
          terminal = illegal_since_policy?"ILLEGAL_BODY_SUPPORT":"FALL";
          break;
        }
        illegal_since_policy = false;
        bool stale = false;
        command = node?node->watchdog.value(steady_seconds(), cfg.watchdog_seconds, &stale):command_at(sequence, tick * cfg.native_dt);
        if (stale)
          ++watchdog_stale_ticks;
        if (tick == 0)
        {
          iface->current[6] = float(command[0] * 2);
          iface->current[7] = float(command[1] * 2);
          iface->current[8] = float(command[2] * .25);
        }
        else
          iface->observe(robot->sample(command));
        std::vector<float>action;
        auto began = Clock::now();
        ++policy_attempts;
        if (policy)
        {
          ++onnx_attempts;
          action = policy->infer(iface->current, iface->history);
        }
        else
        {
          if (policy_count >= replay_actions.size())
          {
            terminal = "ACTION_FIXTURE_END";
            break;
          }
          action = replay_actions.at(policy_count).get<std::vector<float> >();
        }
        last_latency = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
        if (policy)
          latencies.push_back(last_latency);
        if (policy && last_latency > 20.)
        {
          iface->fallback();
          terminal = "INFERENCE_DEADLINE";
          break;
        }
        iface->accept(action);
        ++policy_count;
        policy_times.push_back(tick * cfg.native_dt);
      }
      if (tick % cfg.pd_decimation == 0)
      {
        auto s = robot->sample(command);
        held = pd(cfg, iface->executed, s.q, s.dq);
        ++pd_count;
        pd_times.push_back(tick * cfg.native_dt);
      }
      robot->apply(held.value);
      ++attempted;
      counts = {attempted, completed};
      if (::pwrite(counter_fd, counts.data(), sizeof(counts), 0) != ssize_t(sizeof(counts)))
        throw std::runtime_error("step counter write failed before physics");
      mj_step(robot->model.get(), robot->data.get());
      ++completed;
      counts = {attempted, completed};
      if (::pwrite(counter_fd, counts.data(), sizeof(counts), 0) != ssize_t(sizeof(counts)))
        throw std::runtime_error("step counter write failed after physics");
      robot->check_finite();
      illegal_since_policy |= robot->illegal_support_contact();
      auto row = robot->truth();
      row["native_count"] = completed;
      row["pd_count"] = pd_count;
      row["policy_count"] = policy_count;
      row["command"] = command;
      row["issued_action"] = iface->last_issued;
      row["executed_action"] = iface->executed;
      row["torque"] = held.value;
      row["raw_torque"] = held.raw;
      row["saturation"] = held.saturated;
      row["policy_latency_ms"] = last_latency;
      row["illegal_body_support_since_policy"] = illegal_since_policy;
      trace << row.dump() << '\n';
      if (tick % 10 == 9)
      {
        trace.flush();
        if (!trace)
          throw std::runtime_error("trace write failed");
        if (node)
          node->publish_truth(row);
      }
    }
    // The boundary was already refreshed if the loop stopped before integration.
    if (completed == budget || (completed > 0 && completed % cfg.policy_decimation != 0)) robot->refresh();
    robot->check_finite();
    illegal_since_policy |= robot->illegal_support_contact();
    auto final_truth = robot->truth();
    if (terminal == "TIME_LIMIT" && (double(final_truth["tilt"]) > .6 || double(final_truth["base_z"]) < robot->h_nom - .18 || illegal_since_policy))
      terminal = illegal_since_policy?"ILLEGAL_BODY_SUPPORT":"FALL";
    if (completed == budget && budget < planned && terminal == "TIME_LIMIT")
      terminal = "NATIVE_BUDGET_LIMIT";
    iface->fallback();
    robot->apply(pd(cfg, iface->executed, robot->sample({}).q, robot->sample({}).dq).value);
    report["status"] = (terminal == "TIME_LIMIT" || terminal == "NATIVE_BUDGET_LIMIT" || terminal == "ACTION_FIXTURE_END" || terminal == "ROS_SHUTDOWN")?"COMPLETED":"FAILED";
    report["terminal_reason"] = terminal;
    report["final_truth"] = final_truth;
    report["inference_latency"] = latency_stats(latencies);
    report["live_wall_lateness"] = latency_stats(live_lateness);
    report["pd_update_sim_times"] = pd_times;
    report["policy_update_sim_times"] = policy_times;
    report["native_dt"] = cfg.native_dt;
    report["pd_dt"] = cfg.native_dt * cfg.pd_decimation;
    report["policy_dt"] = cfg.native_dt * cfg.policy_decimation;
    report["watchdog_stale_policy_ticks"] = watchdog_stale_ticks;
    report["invalid_live_commands"] = node?node->invalid_commands:0;
    report["post_stop_fallback"] = "zero issued/executed action and cleared FIFO; bounded standing PD assigned, simulation stopped with zero additional physics steps; physical recovery not claimed";
  }catch (const std::exception&e) {report["status"] = "FAILED";
                                   report["error"] = e.what();
                                   if (iface)
                                     iface->fallback();
                                   if (robot)
                                     robot->apply(J16{});
                                   report["fault_fallback"] = "simulation stopped; action queue zeroed; applied generalized force zeroed; no additional dynamics step";
  }
  report["native_steps_attempted"] = attempted;
  report["native_steps_completed"] = completed;
  report["pd_updates"] = pd_count;
  report["policy_updates"] = policy_count;
  report["policy_action_tick_attempts"] = policy_attempts;
  report["policy_inference_attempts"] = onnx_attempts;
  report["wall_seconds"] = steady_seconds() - started;
  if (counter_fd >= 0)
  {
    ::fsync(counter_fd);
    ::close(counter_fd);
  }
  if (ros_started)
    rclcpp::shutdown();
  if (owns_output)
    try{write_json(output / "report.json", report);
    }catch (const std::exception&e) {std::cerr << e.what() << '\n';
    }
  std::cout << report.dump(2) << '\n';
  return report["status"] == "COMPLETED"?0:1;
}
