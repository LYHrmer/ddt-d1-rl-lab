// C34 yaw reference state plus CPU actor. Uses the version19 wire protocol.
// Each server instance owns exactly one episode; no robot or physics is constructed.
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef D1_ACTOR34_SOURCE_SHA256
#error "Compile with the SHA256 of this source in D1_ACTOR34_SOURCE_SHA256"
#endif

using Json = nlohmann::json;
namespace fs = std::filesystem;

static void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

static std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot hash file");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1, "SHA initialization failed");
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    require(!input.bad(), "SHA file read failed");
    if (input.gcount()) require(EVP_DigestUpdate(context.get(), buffer.data(), input.gcount()) == 1, "SHA update failed");
  }
  unsigned char bytes[EVP_MAX_MD_SIZE];
  unsigned size = 0;
  require(EVP_DigestFinal_ex(context.get(), bytes, &size) == 1, "SHA finalization failed");
  std::ostringstream text;
  for (unsigned i = 0; i < size; ++i) text << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
  return text.str();
}

static std::string input_sha(const std::array<float, 627>& values) {
  unsigned char bytes[EVP_MAX_MD_SIZE];
  unsigned size = 0;
  require(EVP_Digest(values.data(), sizeof(float)*values.size(), bytes, &size,
                     EVP_sha256(), nullptr) == 1, "input SHA failed");
  std::ostringstream text;
  for (unsigned i = 0; i < size; ++i)
    text << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
  return text.str();
}

static bool is_sha(const std::string& text) {
  return text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static bool line(std::string& result) {
  result.clear();
  char value;
  while (std::cin.get(value)) {
    if (value == '\n') return true;
    require(result.size() < 65536, "input frame exceeds 65536 bytes");
    result.push_back(value);
  }
  require(std::cin.eof() && result.empty(), "incomplete JSONL frame at EOF");
  return false;
}

static void emit(const Json& value) {
  std::cout << value.dump() << '\n' << std::flush;
  require(bool(std::cout), "stdout protocol write failed");
}

class ReferenceShaper34 {
  bool initialized_ = false;
  double state_ = 0.;
  std::array<float, 10> history_{};
  std::array<float, 10> raw_history_{};
public:
  void apply(std::array<float, 627>& observation) {
    std::array<float, 11> raw{}, target{};
    for (size_t i = 0; i < 11; ++i) {
      raw[i] = observation[i*57+8];
      const float multiplied = raw[i]*1.5f;
      target[i] = std::max(-.25f, std::min(.25f, multiplied));
    }
    if (!initialized_) {
      for (float value : raw) require(value == raw[0], "reset requires constant prehistory");
      state_ = target[0];
      for (size_t i = 0; i < 10; ++i) {
        history_[i] = target[i+1];
        raw_history_[i] = raw[i+1];
      }
      initialized_ = true;
    }
    for (size_t i = 0; i < 10; ++i)
      require(raw[i+1] == raw_history_[i], "raw history differs from actual previous inputs");
    const double alpha = std::exp(-.2);
    state_ = alpha*state_ + (1.-alpha)*double(target[0]);
    const float current = static_cast<float>(state_);
    observation[8] = current;
    for (size_t i = 0; i < 10; ++i) observation[(i+1)*57+8] = history_[i];
    for (size_t i = 0; i < 9; ++i) {
      history_[i] = history_[i+1];
      raw_history_[i] = raw_history_[i+1];
    }
    history_[9] = current;
    raw_history_[9] = raw[0];
  }
};

class Actor {
  Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "d1_actor34"};
  std::unique_ptr<Ort::Session> session_;
public:
  Json signature;
  explicit Actor(const std::string& path) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_ = std::make_unique<Ort::Session>(environment_, path.c_str(), options);
    require(session_->GetInputCount() == 2 && session_->GetOutputCount() == 1, "wrong ONNX input/output count");
    Ort::AllocatorWithDefaultOptions allocator;
    signature = {{"inputs", Json::object()}, {"outputs", Json::object()}};
    for (size_t index = 0; index < 2; ++index) {
      char* name = session_->GetInputName(index, allocator);
      std::string owned_name(name);
      allocator.Free(name);
      auto type = session_->GetInputTypeInfo(index);
      auto tensor = type.GetTensorTypeAndShapeInfo();
      require(tensor.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "ONNX input must be float32");
      signature["inputs"][owned_name] = tensor.GetShape();
    }
    char* name = session_->GetOutputName(0, allocator);
    std::string owned_name(name);
    allocator.Free(name);
    auto type = session_->GetOutputTypeInfo(0);
    auto tensor = type.GetTensorTypeAndShapeInfo();
    require(tensor.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "ONNX output must be float32");
    signature["outputs"][owned_name] = tensor.GetShape();
    require(signature == Json{{"inputs", {{"obs", {1, 57}}, {"history", {1, 10, 57}}}},
                              {"outputs", {{"actions", {1, 16}}}}}, "wrong fixed ONNX signature");
  }

  std::vector<float> infer(std::array<float, 627>& actor, uint64_t& attempted, uint64_t& returned) {
    const std::array<int64_t, 2> obs_shape{1, 57};
    const std::array<int64_t, 3> history_shape{1, 10, 57};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, actor.data(), 57, obs_shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, actor.data()+57, 570, history_shape.data(), 3));
    const char* names[] = {"obs", "history"};
    const char* outputs[] = {"actions"};
    ++attempted;
    auto values = session_->Run(Ort::RunOptions{}, names, inputs.data(), 2, outputs, 1);
    ++returned;
    require(values.size() == 1 && values[0].IsTensor(), "ONNX returned a non-tensor output");
    auto info = values[0].GetTensorTypeAndShapeInfo();
    require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && info.GetShape() == std::vector<int64_t>{1, 16}, "ONNX returned the wrong shape or dtype");
    const float* data = values[0].GetTensorData<float>();
    std::vector<float> result(data, data+16);
    for (float value : result) require(std::isfinite(value) && value >= -1.f && value <= 1.f,
                                      "ONNX returned nonfinite or unclipped actions");
    return result;
  }
};

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  uint64_t requests = 0, attempted = 0, returned = 0;
  try {
    std::string policy, expected_policy, expected_source;
    uint64_t maximum = 0;
    for (int index = 1; index < argc; index += 2) {
      require(index+1 < argc, "every option requires a value");
      const std::string key(argv[index]), value(argv[index+1]);
      if (key == "--policy" && policy.empty()) policy = value;
      else if (key == "--expected-model-sha256" && expected_policy.empty()) expected_policy = value;
      else if (key == "--expected-source-sha256" && expected_source.empty()) expected_source = value;
      else if (key == "--max-requests" && maximum == 0) {
        require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid request cap");
        maximum = std::stoull(value);
      } else throw std::runtime_error("unknown or duplicate option");
    }
    require(!policy.empty() && is_sha(expected_policy) && is_sha(expected_source)
            && maximum > 0 && maximum <= 21664, "explicit model/source identity and bounded request cap required");
    require(expected_source == D1_ACTOR34_SOURCE_SHA256, "compiled source SHA differs");
    require(sha256(policy) == expected_policy, "model SHA differs");
    require(std::string(OrtGetApiBase()->GetVersionString()) == "1.10.0", "ORT must be exactly 1.10.0");
    Dl_info library{};
    require(dladdr(reinterpret_cast<void*>(OrtGetApiBase), &library) != 0 && library.dli_fname,
            "actual ORT library identity unavailable");
    Actor actor(policy);
    ReferenceShaper34 shaper;
    emit({{"format", "d1-cpp-actor19-ready-v1"}, {"status", "READY"},
          {"source_sha256", D1_ACTOR34_SOURCE_SHA256}, {"binary_sha256", sha256(fs::canonical("/proc/self/exe"))},
          {"model_sha256", expected_policy}, {"ort_library_sha256", sha256(fs::canonical(library.dli_fname))},
          {"onnxruntime_version", "1.10.0"}, {"signature", actor.signature}, {"cpu_threads", 1},
          {"reference_shaping", {{"gain", 1.5}, {"tau_s", .1}, {"dt_s", .02}, {"reset", "new_process"}}},
          {"max_requests", maximum}, {"inference_attempts", 0}, {"inference_returns", 0},
          {"robot_models_constructed", 0}, {"physics_steps_executed", 0}});
    std::string frame;
    while (line(frame)) {
      require(requests < maximum, "request cap exceeded");
      const Json request = Json::parse(frame);
      require(request.is_object() && request.size() == 3
              && request.at("format") == "d1-cpp-actor19-request-v1"
              && request.at("seq").is_number_unsigned() && request.at("seq").get<uint64_t>() == requests,
              "request format or sequence differs");
      const auto& input = request.at("actor");
      require(input.is_array() && input.size() == 627, "actor must contain exactly 627 floats");
      std::array<float, 627> packed{};
      for (size_t index = 0; index < packed.size(); ++index) {
        require(input[index].is_number(), "actor values must be numeric, not boolean");
        const double value = input[index].get<double>();
        require(std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max(), "actor is not finite float32");
        packed[index] = static_cast<float>(value);
      }
      shaper.apply(packed);
      std::array<float, 11> effective_commands{};
      for (size_t i = 0; i < 11; ++i) effective_commands[i] = packed[i*57+8];
      const auto effective_input_hash = input_sha(packed);
      auto actions = actor.infer(packed, attempted, returned);
      emit({{"format", "d1-cpp-actor19-response-v1"}, {"status", "OK"}, {"seq", requests},
            {"actions", actions}, {"effective_yaw_commands", effective_commands},
            {"effective_input_sha256", effective_input_hash}, {"inference_attempts", attempted}, {"inference_returns", returned}});
      ++requests;
    }
    emit({{"format", "d1-cpp-actor19-closed-v1"}, {"status", "CLOSED"}, {"requests", requests},
          {"inference_attempts", attempted}, {"inference_returns", returned},
          {"robot_models_constructed", 0}, {"physics_steps_executed", 0}});
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    try { emit({{"format", "d1-cpp-actor19-failure-v1"}, {"status", "FAIL"}, {"error", error.what()},
                {"accepted_requests", requests}, {"inference_attempts", attempted}, {"inference_returns", returned}}); }
    catch (...) {}
    return 2;
  }
}
