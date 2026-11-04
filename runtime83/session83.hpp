#pragma once
// Opt-in synthetic integration only. Owns one serialized adapter and child.
// No hardware SDK, robot model, physics, ROS or model loader is linked here.
#include "runtime82/adapter82.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <poll.h>
#include <set>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;
namespace d1_adapter83 {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using d1_adapter82::Sample;
using d1_adapter82::Vector3;
constexpr std::int64_t kDeadline = d1_adapter82::kDeadlineNs;
inline std::int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void require(bool okay, const std::string& why) {
  if (!okay) throw std::runtime_error(why);
}
inline bool is_sha(const std::string& s) {
  return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
inline std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot open SHA input: " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA init");
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    require(!input.bad(), "SHA read");
    if (input.gcount()) require(EVP_DigestUpdate(ctx.get(), buffer.data(), input.gcount()) == 1, "SHA update");
  }
  unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned size = 0;
  require(EVP_DigestFinal_ex(ctx.get(), bytes, &size) == 1, "SHA finish");
  std::ostringstream text;
  for (unsigned i = 0; i < size; ++i) text << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
  return text.str();
}
inline Json parse(const std::string& text) {
  require(text.size() <= 65536, "protocol line too large");
  std::vector<std::set<std::string>> keys;
  auto callback = [&keys](int, Json::parse_event_t event, Json& value) {
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key)
      require(!keys.empty() && keys.back().insert(value.get<std::string>()).second, "duplicate JSON key");
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  return Json::parse(text, callback);
}
inline void write_json(const fs::path& path, const Json& value) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  require(bool(out), "cannot open receipt");
  out << value.dump(2) << '\n'; out.flush();
  require(bool(out), "receipt write failed"); out.close(); fs::rename(temporary, path);
}

struct Config {
  fs::path binary, model, source, ort_library;
  std::string binary_sha256, model_sha256, source_sha256, ort_library_sha256;
  unsigned max_requests = 0;  // Whole engine lifetime, including reset sessions.
  static Config read(const fs::path& manifest) {
    std::ifstream in(manifest); require(bool(in), "manifest missing");
    std::string text((std::istreambuf_iterator<char>(in)), {});
    const auto j = parse(text);
    require(j.at("schema") == "d1-adapter83-runtime-v1", "manifest schema");
    Config c;
    c.binary = j.at("binary").get<std::string>(); c.model = j.at("model").get<std::string>();
    c.source = j.at("source").get<std::string>(); c.ort_library = j.at("ort_library").get<std::string>();
    c.binary_sha256 = j.at("binary_sha256"); c.model_sha256 = j.at("model_sha256");
    c.source_sha256 = j.at("source_sha256"); c.ort_library_sha256 = j.at("ort_library_sha256");
    require(j.at("max_requests").is_number_unsigned(), "integer request cap required");
    require(j.at("max_requests").get<std::uint64_t>() <= 300, "request cap exceeds 300");
    c.max_requests = j.at("max_requests").get<unsigned>(); c.verify(); return c;
  }
  void verify() const {
    require(max_requests > 0 && max_requests <= 300, "request cap outside [1,300]");
    for (const auto& item : std::vector<std::pair<fs::path, std::string>>{
             {binary, binary_sha256}, {model, model_sha256}, {source, source_sha256}, {ort_library, ort_library_sha256}})
      require(is_sha(item.second) && fs::is_regular_file(item.first) && sha256(item.first) == item.second,
              "runtime identity mismatch: " + item.first.string());
  }
  Json identities() const {
    return {{"binary_sha256", binary_sha256}, {"model_sha256", model_sha256},
            {"source_sha256", source_sha256}, {"ort_library_sha256", ort_library_sha256}};
  }
};

// Nonblocking pipes share ONE absolute deadline for request bytes, read, parse
// and validation. Startup is separate and never authorizes an actuator output.
class ActorProcess {
  Config config_;
  fs::path folder_;
  pid_t pid_ = -1;
  int input_ = -1, output_ = -1;
  std::string buffered_;
  unsigned cap_ = 0, attempted_ = 0, completed_ = 0;
  bool ready_ = false, closed_ = false, reaped_ = false;
  Json stats_;
  static void close_fd(int& fd) { if (fd >= 0) { ::close(fd); fd = -1; } }
  void await(int fd, short event, std::int64_t deadline) {
    for (;;) {
      const auto left = deadline - steady_ns(); require(left >= 0, "RPC_DEADLINE");
      pollfd descriptor{fd, event, 0};
      const int result = ::poll(&descriptor, 1, static_cast<int>((left + 999999) / 1000000));
      if (result < 0 && errno == EINTR) continue;
      require(result >= 0, "pipe poll failed");
      require(result > 0 && steady_ns() <= deadline, "RPC_DEADLINE");
      require(!(descriptor.revents & POLLNVAL), "invalid pipe descriptor");
      if (descriptor.revents & (event | POLLHUP | POLLERR)) return;
    }
  }
  void send(const std::string& bytes, std::int64_t deadline) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      await(input_, POLLOUT, deadline);
      const auto count = ::write(input_, bytes.data() + offset, bytes.size() - offset);
      if (count < 0 && (errno == EAGAIN || errno == EINTR)) continue;
      require(count > 0, "actor input pipe closed"); offset += static_cast<std::size_t>(count);
    }
    require(steady_ns() <= deadline, "RPC_DEADLINE");
  }
  Json receive(std::int64_t deadline) {
    while (buffered_.find('\n') == std::string::npos) {
      await(output_, POLLIN, deadline);
      std::array<char, 4096> block{};
      const auto count = ::read(output_, block.data(), block.size());
      if (count < 0 && (errno == EAGAIN || errno == EINTR)) continue;
      require(count > 0, "actor output EOF");
      buffered_.append(block.data(), static_cast<std::size_t>(count));
      require(buffered_.size() <= 65536, "actor output exceeds cap");
    }
    const auto at = buffered_.find('\n'); const auto text = buffered_.substr(0, at);
    buffered_.erase(0, at + 1);
    // Unsolicited extra output is a dirty stream, not a future response cache.
    require(buffered_.empty(), "unsolicited actor output");
    auto value = parse(text);
    require(value.is_object(), "actor response must be object");
    require(steady_ns() <= deadline, "RPC_DEADLINE");
    return value;
  }
  void reap(std::int64_t deadline) {
    if (pid_ <= 0) { reaped_ = true; return; }
    if (reaped_) return;
    int status = 0;
    for (;;) {
      const auto result = ::waitpid(pid_, &status, WNOHANG);
      if (result == pid_) break;
      if (result < 0 && errno == EINTR) continue;
      require(result == 0, "owned actor waitpid failed");
      require(steady_ns() < deadline, "owned actor could not be reaped");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    reaped_ = true; stats_["wait_status"] = status;
  }
 public:
  ActorProcess(const Config& config, fs::path folder, unsigned cap)
      : config_(config), folder_(std::move(folder)), cap_(cap) {
    require(cap > 0 && cap <= config.max_requests, "session cap");
    require(fs::create_directory(folder_), "session output exists");
    stats_ = {{"status", "INCOMPLETE"}, {"requests_attempted", 0}, {"requests_completed", 0},
              {"inference_attempts", nullptr}, {"inference_returns", nullptr}, {"cleanup_complete", false}};
    write_json(folder_ / "report.json", stats_);
  }
  ~ActorProcess() { try { abort_now(); reap(steady_ns() + 300000000); persist(); } catch (...) {} }
  void start() {
    config_.verify();
    int to_child[2]{-1,-1}, from_child[2]{-1,-1};
    require(::pipe2(to_child, O_CLOEXEC) == 0, "input pipe");
    if (::pipe2(from_child, O_CLOEXEC) != 0) { ::close(to_child[0]); ::close(to_child[1]); throw std::runtime_error("output pipe"); }
    const int error_fd = ::open((folder_/"stderr.log").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (error_fd < 0) { for (int fd : {to_child[0],to_child[1],from_child[0],from_child[1]}) ::close(fd); throw std::runtime_error("stderr open"); }
    std::vector<std::string> args{config_.binary.string(), "--policy", config_.model.string(),
        "--expected-model-sha256", config_.model_sha256, "--expected-source-sha256", config_.source_sha256,
        "--max-requests", std::to_string(cap_)};
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data()); argv.push_back(nullptr);
    posix_spawn_file_actions_t actions{}; posix_spawnattr_t attributes{};
    bool actions_initialized=false, attributes_initialized=false;
    auto setup_check=[](int code,const std::string& operation) {
      require(code==0,operation+": "+std::string(std::strerror(code)));
    };
    auto destroy_setup=[&]() {
      int first_error=0;
      if(actions_initialized) {
        actions_initialized=false;
        const int code=posix_spawn_file_actions_destroy(&actions);
        if(code!=0) first_error=code;
      }
      if(attributes_initialized) {
        attributes_initialized=false;
        const int code=posix_spawnattr_destroy(&attributes);
        if(code!=0 && first_error==0) first_error=code;
      }
      return first_error;
    };
    try {
      setup_check(posix_spawn_file_actions_init(&actions),"spawn actions init"); actions_initialized=true;
      setup_check(posix_spawnattr_init(&attributes),"spawn attributes init"); attributes_initialized=true;
      setup_check(posix_spawn_file_actions_adddup2(&actions,to_child[0],STDIN_FILENO),"spawn stdin action");
      setup_check(posix_spawn_file_actions_adddup2(&actions,from_child[1],STDOUT_FILENO),"spawn stdout action");
      setup_check(posix_spawn_file_actions_adddup2(&actions,error_fd,STDERR_FILENO),"spawn stderr action");
      setup_check(posix_spawnattr_setflags(&attributes,POSIX_SPAWN_SETPGROUP),"spawn process-group flag");
      setup_check(posix_spawnattr_setpgroup(&attributes,0),"spawn process-group identity");
      pid_t spawned_pid=-1;
      setup_check(posix_spawn(&spawned_pid,args[0].c_str(),&actions,&attributes,argv.data(),environ),"actor spawn failed");
      // POSIX does not define the output PID on failure. Never signal it unless
      // posix_spawn itself succeeded and transferred ownership of a real child.
      pid_=spawned_pid;
      setup_check(destroy_setup(),"spawn setup destruction");
    } catch(...) {
      const int cleanup_error=destroy_setup();
      if(cleanup_error!=0) stats_["spawn_setup_cleanup_error"]=std::strerror(cleanup_error);
      for(int fd:{to_child[0],to_child[1],from_child[0],from_child[1],error_fd}) ::close(fd);
      if(pid_>0) abort_now();
      throw;
    }
    ::close(to_child[0]); ::close(from_child[1]); ::close(error_fd);
    input_ = to_child[1]; output_ = from_child[0];
    require(::fcntl(input_, F_SETFL, O_NONBLOCK) == 0 && ::fcntl(output_, F_SETFL, O_NONBLOCK) == 0,
            "cannot make actor pipes nonblocking");
    const auto response = receive(steady_ns() + 10000000000LL);
    require(response.value("format", "") == "d1-cpp-actor19-ready-v1" && response.value("status", "") == "READY", "actor not READY");
    const auto identities = config_.identities();
    for (const auto& item : identities.items()) require(response.at(item.key()) == item.value(), "READY identity differs");
    require(response.at("onnxruntime_version") == "1.10.0" && response.at("cpu_threads") == 1 &&
            response.at("max_requests") == cap_, "READY version/threads/cap differs");
    require(response.at("signature") == Json{{"inputs", {{"obs", {1,57}}, {"history", {1,10,57}}}},
                                            {"outputs", {{"actions", {1,16}}}}}, "READY signature differs");
    require(response.at("reference_shaping") == Json{{"gain",1.5},{"tau_s",.1},{"dt_s",.02},{"reset","new_process"}}, "READY shaper differs");
    for (const auto* key : {"inference_attempts","inference_returns","robot_models_constructed","physics_steps_executed"})
      require(response.at(key).is_number_unsigned() && response.at(key) == 0, "READY counters differ");
    ready_ = true; stats_["ready"] = response; stats_["status"] = "READY";
    stats_["inference_attempts"] = 0; stats_["inference_returns"] = 0; persist();
  }
  Json exchange(const d1_adapter82::ActorInput& raw, std::int64_t deadline, std::int64_t read_delay_ns=0) {
    require(ready_ && !closed_ && attempted_ < cap_, "actor unavailable/request cap");
    const unsigned seq = attempted_++;
    stats_["inference_attempts"] = nullptr; stats_["inference_returns"] = nullptr;
    const Json request{{"format","d1-cpp-actor19-request-v1"},{"seq",seq},{"actor",raw}};
    send(request.dump() + '\n', deadline);
    if (read_delay_ns) std::this_thread::sleep_for(std::chrono::nanoseconds(read_delay_ns));
    const auto response = receive(deadline);
    require(response.size() == 8 && response.at("format") == "d1-cpp-actor19-response-v1" && response.at("status") == "OK" &&
            response.at("seq").is_number_unsigned() && response.at("seq") == seq, "actor response schema/sequence");
    for (const auto* key : {"inference_attempts", "inference_returns"}) {
      require(response.at(key).is_number_unsigned() && response.at(key) == seq + 1, "actor inference counters differ");
      stats_[key] = response.at(key);
    }
    for (const auto& field : std::vector<std::pair<std::string,std::size_t>>{{"actions",16},{"effective_yaw_commands",11}}) {
      const auto& values = response.at(field.first);
      require(values.is_array() && values.size() == field.second, "response vector shape");
      for (const auto& item : values) {
        require(item.is_number(), "response vector numeric"); const double v = item.get<double>();
        require(std::isfinite(v) && std::abs(v) <= (field.first == "actions" ? 1. : .25), "response vector bounds");
      }
    }
    require(response.at("effective_input_sha256").is_string() && is_sha(response.at("effective_input_sha256").get<std::string>()), "effective input SHA format");
    require(steady_ns() <= deadline, "RPC_DEADLINE");
    ++completed_; return response;
  }
  void abort_now() {
    if (pid_ > 0 && !reaped_ && !closed_) ::kill(-pid_, SIGKILL);
    if (!closed_) stats_["status"] = "INCOMPLETE";
    close_fd(input_); close_fd(output_); closed_ = true; ready_ = false;
  }
  void finish(bool graceful) {
    if (!closed_ && graceful && ready_) {
      close_fd(input_);
      try {
        const auto end = receive(steady_ns() + 200000000);
        require(end.at("format") == "d1-cpp-actor19-closed-v1" && end.at("status") == "CLOSED", "missing CLOSED receipt");
        for (const auto* key : {"requests","inference_attempts","inference_returns"}) require(end.at(key) == completed_, "CLOSED count");
        require(attempted_ == completed_ && end.at("robot_models_constructed") == 0 && end.at("physics_steps_executed") == 0, "CLOSED completion");
        stats_["eof_receipt"] = end;
        reap(steady_ns() + 300000000);
        require(WIFEXITED(stats_["wait_status"].get<int>()) && WEXITSTATUS(stats_["wait_status"].get<int>()) == 0, "actor nonzero exit");
        stats_["status"] = "CLOSED";
      } catch (...) { abort_now(); reap(steady_ns()+300000000); persist(); throw; }
    } else abort_now();
    close_fd(output_); closed_ = true; ready_ = false;
    reap(steady_ns()+300000000); config_.verify(); persist();
  }
  void persist() {
    stats_["requests_attempted"] = attempted_; stats_["requests_completed"] = completed_;
    stats_["cleanup_complete"] = reaped_; write_json(folder_/"report.json", stats_);
  }
  Json stats() const { auto result=stats_; result["requests_attempted"]=attempted_; result["requests_completed"]=completed_; result["cleanup_complete"]=reaped_; return result; }
};

class SessionEngine {
  Config config_;
  fs::path output_;
  d1_adapter82::Adapter adapter_;
  std::unique_ptr<ActorProcess> actor_;
  std::ofstream trace_, pending_;
  Json sessions_ = Json::array();
  unsigned sessions_started_ = 0, attempts_ = 0, completed_ = 0;
  std::int64_t accepted_capture_ = -1;
  std::int64_t command_received_ns_ = -1;
  Vector3 last_command_{};
  bool armed_ = false, all_clean_ = true, ever_failed_ = false, verified_after_ = false;
  std::string fault_;
  void close_session(bool graceful) {
    if (!actor_) return;
    try { actor_->finish(graceful); }
    catch (...) { all_clean_ = false; sessions_.push_back(actor_->stats()); actor_.reset(); throw; }
    sessions_.push_back(actor_->stats()); actor_.reset();
  }
 public:
  SessionEngine(Config config, fs::path output) : config_(std::move(config)), output_(std::move(output)) {
    config_.verify(); require(fs::create_directory(output_), "engine output must be new");
    trace_.open(output_/"ticks.jsonl", std::ios::binary); require(bool(trace_), "trace open");
    pending_.open(output_/"attempts.jsonl", std::ios::binary); require(bool(pending_), "attempt ledger open");
    write_json(output_/"report.json", report());
  }
  ~SessionEngine() { try { finish(); } catch (...) {} }
  void reset() {
    armed_ = false; verified_after_ = false; close_session(fault_.empty());
    require(attempts_ < config_.max_requests && sessions_started_ < 16, "engine cap exhausted");
    config_.verify(); fault_.clear(); accepted_capture_ = -1; command_received_ns_ = -1; last_command_ = {};
    require(adapter_.reset(steady_ns()), "core reset failed");
    actor_ = std::make_unique<ActorProcess>(config_, output_/("session_"+std::to_string(sessions_started_++)), config_.max_requests-attempts_);
    try { actor_->start(); armed_ = true; }
    catch (const std::exception& e) { fail(e.what()); throw; }
  }
  void fail(const std::string& reason) {
    if (fault_.empty()) fault_ = reason;
    armed_ = false; ever_failed_ = true; accepted_capture_ = -1;
    // Authorization is revoked before cleanup. Do not block a fault path on waitpid.
    if (actor_) actor_->abort_now();
  }
  bool command(Vector3 value) {
    if (!armed_) return false;
    const auto now=steady_ns();
    if (!adapter_.receive_command(value, now)) { fail(adapter_.fault()); return false; }
    for (auto& v:value) v=std::clamp(v,-1.,1.);
    last_command_=value; command_received_ns_=now;
    return true;
  }
  d1_adapter82::Output output(const d1_adapter82::Joints& q, const d1_adapter82::Joints& dq) {
    if (!armed_ || !fault_.empty()) { d1_adapter82::Output out; out.reason=fault_.empty()?"NOT_ARMED":fault_; return out; }
    const auto now = steady_ns();
    if (accepted_capture_ >= 0 && now-accepted_capture_ > kDeadline) { fail("SAMPLE_ACTION_DEADLINE"); return output(q,dq); }
    auto out = adapter_.torque_output(q,dq,now);
    if (!adapter_.fault().empty()) fail(adapter_.fault());
    return out;
  }
  Json tick(const Sample& sample, std::int64_t captured_ns, std::int64_t artificial_read_delay_ns=0) {
    const auto began = steady_ns();
    Json row{{"schema","d1-adapter83-tick-v1"},{"synthetic",true},{"capture_ns",captured_ns},
             {"begin_ns",began},{"generation",adapter_.generation()},{"status","INCOMPLETE"},
             {"gyro3",sample.gyro},{"quat_wxyz4",sample.quaternion_wxyz},{"q16",sample.q},{"dq16",sample.dq},
             {"command3",last_command_},{"command_received_ns",command_received_ns_},
             {"command_age_ns",command_received_ns_>=0?Json(began-command_received_ns_):Json(nullptr)}};
    try {
      require(armed_ && fault_.empty(), "engine not armed");
      require(captured_ns >= 0 && captured_ns <= began && began-captured_ns <= kDeadline, "SAMPLE_AGE");
      require(attempts_ < config_.max_requests, "engine request cap");
      const auto request = adapter_.begin_tick(sample, began); require(request.has_value(), adapter_.fault());
      row["token_sequence"] = request->token.sequence; row["raw_actor627"] = request->actor;
      ++attempts_;
      // Flushed lower-bound intent before any RPC write. A killed process has
      // unknown ORT attempts/returns for this row, never an invented zero.
      auto intent=row; intent["phase"]="BEFORE_RPC"; intent["request_attempt"]=attempts_;
      intent["actual_inference_attempts"]=nullptr; intent["actual_inference_returns"]=nullptr;
      pending_<<intent.dump()<<'\n'; pending_.flush(); require(bool(pending_),"attempt ledger write");
      require(artificial_read_delay_ns >= 0 && artificial_read_delay_ns <= 30000000, "fault injection bound");
      const auto response = actor_->exchange(request->actor, captured_ns+kDeadline, artificial_read_delay_ns);
      row["response"] = response;
      const auto action = response.at("actions").get<d1_adapter82::Action>();
      const auto accepted = steady_ns(); require(accepted <= captured_ns+kDeadline, "RPC_DEADLINE");
      require(adapter_.complete_tick(request->token, action, accepted), adapter_.fault());
      accepted_capture_ = captured_ns; ++completed_;
      row["accepted_ns"] = accepted; row["status"] = "COMPLETE";
    } catch (const std::exception& e) { row["error"] = e.what(); fail(e.what()); }
    const auto out = output(sample.q,sample.dq);
    row["authorized"] = out.authorized; row["effort16"] = out.effort;
    row["reason"] = out.reason; row["has_targets"] = bool(out.targets);
    row["end_ns"] = steady_ns();
    if (!out.authorized && row["status"] == "COMPLETE") row["status"] = "INCOMPLETE";
    trace_ << row.dump() << '\n'; trace_.flush();
    if (!trace_) { fail("TRACE_WRITE_FAILED"); throw std::runtime_error("TRACE_WRITE_FAILED"); }
    require(fs::file_size(output_/"ticks.jsonl") <= 8*1024*1024, "trace size cap");
    return row;
  }
  void finish() {
    armed_ = false;
    try { close_session(fault_.empty()); config_.verify(); verified_after_ = true; }
    catch (...) { all_clean_ = false; throw; }
    if (trace_.is_open()) { trace_.flush(); require(bool(trace_), "trace flush"); trace_.close(); }
    if (pending_.is_open()) { pending_.flush(); require(bool(pending_), "attempt ledger flush"); pending_.close(); }
    write_json(output_/"report.json", report());
  }
  Json report() const {
    return {{"schema","d1-adapter83-session-v1"},{"status",!ever_failed_ && verified_after_ && fault_.empty() && !armed_ && !actor_ && all_clean_ && completed_>0?"COMPLETE":"INCOMPLETE"},
            {"synthetic",true},{"hardware_connected",false},{"physics_steps",0},{"parameter_updates",0},
            {"identities",config_.identities()},{"requests_attempted",attempts_},{"requests_completed",completed_},
            {"sessions_started",sessions_started_},{"sessions",sessions_},{"fault",fault_},
            {"cleanup_complete",!actor_ && all_clean_},{"verification_after",verified_after_?"COMPLETE":"INCOMPLETE"},
            {"request_cap",config_.max_requests}};
  }
};

inline Sample synthetic_sample(unsigned tick) {
  Sample sample;
  sample.gyro = {.01*std::sin(.1*tick),-.02,.03};
  for (std::size_t i=0; i<16; ++i) { sample.q[i]=d1_adapter82::q0(i)+.001*std::sin(.2*tick+i); sample.dq[i]=.02*std::cos(.1*tick+i); }
  return sample;
}
inline Json sample_json(const Sample& s) {
  std::vector<double> values;
  for (auto v:s.gyro) values.push_back(v);
  for (auto v:s.quaternion_wxyz) values.push_back(v);
  for (auto v:s.q) values.push_back(v);
  for (auto v:s.dq) values.push_back(v);
  return values;
}
inline Sample parse_sample(const Json& values) {
  require(values.is_array() && values.size()==39,"coherent sensor package must have 39 values");
  std::array<double,39> data{};
  for(std::size_t i=0;i<39;++i) { require(values[i].is_number(),"sensor value must be numeric"); data[i]=values[i].get<double>(); require(std::isfinite(data[i]),"nonfinite sensor package"); }
  Sample s; std::copy_n(data.begin(),3,s.gyro.begin()); std::copy_n(data.begin()+3,4,s.quaternion_wxyz.begin());
  std::copy_n(data.begin()+7,16,s.q.begin()); std::copy_n(data.begin()+23,16,s.dq.begin()); return s;
}
}  // namespace d1_adapter83
