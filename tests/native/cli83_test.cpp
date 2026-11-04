// Cold argv contract only: no actor, model, ROS initialization, or output folder.
// ADAPTER83_CLI_LEGACY_SOURCE_ONLY exposes the frozen arity defect as a red test.
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ADAPTER83_CLI_LEGACY_SOURCE_ONLY
#include "runtime83/cli83.hpp"
#endif

namespace {
unsigned assertions = 0;
void require(bool value, const std::string& message) {
  ++assertions;
  if (!value) throw std::runtime_error(message);
}
std::vector<std::string> complete_arguments() {
  return {"--execute-synthetic-ros", "true", "--manifest", "/unused/cold_manifest.json",
          "--output", "/unused/new_cold_output", "--domain", "183",
          "--namespace", "/adapter83/project83_nominal", "--ticks", "24", "--mode", "nominal"};
}
unsigned frozen_required_pairs(const std::string& path) {
  std::ifstream input(path);
  require(bool(input), "frozen source missing");
  const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  const std::regex pattern(R"(args\.size\(\)\s*==\s*([0-9]+))");
  std::sregex_iterator found(text.begin(), text.end(), pattern), end;
  require(found != end, "frozen arity guard not found");
  const unsigned count = static_cast<unsigned>(std::stoul((*found)[1].str()));
  require(++found == end, "ambiguous frozen arity guard");
  return count;
}
std::size_t actual_pair_count(const std::vector<std::string>& arguments) {
  require(arguments.size() % 2 == 0, "test argv itself has an incomplete pair");
  std::map<std::string, std::string> options;
  for (std::size_t i = 0; i < arguments.size(); i += 2)
    require(options.emplace(arguments[i], arguments[i+1]).second, "duplicate test argv key");
  return options.size();
}
#ifndef ADAPTER83_CLI_LEGACY_SOURCE_ONLY
void rejects(const std::vector<std::string>& arguments, const std::string& name) {
  bool rejected = false;
  try { (void)d1_cli83::parse(arguments); }
  catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid argv accepted: " + name);
}
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "one frozen ros_probe83.cpp path required");
    const auto complete = complete_arguments();
    const auto actual = actual_pair_count(complete);
    const auto frozen = frozen_required_pairs(argv[1]);
    require(actual == 7, "real complete CLI must contain exactly seven pairs");
#ifdef ADAPTER83_CLI_LEGACY_SOURCE_ONLY
    // This asserts the desired contract against the actual frozen source guard.
    // It intentionally fails with required=8, supplied=7 before any ROS/actor.
    require(frozen == actual, "frozen runtime83 rejects the complete seven-pair CLI: required=" +
            std::to_string(frozen) + ", supplied=" + std::to_string(actual));
#else
    require(frozen == 8, "historical defect evidence unexpectedly changed");
    const auto parsed = d1_cli83::parse(complete);
    require(parsed.manifest == "/unused/cold_manifest.json" && parsed.output == "/unused/new_cold_output",
            "path parsing differs");
    require(parsed.domain == "183" && parsed.name_space == "/adapter83/project83_nominal" &&
            parsed.mode == "nominal" && parsed.ticks == 24 && !parsed.validate_only, "nominal parsing differs");
    auto cold = complete; cold.insert(cold.begin(), "--validate-only");
    require(d1_cli83::parse(cold).validate_only, "cold validation flag ignored");
    for (const std::string mode : {"nominal", "late_receive", "late_response", "stale_command"}) {
      auto args = complete;
      args[9] = "/adapter83/project83_" + mode;
      args[11] = mode == "nominal" ? "24" : "1";
      args[13] = mode;
      require(d1_cli83::parse(args).mode == mode, "registered mode rejected");
    }
    auto bad = complete; bad.insert(bad.end(), {"--unknown", "value"}); rejects(bad, "eighth pair");
    bad = complete; bad[0] = "--unknown"; rejects(bad, "unknown key at same arity");
    bad = complete; bad.erase(bad.end()-2, bad.end()); rejects(bad, "missing pair");
    bad = complete; bad.pop_back(); rejects(bad, "missing value");
    bad = complete; bad[2] = "--domain"; rejects(bad, "duplicate key");
    bad = complete; bad[1] = "false"; rejects(bad, "opt-in false");
    bad = complete; bad[3] = ""; rejects(bad, "empty manifest");
    bad = complete; bad[5] = ""; rejects(bad, "empty output");
    for (const std::string value : {"-1", "233", "1.5", "", "999999999999999999999999"}) {
      bad = complete; bad[7] = value; rejects(bad, "invalid domain");
    }
    for (const std::string value : {"/adapter83/", "/wrong/run", "/adapter83/bad/name", "/adapter83/bad-name"}) {
      bad = complete; bad[9] = value; rejects(bad, "invalid namespace");
    }
    for (const std::string value : {"0", "31", "-1", "1.5", "", "999999999999999999999999"}) {
      bad = complete; bad[11] = value; rejects(bad, "invalid ticks");
    }
    bad = complete; bad[13] = "other"; rejects(bad, "invalid mode");
    bad = complete; bad[13] = "late_receive"; rejects(bad, "late_receive requires one tick");
    bad = cold; bad.insert(bad.begin()+1, "true"); rejects(bad, "validate-only has no value");
    bad = cold; bad.insert(bad.begin(), "--validate-only"); rejects(bad, "duplicate cold flag");
#endif
    std::cout << "{\"status\":\"PASS\",\"assertions\":" << assertions
              << ",\"actor_calls\":0,\"model_calls\":0,\"physics_steps\":0,\"ros_initialized\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "COLD_ARGV_CONTRACT_FAILURE: " << error.what() << '\n';
    return 2;
  }
}
