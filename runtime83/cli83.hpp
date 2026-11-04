#pragma once
// Cold argv validation. No filesystem, actor, ROS, or output construction.
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace d1_cli83 {
struct Options {
  std::string manifest, output, domain, name_space, mode;
  unsigned ticks = 0;
  bool validate_only = false;
};

inline void require(bool okay, const char* message) {
  if (!okay) throw std::invalid_argument(message);
}

inline unsigned decimal(const std::string& value, unsigned maximum) {
  require(!value.empty(), "unsigned decimal must not be empty");
  unsigned result = 0;
  for (const char digit : value) {
    require(digit >= '0' && digit <= '9', "unsigned decimal required");
    const unsigned next = static_cast<unsigned>(digit - '0');
    require(result <= maximum / 10 &&
            (result < maximum / 10 || next <= maximum % 10), "unsigned decimal out of range");
    result = result * 10 + next;
  }
  return result;
}

inline Options parse(const std::vector<std::string>& arguments) {
  Options parsed;
  const std::size_t first = !arguments.empty() && arguments.front() == "--validate-only" ? 1 : 0;
  parsed.validate_only = first != 0;
  require(arguments.size() - first == 14, "exactly seven option/value pairs required");
  std::map<std::string, std::string> values;
  for (std::size_t index = first; index < arguments.size(); index += 2)
    require(values.emplace(arguments[index], arguments[index+1]).second, "duplicate option");
  const std::vector<std::string> required{"--execute-synthetic-ros", "--manifest", "--output",
                                        "--domain", "--namespace", "--ticks", "--mode"};
  for (const auto& key : required) require(values.count(key) == 1, "missing required or unknown option");
  require(values.at("--execute-synthetic-ros") == "true", "explicit synthetic-only opt-in required");
  parsed.manifest = values.at("--manifest"); parsed.output = values.at("--output");
  require(!parsed.manifest.empty() && !parsed.output.empty(), "nonempty manifest/output required");
  parsed.domain = values.at("--domain"); (void)decimal(parsed.domain, 232);
  parsed.name_space = values.at("--namespace");
  const std::string prefix = "/adapter83/";
  require(parsed.name_space.rfind(prefix, 0) == 0 && parsed.name_space.size() > prefix.size() &&
          parsed.name_space.substr(prefix.size()).find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos,
          "private adapter83 namespace required");
  parsed.mode = values.at("--mode");
  require(parsed.mode == "nominal" || parsed.mode == "stale_command" ||
          parsed.mode == "late_response" || parsed.mode == "late_receive", "unknown mode");
  parsed.ticks = decimal(values.at("--ticks"), 30);
  require(parsed.ticks > 0, "ticks must be positive");
  require(parsed.mode != "late_receive" || parsed.ticks == 1, "late_receive requires one packet/RPC");
  return parsed;
}
}  // namespace d1_cli83
