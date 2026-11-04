// Native synthetic sensor -> adapter82 -> unchanged real actor34 -> clipped PD.
#include "runtime83/session83.hpp"
#include <iostream>
#include <map>
using namespace d1_adapter83;

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);  // Broken owned child pipes become EPIPE receipts.
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "probe83 --manifest JSON --output NEW --ticks 32 --sessions 2\n"
                 "Synthetic sensors, real SHA-bound actor34, zero physics/hardware.\n";
    return 0;
  }
  fs::path output;
  std::unique_ptr<SessionEngine> engine;
  Json report{{"schema","d1-adapter83-native-v1"},{"status","INCOMPLETE"},
              {"synthetic",true},{"physics_steps",0},{"hardware_connected",false},{"error",nullptr}};
  bool owned = false;
  try {
    std::map<std::string,std::string> args;
    for (int i=1; i<argc; i+=2) {
      require(i+1<argc, "option value missing");
      require(args.emplace(argv[i],argv[i+1]).second, "duplicate option");
    }
    require(args.size()==4 && args.count("--manifest") && args.count("--output") && args.count("--ticks") && args.count("--sessions"), "explicit manifest/output/ticks/sessions required");
    auto integer = [&](const std::string& key) {
      const auto& value=args.at(key); require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"unsigned decimal required");
      return std::stoul(value);
    };
    const auto ticks=integer("--ticks"), sessions=integer("--sessions");
    require(ticks>0 && ticks<=150 && sessions>0 && sessions<=4 && ticks*sessions<=300,"native count bounds");
    output=args.at("--output"); require(fs::create_directory(output),"output must be new"); owned=true;
    write_json(output/"report.json",report);
    const auto manifest=fs::path(args.at("--manifest"));
    const auto manifest_sha=sha256(manifest); const auto config=Config::read(manifest);
    require(ticks*sessions<=config.max_requests,"manifest cap smaller than request plan");
    fs::copy_file(manifest,output/"manifest.json");
    engine=std::make_unique<SessionEngine>(config,output/"engine");
    std::vector<Json> first;
    for (unsigned session=0; session<sessions; ++session) {
      engine->reset();
      auto next=std::chrono::steady_clock::now();
      for (unsigned tick=0; tick<ticks; ++tick) {
        std::this_thread::sleep_until(next); next += std::chrono::milliseconds(20);
        require(engine->command({.1,0.,tick<ticks/2?.2:-.2}),"command refused");
        const auto captured=steady_ns(); const auto sample=synthetic_sample(tick);
        const auto row=engine->tick(sample,captured);
        require(row.at("status")=="COMPLETE" && row.at("authorized")==true,"native tick failed");
        Json comparison{{"raw_actor627",row.at("raw_actor627")},{"actions",row.at("response").at("actions")},
                        {"effective_yaw_commands",row.at("response").at("effective_yaw_commands")},
                        {"effective_input_sha256",row.at("response").at("effective_input_sha256")}};
        if (session==0) first.push_back(comparison);
        else require(comparison==first.at(tick),"fresh-session synthetic replay differs");
      }
    }
    engine->finish(); report["engine"]=engine->report();
    require(report.at("engine").at("status")=="COMPLETE", "engine incomplete");
    require(sha256(manifest)==manifest_sha,"manifest changed");
    report["manifest_sha256"]=manifest_sha; report["sessions"]=sessions; report["ticks_per_session"]=ticks;
    report["fresh_session_value_and_effective_hash_replay"] = sessions>1 ? Json(true) : Json(nullptr);
    report["verification_after"]="COMPLETE"; report["status"]="COMPLETE";
  } catch (const std::exception& e) { report["error"]=e.what(); }
  if (engine) {
    try { engine->finish(); } catch (const std::exception& e) { report["status"]="INCOMPLETE"; report["cleanup_error"]=e.what(); }
    report["engine"]=engine->report();
  }
  if (owned) { try { write_json(output/"report.json",report); } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; } }
  std::cout<<report.dump()<<'\n'; return report.at("status")=="COMPLETE"?0:2;
}
