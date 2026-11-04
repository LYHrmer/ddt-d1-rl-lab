// Protocol fixture only: no ORT/model/physics. Test identity files are inert text.
#include "runtime83/session83.hpp"
#include <cstdlib>
#include <iostream>
#include <map>
using namespace d1_adapter83;
int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2) { require(i+1<argc,"args"); args.emplace(argv[i],argv[i+1]); }
    const auto* mode_env=std::getenv("D1_ADAPTER83_FAKE_MODE");
    const std::string mode=mode_env?mode_env:"normal";
    const auto* library=std::getenv("D1_ADAPTER83_FAKE_ORT"); require(library,"fixture ORT identity path");
    const unsigned cap=static_cast<unsigned>(std::stoul(args.at("--max-requests")));
    Json ready{{"format","d1-cpp-actor19-ready-v1"},{"status","READY"},
               {"binary_sha256",sha256(fs::canonical("/proc/self/exe"))},{"model_sha256",args.at("--expected-model-sha256")},
               {"source_sha256",args.at("--expected-source-sha256")},{"ort_library_sha256",sha256(library)},
               {"onnxruntime_version","1.10.0"},{"cpu_threads",1},{"max_requests",cap},
               {"signature",{{"inputs",{{"obs",{1,57}},{"history",{1,10,57}}}},{"outputs",{{"actions",{1,16}}}}}},
               {"reference_shaping",{{"gain",1.5},{"tau_s",.1},{"dt_s",.02},{"reset","new_process"}}},
               {"inference_attempts",0},{"inference_returns",0},{"robot_models_constructed",0},{"physics_steps_executed",0}};
    if(mode=="bad_identity") ready["model_sha256"]=std::string(64,'f');
    std::cout<<ready.dump()<<std::endl;
    if(mode=="eof") return 0;
    std::string line; unsigned count=0;
    while(std::getline(std::cin,line)) {
      const auto input=parse(line); require(input.at("seq")==count,"fixture seq");
      if(mode=="late") std::this_thread::sleep_for(std::chrono::milliseconds(35));
      if(mode=="malformed") { std::cout<<"{bad json}\n"<<std::flush; return 0; }
      std::array<float,16> actions{}; actions.fill(.25f);
      Json response{{"format","d1-cpp-actor19-response-v1"},{"status","OK"},{"seq",count},
                    {"actions",actions},{"effective_yaw_commands",std::array<float,11>{}},
                    {"effective_input_sha256",std::string(64,'0')},{"inference_attempts",count+1},{"inference_returns",count+1}};
      if(mode=="wrong_seq") response["seq"]=count+1;
      if(mode=="out_of_bounds") response["actions"][0]=1.1;
      std::cout<<response.dump()<<std::endl; ++count;
    }
    std::cout<<Json{{"format","d1-cpp-actor19-closed-v1"},{"status","CLOSED"},{"requests",count},
                   {"inference_attempts",count},{"inference_returns",count},
                   {"robot_models_constructed",0},{"physics_steps_executed",0}}.dump()<<std::endl;
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
