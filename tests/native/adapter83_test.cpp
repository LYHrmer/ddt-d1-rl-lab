#include "runtime83/session83.hpp"
#include <cstdlib>
#include <iostream>
using namespace d1_adapter83;
static unsigned assertions=0;
void expect(bool value,const std::string& what) { ++assertions; require(value,what); }
int main(int argc,char** argv) {
  std::signal(SIGPIPE,SIG_IGN);
  try {
    require(argc==3,"native test needs fixture binary and new directory");
    const fs::path directory=argv[2]; require(fs::create_directory(directory),"new fixture directory");
    for(const auto* name:{"model","source","ort"}) { std::ofstream file(directory/name); file<<"inert adapter83 fixture "<<name; }
    Config c; c.binary=argv[1]; c.model=directory/"model"; c.source=directory/"source"; c.ort_library=directory/"ort";
    c.binary_sha256=sha256(c.binary); c.model_sha256=sha256(c.model); c.source_sha256=sha256(c.source);
    c.ort_library_sha256=sha256(c.ort_library); c.max_requests=8;
    setenv("D1_ADAPTER83_FAKE_ORT",c.ort_library.c_str(),1);
    for(const std::string mode:{"normal","late","eof","malformed","wrong_seq","out_of_bounds","bad_identity"}) {
      setenv("D1_ADAPTER83_FAKE_MODE",mode.c_str(),1);
      SessionEngine engine(c,directory/mode);
      bool startup=false;
      try { engine.reset(); startup=true; } catch(...) {}
      if(mode=="bad_identity") {
        expect(!startup,"bad identity accepted");
        engine.finish(); expect(engine.report().at("status")=="INCOMPLETE","bad identity requalified"); continue;
      }
      expect(startup,"fixture startup unexpectedly failed");
      expect(engine.command({.1,0.,.2}),"fresh command");
      const auto sample=synthetic_sample(0); const auto captured=steady_ns();
      const auto result=engine.tick(sample,captured);
      if(mode=="normal") {
        expect(result.at("status")=="COMPLETE" && result.at("authorized")==true,"normal RPC failed");
        engine.reset(); expect(engine.command({.1,0.,.2}),"fresh command after reset");
        const auto again=engine.tick(sample,steady_ns());
        expect(again.at("generation")==2,"reset did not change generation");
        expect(again.at("raw_actor627")==result.at("raw_actor627"),"reset retained raw history/filter/action");
        expect(again.at("response").at("seq")==0,"new process sequence not zero");
      } else {
        expect(result.at("status")=="INCOMPLETE" && result.at("authorized")==false,"bad RPC authorized");
        for(const auto& value:result.at("effort16")) expect(value==0.,"bad RPC leaves effort");
        expect(result.at("has_targets")==false,"bad RPC exposes targets");
        if(mode=="late") {
          setenv("D1_ADAPTER83_FAKE_MODE","normal",1); engine.reset();
          expect(engine.command({0.,0.,0.}),"recovery fresh command");
          const auto recovered=engine.tick(sample,steady_ns());
          expect(recovered.at("generation")==2 && recovered.at("response").at("seq")==0 && recovered.at("authorized")==true,
                 "late old session polluted fresh generation");
        }
      }
      engine.finish(); const auto receipt=engine.report();
      expect(receipt.at("cleanup_complete")==true,"owned child not reaped");
      expect(receipt.at("status")== (mode=="normal"?"COMPLETE":"INCOMPLETE"),"failure history lost");
    }
    setenv("D1_ADAPTER83_FAKE_MODE","normal",1);
    SessionEngine old_sample(c,directory/"old_sample"); old_sample.reset(); old_sample.command({0.,0.,0.});
    const auto stale=old_sample.tick(synthetic_sample(0),steady_ns()-kDeadline-1);
    expect(stale.at("error")=="SAMPLE_AGE" && stale.at("authorized")==false,"stale sample accepted");
    old_sample.finish(); expect(old_sample.report().at("requests_attempted")==0,"stale sample inferred");
    const auto denied_binary=directory/"not_executable";
    { std::ofstream file(denied_binary); file<<"inert non-executable fixture\n"; }
    fs::permissions(denied_binary,fs::perms::owner_read|fs::perms::owner_write,fs::perm_options::replace);
    auto denied_config=c; denied_config.binary=denied_binary;
    denied_config.binary_sha256=sha256(denied_binary);
    SessionEngine denied(denied_config,directory/"spawn_denied");
    bool spawn_denied=false;
    try { denied.reset(); }
    catch(const std::exception& e) {
      spawn_denied=std::string(e.what()).find("actor spawn failed")!=std::string::npos;
    }
    expect(spawn_denied,"non-executable actor did not fail at spawn");
    denied.finish(); const auto denied_receipt=denied.report();
    expect(denied_receipt.at("status")=="INCOMPLETE","spawn failure requalified");
    expect(denied_receipt.at("cleanup_complete")==true,"failed spawn cleanup incomplete");
    expect(denied_receipt.at("requests_attempted")==0 && denied_receipt.at("requests_completed")==0,
           "failed spawn invented requests");
    expect(denied_receipt.at("sessions").size()==1 &&
           denied_receipt.at("sessions")[0].at("cleanup_complete")==true,"failed spawn session receipt missing");
    bool duplicate_rejected=false;
    try { (void)parse("{\"a\":1,\"a\":2}"); } catch(...) { duplicate_rejected=true; }
    expect(duplicate_rejected,"duplicate JSON field accepted");
    std::cout<<Json{{"status","PASS"},{"assertions",assertions},{"fake_server_only",true},
                    {"actual_onnx_calls",0},{"physics_steps",0},{"ros_nodes",0},
                    {"non_executable_spawn_failure_checked",true}}.dump()<<'\n';
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
