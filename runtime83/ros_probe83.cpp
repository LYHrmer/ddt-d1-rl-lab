// Real DDS + real actor34, fixed coherent synthetic sensor packets only.
// Never publishes a controller-manager, bridge, CAN or actuator command topic.
#include "runtime83/session83.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
using namespace d1_adapter83;

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc==2 && std::string(argv[1])=="--help") {
    std::cout << "ros_probe83 --execute-synthetic-ros true --manifest JSON --output NEW "
                 "--domain N --namespace /adapter83/RUN --ticks N --mode nominal|stale_command|late_response|late_receive\n";
    return 0;
  }
  fs::path output;
  bool owned=false;
  Json report{{"schema","d1-adapter83-dds-synthetic-v1"},{"status","INCOMPLETE"},
              {"hardware_connected",false},{"physics_steps",0},{"synthetic",true},{"error",nullptr},
              {"scenario_assertions_passed",false},{"runtime_accepted",false},{"runtime_status","NOT_STARTED"}};
  std::unique_ptr<SessionEngine> engine;
  std::shared_ptr<rclcpp::Context> context;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::vector<rclcpp::Node::SharedPtr> nodes;
  std::ofstream packets, outputs, publications;
  unsigned sent=0, received=0, observed=0;
  unsigned publish_attempted=0, publish_returned=0, late_publish_returns=0;
  unsigned authorized_observed=0, authorized_consumed=0, expired_authorized_rejected=0;
  unsigned revoked_authorized_rejected=0;
  bool fault_seen=false;
  std::string callback_error;
  try {
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2) { require(i+1<argc,"option requires value"); require(args.emplace(argv[i],argv[i+1]).second,"duplicate option"); }
    require(args.size()==8 && args.at("--execute-synthetic-ros")=="true", "explicit synthetic-only opt-in required");
    const std::string name_space=args.at("--namespace"), mode=args.at("--mode"), domain=args.at("--domain");
    require(name_space.rfind("/adapter83/",0)==0 && name_space.size()>11 &&
            name_space.substr(11).find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")==std::string::npos,
            "private adapter83 namespace required");
    require(!domain.empty() && domain.find_first_not_of("0123456789")==std::string::npos && std::stoul(domain)<=232,"domain bound");
    const auto* actual_domain=std::getenv("ROS_DOMAIN_ID"); const auto* localhost=std::getenv("ROS_LOCALHOST_ONLY");
    require(actual_domain && domain==actual_domain && localhost && std::string(localhost)=="1", "explicit matching domain and localhost-only required");
    require(mode=="nominal" || mode=="stale_command" || mode=="late_response" || mode=="late_receive", "unknown mode");
    const auto& ticks_text=args.at("--ticks");
    require(!ticks_text.empty() && ticks_text.find_first_not_of("0123456789")==std::string::npos,"unsigned ticks required");
    const auto ticks=std::stoul(ticks_text); require(ticks>0 && ticks<=30,"DDS cap is 30 ticks");
    require(mode!="late_receive" || ticks==1,"late-receive scenario uses exactly one packet/RPC");
    report["mode"]=mode; report["planned_packets"]=ticks; report["fault_expected"]=mode!="nominal";
    output=args.at("--output"); require(fs::create_directory(output),"new output required"); owned=true;
    write_json(output/"report.json",report);
    const fs::path manifest=args.at("--manifest"); const auto manifest_sha=sha256(manifest);
    const auto config=Config::read(manifest); require(config.max_requests<=30 && ticks<=config.max_requests,"DDS manifest cap");
    fs::copy_file(manifest,output/"manifest.json");
    packets.open(output/"received_packets.jsonl"); outputs.open(output/"observed_efforts.jsonl");
    publications.open(output/"publication_events.jsonl");
    require(bool(packets)&&bool(outputs)&&bool(publications),"DDS logs open");
    engine=std::make_unique<SessionEngine>(config,output/"engine"); engine->reset();
    if(mode=="stale_command") {
      require(engine->command({.1,0.,.2}),"stale fixture initial command refused");
      // Explicit fixture injection before DDS starts: no inference is needed
      // to establish that this received command is already over500ms old.
      const auto wait_started=steady_ns();
      std::this_thread::sleep_for(std::chrono::milliseconds(501));
      report["fault_injection"]={{"kind","command_held_without_refresh_before_DDS"},
                                 {"wait_started_ns",wait_started},{"wait_ended_ns",steady_ns()}};
    }
    context=std::make_shared<rclcpp::Context>(); const char* ros_argv[]={argv[0]}; context->init(1,ros_argv);
    rclcpp::NodeOptions options; options.context(context); options.use_global_arguments(false);
    auto fixture=std::make_shared<rclcpp::Node>("fixture",name_space,options);
    auto adapter=std::make_shared<rclcpp::Node>("adapter",name_space,options);
    auto observer=std::make_shared<rclcpp::Node>("observer",name_space,options);
    nodes={fixture,adapter,observer};
    const auto qos=rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    auto publisher=fixture->create_publisher<std_msgs::msg::String>("coherent39",qos);
    auto efforts=adapter->create_publisher<std_msgs::msg::String>("test_effort",qos);
    require(std::string(publisher->get_topic_name())==name_space+"/coherent39" &&
            std::string(efforts->get_topic_name())==name_space+"/test_effort","topic remapping prohibited");
    auto publish_result = [&](Json result) {
      // This timestamp precedes serialization. Actual call/return timestamps
      // live in the separate local ledger, not retroactively in the message.
      result["publish_preparation_ns"]=steady_ns();
      std_msgs::msg::String message; message.data=result.dump();
      Json event{{"seq",result.at("seq")},{"status","INCOMPLETE"},
                 {"capture_ns",result.value("capture_ns",Json(nullptr))},
                 {"publish_preparation_ns",result.at("publish_preparation_ns")},
                 {"publish_call_begin_ns",nullptr},{"publish_return_ns",nullptr},
                 {"sent_authorized",result.at("authorized")},
                 {"published_message_can_be_recalled",false}};
      const auto checked_ns=steady_ns();
      if (result.at("authorized").get<bool>()) {
        require(result.at("capture_ns").is_number_integer(),"publication capture type");
        const auto captured=result.at("capture_ns").get<std::int64_t>();
        require(captured>=0 && captured<=checked_ns,"publication capture time");
        if (checked_ns-captured>kDeadline) {
          engine->fail("PUBLISH_DEADLINE");
          result["authorized"]=false; result["has_targets"]=false;
          result["effort16"]=d1_adapter82::Joints{};
          result["reason"]="PUBLISH_DEADLINE"; result["status"]="INCOMPLETE";
          message.data=result.dump(); event["sent_authorized"]=false;
        }
      }
      ++publish_attempted;
      std::int64_t call_begin=-1;
      try {
        call_begin=steady_ns();
        efforts->publish(message);
        const auto returned=steady_ns(); ++publish_returned;
        event["publish_call_begin_ns"]=call_begin;
        event["publish_return_ns"]=returned;
        const bool expired=result.at("authorized").get<bool>() &&
            returned-result.at("capture_ns").get<std::int64_t>()>kDeadline;
        event["authorized_return_after_deadline"]=expired;
        event["status"]=expired?"INCOMPLETE":"COMPLETE";
        if (expired) {
          ++late_publish_returns; engine->fail("PUBLISH_RETURN_DEADLINE");
          event["reason"]="message already sent; authority revoked; consumer must reject expiry";
        }
      } catch(const std::exception& e) {
        event["publish_call_begin_ns"]=call_begin>=0?Json(call_begin):Json(nullptr);
        event["publish_throw_ns"]=steady_ns(); event["error"]=e.what();
        publications<<event.dump()<<'\n'; publications.flush();
        throw;
      }
      publications<<event.dump()<<'\n'; publications.flush();
      require(bool(publications),"publication ledger write");
    };
    auto observation=observer->create_subscription<std_msgs::msg::String>("test_effort",qos,
      [&](const std_msgs::msg::String::SharedPtr message) {
        const auto received_ns=steady_ns();
        try {
          auto value=parse(message->data);
          require(value.at("seq").is_number_unsigned() && value.at("seq")==observed,"effort sequence gap");
          require(value.at("run_namespace")==name_space && value.at("synthetic")==true,"effort provenance");
          require(value.at("effort16").is_array() && value.at("effort16").size()==16,"effort shape");
          const bool wire_authorized=value.at("authorized").get<bool>();
          value["wire_authorized"]=wire_authorized;
          value["wire_effort16"]=value.at("effort16");
          value["wire_has_targets"]=value.at("has_targets");
          value["wire_status"]=value.value("status",Json(nullptr));
          value["wire_reason"]=value.at("reason");
          value["receive_ns"]=received_ns;
          value["consumed_authorized_effort"]=false;
          if (wire_authorized) {
            ++authorized_observed;
            require(value.at("capture_ns").is_number_integer(),"effort capture timestamp type");
            const auto captured=value.at("capture_ns").get<std::int64_t>();
            require(captured>=0 && captured<=received_ns,"effort capture timestamp range");
            require(value.at("has_targets")==true,"authorized message has no targets");
            for(std::size_t i=0;i<16;++i) {
              const auto& torque=value.at("effort16")[i];
              require(torque.is_number(),"effort must be numeric");
              const double numeric=torque.get<double>();
              require(std::isfinite(numeric) && std::abs(numeric)<=d1_adapter82::torque_limit(i),"effort limit");
            }
            if(mode=="late_receive" && observed==0) {
              const auto injection_started=steady_ns();
              require(injection_started-captured<=kDeadline,"late-receive fixture was already stale before injection");
              std::this_thread::sleep_for(std::chrono::milliseconds(21));
              value["fault_injection"]={{"kind","consumer_delay_after_received_authorized_message"},
                                        {"requested_delay_ns",21000000},
                                        {"started_ns",injection_started},{"ended_ns",steady_ns()}};
            }
            const auto upstream_fault=engine->report().at("fault").get<std::string>();
            const auto consume_checked_ns=steady_ns();
            value["consume_checked_ns"]=consume_checked_ns;
            if (consume_checked_ns-captured>kDeadline) {
              ++expired_authorized_rejected; engine->fail("CONSUMER_DEADLINE");
              value["authorized"]=false; value["has_targets"]=false;
              value["effort16"]=d1_adapter82::Joints{};
              value["reason"]="CONSUMER_DEADLINE"; value["status"]="INCOMPLETE";
            } else if(fault_seen || !upstream_fault.empty()) {
              ++revoked_authorized_rejected; engine->fail("CONSUMER_AUTHORITY_REVOKED");
              value["authorized"]=false; value["has_targets"]=false;
              value["effort16"]=d1_adapter82::Joints{};
              value["reason"]="CONSUMER_AUTHORITY_REVOKED"; value["status"]="INCOMPLETE";
              value["upstream_fault"]=upstream_fault;
            }
          }
          if (!value.at("authorized").get<bool>()) {
            fault_seen=true; require(value.at("has_targets")==false,"fault exposes targets");
            for (const auto& torque:value.at("effort16")) require(torque==0.,"fault retains effort");
            value["status"]="INCOMPLETE";
          } else {
            require(!fault_seen,"authorization resumed without explicit reset");
            ++authorized_consumed; value["consumed_authorized_effort"]=true;
          }
          value["effective_authorized"]=value.at("authorized");
          if(mode=="late_receive" && observed==0) {
            require(wire_authorized && value.at("effective_authorized")==false &&
                    value.at("status")=="INCOMPLETE" && value.at("consumed_authorized_effort")==false,
                    "late authorized message was not rejected before consumption");
          }
          outputs<<value.dump()<<'\n'; outputs.flush(); require(bool(outputs),"effort log write"); ++observed;
        } catch(const std::exception& e) { callback_error=e.what(); engine->fail(callback_error); }
      });
    auto subscription=adapter->create_subscription<std_msgs::msg::String>("coherent39",qos,
      [&](const std_msgs::msg::String::SharedPtr message) {
        try {
          const auto packet=parse(message->data);
          require(packet.at("seq").is_number_unsigned() && packet.at("seq")==received,"sample sequence gap");
          require(packet.at("run_namespace")==name_space && packet.at("synthetic")==true && packet.at("generation")==1,"packet identity");
          require(packet.at("capture_ns").is_number_integer(),"capture timestamp type");
          const auto captured=packet.at("capture_ns").get<std::int64_t>();
          const auto sample=parse_sample(packet.at("sensor39"));
          if (!packet.at("command").is_null()) {
            const auto& command=packet.at("command"); require(command.is_array() && command.size()==3,"command shape");
            Vector3 values{};
            for (std::size_t i=0;i<3;++i) { require(command[i].is_number(),"command numeric"); values[i]=command[i].get<double>(); }
            engine->command(values);
          }
          packets<<packet.dump()<<'\n'; packets.flush(); require(bool(packets),"packet log write");
          auto result=engine->tick(sample,captured,mode=="late_response" && received==0?21000000:0);
          result["seq"]=received; result["run_namespace"]=name_space;
          // Logging/serialization can consume the remaining budget. Revoke
          // stale authorization at the actual ROS publish boundary as well.
          auto current=engine->output(sample.q,sample.dq);
          result["authorized"]=current.authorized; result["effort16"]=current.effort;
          result["has_targets"]=bool(current.targets); result["reason"]=current.reason;
          if(!current.authorized) result["status"]="INCOMPLETE";
          publish_result(std::move(result)); ++received;
        } catch(const std::exception& e) {
          callback_error=e.what(); engine->fail(callback_error);
          // A terminal explicit zero output is still emitted before cleanup.
          Json fault{{"seq",received},{"run_namespace",name_space},{"synthetic",true},
                     {"status","INCOMPLETE"},{"authorized",false},{"has_targets",false},
                     {"effort16",d1_adapter82::Joints{}},{"reason",callback_error}};
          publish_result(std::move(fault));
        }
      });
    const auto discovery_deadline=steady_ns()+5000000000LL;
    auto timer=fixture->create_wall_timer(std::chrono::milliseconds(20),[&]() {
      try {
        if(sent>=ticks) return;
        const bool connected=publisher->get_subscription_count()==1 && subscription->get_publisher_count()==1 &&
                             efforts->get_subscription_count()==1 && observation->get_publisher_count()==1;
        if(!connected && sent==0 && steady_ns()<discovery_deadline) return;
        require(connected,"synthetic endpoints differ from one-to-one");
        // Stop-and-wait prevents QoS depth 1 from silently dropping fixtures.
        if(received!=sent || observed!=sent) return;
        const auto captured=steady_ns(); const auto sample=synthetic_sample(sent);
        const Json command=mode=="stale_command"?Json(nullptr):Json::array({.1,0.,sent<ticks/2?.2:-.2});
        const Json packet{{"run_namespace",name_space},{"synthetic",true},{"generation",1},{"seq",sent},
                          {"capture_ns",captured},{"sensor39",sample_json(sample)},{"command",command}};
        std_msgs::msg::String message; message.data=packet.dump(); publisher->publish(message); ++sent;
      } catch(const std::exception& e) { callback_error=e.what(); engine->fail(callback_error); }
    });
    rclcpp::ExecutorOptions executor_options; executor_options.context=context;
    executor=std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executor_options);
    for(const auto& node:nodes) executor->add_node(node);
    const auto deadline=steady_ns()+10000000000LL;
    while(observed<ticks && callback_error.empty() && context->is_valid() && steady_ns()<deadline) {
      executor->spin_some(std::chrono::milliseconds(1));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    timer->cancel();
    require(callback_error.empty(),callback_error); require(sent==ticks && received==ticks && observed==ticks,"DDS transcript incomplete");
    require(mode=="nominal"?!fault_seen:fault_seen,"expected synthetic fault result differs");
    if(mode=="stale_command") require(engine->report().at("fault")=="COMMAND_STALE","wrong stale-command failure");
    if(mode=="late_response") require(engine->report().at("fault")=="RPC_DEADLINE","wrong late-response failure");
    if(mode=="stale_command") require(engine->report().at("requests_attempted")==0,"stale fixture performed inference");
    if(mode=="late_receive") {
      require(engine->report().at("fault")=="CONSUMER_DEADLINE","wrong late-receive failure");
      require(engine->report().at("requests_attempted")==1 && engine->report().at("requests_completed")==1 &&
              authorized_observed==1 && authorized_consumed==0 && expired_authorized_rejected==1,
              "late-receive counts differ or stale message consumed");
    }
    engine->finish(); config.verify(); require(sha256(manifest)==manifest_sha,"manifest changed");
    if(mode=="nominal") require(engine->report().at("status")=="COMPLETE","nominal engine incomplete");
    else require(engine->report().at("status")=="INCOMPLETE","faulted runtime requalified");
    report["status"]="COMPLETE"; report["verification_after"]="COMPLETE";
    report["status_scope"]="synthetic scenario execution/assertions only; runtime qualification is separate";
    report["scenario_assertions_passed"]=true;
    report["runtime_accepted"]=mode=="nominal";
    report["runtime_status"]=engine->report().at("status");
    report["manifest_sha256"]=manifest_sha; report["mode"]=mode; report["fault_expected"]=mode!="nominal";
    report["domain"]=domain; report["namespace"]=name_space; report["actual_sensor_synchronization_validated"]=false;
  } catch(const std::exception& e) { report["error"]=e.what(); }
  if(engine) { try { engine->finish(); } catch(const std::exception& e) { report["status"]="INCOMPLETE"; report["cleanup_error"]=e.what(); } report["engine"]=engine->report(); }
  try { if(executor) { executor->cancel(); for(const auto& node:nodes) executor->remove_node(node); executor.reset(); } nodes.clear(); if(context && context->is_valid()) context->shutdown("adapter83 owned context"); }
  catch(const std::exception& e) { report["status"]="INCOMPLETE"; report["ros_cleanup_error"]=e.what(); }
  if(engine) report["runtime_status"]=engine->report().at("status");
  report["runtime_accepted"]=report.at("status")=="COMPLETE" && report.value("mode",std::string{})=="nominal" &&
                              report.at("runtime_status")=="COMPLETE";
  report["packets_sent"]=sent; report["packets_received"]=received; report["efforts_observed"]=observed;
  report["publish_calls_attempted"]=publish_attempted; report["publish_calls_returned"]=publish_returned;
  report["authorized_publish_returns_after_deadline"]=late_publish_returns;
  report["wire_authorized_messages_observed"]=authorized_observed;
  report["authorized_efforts_consumed"]=authorized_consumed;
  report["expired_authorized_messages_rejected"]=expired_authorized_rejected;
  report["revoked_authorized_messages_rejected"]=revoked_authorized_rejected;
  report["deadline_scope"]="capture to publish entry/return and consumer acceptance checked separately; sent DDS messages cannot be recalled";
  if(owned) { try { write_json(output/"report.json",report); } catch(...) { return 2; } }
  std::cout<<report.dump()<<'\n'; return report.at("status")=="COMPLETE"?0:2;
}
