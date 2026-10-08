#include "mujoco_ros2_control/step_evidence.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace mujoco_ros2_control {
namespace {
const char *const kWheelBodies[] = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
constexpr uint64_t kMaxEvidenceBytes = 128ull * 1024 * 1024;

void Number(std::ostream &out, double value) {
  if (std::isfinite(value)) out << value;
  else out << "null";
}

void Array(std::ostream &out, const mjtNum *values, int count) {
  out << '[';
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    Number(out, values[i]);
  }
  out << ']';
}

void String(std::ostream &out, const char *value) {
  if (!value) { out << "null"; return; }
  out << '"';
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
    if (*p == '"' || *p == '\\') out << '\\' << *p;
    else if (*p < 32) {
      const char hex[] = "0123456789abcdef";
      out << "\\u00" << hex[*p >> 4] << hex[*p & 15];
    } else out << *p;
  }
  out << '"';
}

struct Bounds {
  mjtNum min[3];
  mjtNum max[3];
};

// Transform the compiler's local geom AABB, including its center offset.
Bounds GeomBounds(const mjModel *model, const mjData *data, int id, bool cylinder) {
  Bounds bounds;
  const mjtNum *rotation = data->geom_xmat + 9 * id;
  const mjtNum *position = data->geom_xpos + 3 * id;
  const mjtNum *aabb = model->geom_aabb + 6 * id;
  const mjtNum *size = model->geom_size + 3 * id;
  for (int row = 0; row < 3; ++row) {
    mjtNum center = position[row], extent = 0;
    if (cylinder) {
      const mjtNum axis = rotation[3 * row + 2];
      extent = size[0] * std::sqrt(std::max(mjtNum(0), 1 - axis * axis)) +
               size[1] * std::abs(axis);
    } else {
      for (int col = 0; col < 3; ++col) {
        center += rotation[3 * row + col] * aabb[col];
        extent += std::abs(rotation[3 * row + col]) * aabb[col + 3];
      }
    }
    bounds.min[row] = center - extent;
    bounds.max[row] = center + extent;
  }
  return bounds;
}

void BoundsJson(std::ostream &out, const Bounds &bounds) {
  out << "{\"min\":";
  Array(out, bounds.min, 3);
  out << ",\"max\":";
  Array(out, bounds.max, 3);
  out << '}';
}
}  // namespace

StepEvidence::StepEvidence(const std::string &path, const mjModel *model,
                           const mjData *data, uint64_t max_native_steps)
    : max_native_steps_(max_native_steps), is_robot_(model->ngeom, false),
      is_wheel_(model->ngeom, false) {
  if (model->opt.integrator != mjINT_EULER) {
    throw std::runtime_error("Native-step evidence currently qualifies only the unchanged Euler integrator");
  }
  base_id_ = mj_name2id(model, mjOBJ_BODY, "base_link");
  floor_id_ = mj_name2id(model, mjOBJ_GEOM, "floor");
  step_id_ = mj_name2id(model, mjOBJ_GEOM, "target_step");
  if (base_id_ < 0 || floor_id_ < 0) {
    throw std::runtime_error("Step evidence requires base_link and floor");
  }
  // The initial mj_forward has already populated all derived geometry.
  for (int i = 0; i < 4; ++i) {
    const int body = mj_name2id(model, mjOBJ_BODY, kWheelBodies[i]);
    int found = -1;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      if (model->geom_bodyid[geom] == body && model->geom_type[geom] == mjGEOM_CYLINDER &&
          (model->geom_contype[geom] || model->geom_conaffinity[geom])) {
        if (found >= 0) throw std::runtime_error("Wheel body has multiple collision cylinders");
        found = geom;
      }
    }
    if (found < 0) throw std::runtime_error("Wheel collision cylinder missing");
    wheel_ids_[i] = found;
    is_wheel_[found] = true;
  }
  for (int geom = 0; geom < model->ngeom; ++geom) {
    int body = model->geom_bodyid[geom];
    while (body > 0 && body != base_id_) body = model->body_parentid[body];
    if (body == base_id_ && (model->geom_contype[geom] || model->geom_conaffinity[geom])) {
      is_robot_[geom] = true;
      robot_geoms_.push_back(geom);
    }
  }
  const int sensor = mj_name2id(model, mjOBJ_SENSOR, "frame_vel");
  const int site = mj_name2id(model, mjOBJ_SITE, "trunk_imu");
  if (sensor < 0 || site < 0 || model->sensor_type[sensor] != mjSENS_FRAMELINVEL ||
      model->sensor_dim[sensor] != 3 || model->sensor_refid[sensor] != -1 ||
      model->sensor_objtype[sensor] != mjOBJ_SITE || model->sensor_objid[sensor] != site ||
      model->site_bodyid[site] != base_id_ || mju_norm3(model->site_pos + 3 * site) > 1e-9) {
    throw std::runtime_error("Step evidence requires world frame_vel at base_link origin");
  }
  velocity_adr_ = model->sensor_adr[sensor];
  if (step_id_ >= 0 && (model->geom_type[step_id_] != mjGEOM_BOX ||
                       model->geom_bodyid[step_id_] != 0)) {
    throw std::runtime_error("target_step must be a static world box");
  }
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) throw std::runtime_error("Cannot exclusively create evidence: " + path + ": " + std::strerror(errno));
  file_ = fdopen(fd, "w");
  if (!file_) { close(fd); throw std::runtime_error("fdopen evidence failed"); }

  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"type\":\"header\",\"schema_version\":1,\"dt\":";
  Number(out, model->opt.timestep);
  out << ",\"nq\":" << model->nq << ",\"nv\":" << model->nv
      << ",\"integrator\":" << model->opt.integrator
      << ",\"max_bytes\":" << kMaxEvidenceBytes
      << ",\"floor_id\":" << floor_id_ << ",\"max_native_steps\":" << max_native_steps_
      << ",\"target_step\":";
  if (step_id_ < 0) out << "null";
  else {
    out << "{\"geom_id\":" << step_id_ << ",\"position\":";
    Array(out, data->geom_xpos + 3 * step_id_, 3);
    out << ",\"size\":";
    Array(out, model->geom_size + 3 * step_id_, 3);
    out << ",\"bounds\":";
    BoundsJson(out, GeomBounds(model, data, step_id_, false));
    out << '}';
  }
  out << ",\"wheels\":[";
  for (int i = 0; i < 4; ++i) {
    if (i) out << ',';
    const int geom = wheel_ids_[i];
    out << "{\"body\":\"" << kWheelBodies[i] << "\",\"geom_id\":" << geom
        << ",\"radius\":" << model->geom_size[3 * geom]
        << ",\"half_width\":" << model->geom_size[3 * geom + 1] << '}';
  }
  out << "],\"robot_collision_geom_ids\":[";
  for (size_t i = 0; i < robot_geoms_.size(); ++i) {
    if (i) out << ',';
    out << robot_geoms_[i];
  }
  out << "],\"state_time\":\"pre_time: derived geometry, pose, sensor velocity and solved contact force; post_time: integrated state time\"}";
  try { Write(out.str(), true); }
  catch (...) { fclose(file_); file_ = nullptr; throw; }
}

StepEvidence::~StepEvidence() {
  if (file_) {
    try {
      std::ostringstream end;
      end << std::setprecision(17) << "{\"type\":\"end\",\"n\":" << count_
          << ",\"last_post_time\":";
      Number(end, last_post_time_);
      end << ",\"bytes_before_footer\":" << bytes_written_ << '}';
      Write(end.str(), true);
    } catch (const std::exception &error) {
      std::fprintf(stderr, "Evidence finalization failed: %s\n", error.what());
    }
    fclose(file_);
  }
}

void StepEvidence::Write(const std::string &line, bool flush) {
  if (bytes_written_ + line.size() + 1 > kMaxEvidenceBytes) {
    throw std::runtime_error("Native-step evidence exceeded the 128 MiB bound; stopping");
  }
  if (fwrite(line.data(), 1, line.size(), file_) != line.size() || fputc('\n', file_) == EOF ||
      (flush && fflush(file_) != 0)) {
    throw std::runtime_error("Native-step evidence write failed");
  }
  bytes_written_ += line.size() + 1;
}

void StepEvidence::Record(const mjModel *model, const mjData *data,
                          uint64_t control_count, bool split_step) {
  ++count_;
  last_post_time_ = data->time;
  std::ostringstream out;
  out << std::setprecision(17) << "{\"type\":\"step\",\"n\":" << count_
      << ",\"native_route\":\"" << (split_step ? "split" : "resync")
      << "\",\"monotonic_wall_s\":"
      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count()
      << ",\"control_count\":" << control_count << ",\"pre_time\":";
  Number(out, data->time - model->opt.timestep);
  out << ",\"post_time\":";
  Number(out, data->time);
  mjtNum external_force_max_abs = 0;
  for (int i = 0; i < 6 * model->nbody; ++i) {
    if (!std::isfinite(data->xfrc_applied[i])) {
      external_force_max_abs = std::numeric_limits<mjtNum>::quiet_NaN();
      break;
    }
    external_force_max_abs = std::max(external_force_max_abs, std::abs(data->xfrc_applied[i]));
  }
  out << ",\"external_force_max_abs\":";
  Number(out, external_force_max_abs);
  out << ",\"base\":{\"position\":";
  Array(out, data->xpos + 3 * base_id_, 3);
  out << ",\"quaternion_wxyz\":";
  Array(out, data->xquat + 4 * base_id_, 4);
  out << ",\"velocity\":";
  Array(out, data->sensordata + velocity_adr_, 3);
  out << "},\"wheels\":[";
  for (int i = 0; i < 4; ++i) {
    if (i) out << ',';
    const int geom = wheel_ids_[i];
    const mjtNum *rotation = data->geom_xmat + 9 * geom;
    const mjtNum axis[] = {rotation[2], rotation[5], rotation[8]};
    out << "{\"body\":\"" << kWheelBodies[i] << "\",\"geom_id\":" << geom
        << ",\"position\":";
    Array(out, data->geom_xpos + 3 * geom, 3);
    out << ",\"axis\":";
    Array(out, axis, 3);
    out << ",\"bounds\":";
    BoundsJson(out, GeomBounds(model, data, geom, true));
    out << '}';
  }
  Bounds robot_bounds;
  for (int j = 0; j < 3; ++j) {
    robot_bounds.min[j] = std::numeric_limits<mjtNum>::infinity();
    robot_bounds.max[j] = -std::numeric_limits<mjtNum>::infinity();
  }
  for (const int geom : robot_geoms_) {
    const Bounds b = GeomBounds(model, data, geom, false);
    for (int j = 0; j < 3; ++j) {
      robot_bounds.min[j] = std::min(robot_bounds.min[j], b.min[j]);
      robot_bounds.max[j] = std::max(robot_bounds.max[j], b.max[j]);
    }
  }
  out << "],\"robot_bounds\":";
  BoundsJson(out, robot_bounds);
  out << ",\"contacts\":[";
  bool first = true;
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact &contact = data->contact[i];
    const int g1 = contact.geom[0], g2 = contact.geom[1];
    if (g1 < 0 || g2 < 0 || is_robot_[g1] == is_robot_[g2]) continue;
    if (!first) out << ',';
    first = false;
    mjtNum force[6];
    mj_contactForce(model, data, i, force);
    const int robot_geom = is_robot_[g1] ? g1 : g2;
    out << "{\"geom1\":" << g1 << ",\"geom2\":" << g2 << ",\"name1\":";
    String(out, mj_id2name(model, mjOBJ_GEOM, g1));
    out << ",\"name2\":";
    String(out, mj_id2name(model, mjOBJ_GEOM, g2));
    out << ",\"dist\":";
    Number(out, contact.dist);
    out << ",\"efc_address\":" << contact.efc_address << ",\"position\":";
    Array(out, contact.pos, 3);
    out << ",\"normal\":";
    Array(out, contact.frame, 3);
    out << ",\"fn\":";
    Number(out, force[0]);
    out << ",\"nonwheel_environment\":" << (is_wheel_[robot_geom] ? "false" : "true") << '}';
  }
  out << "]}";
  Write(out.str(), count_ % 20 == 0);
  if (max_native_steps_ && count_ == max_native_steps_) {
    std::ostringstream limit;
    limit << std::setprecision(17) << "{\"type\":\"limit\",\"n\":" << count_
          << ",\"post_time\":";
    Number(limit, data->time);
    limit << ",\"max_native_steps\":" << max_native_steps_ << '}';
    Write(limit.str(), true);
  }
}
}  // namespace mujoco_ros2_control
