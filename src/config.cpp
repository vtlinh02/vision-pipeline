// Purpose: Implements flat YAML (key: value) parsing for demo settings.

#include "visioncore/config.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace visioncore {

static inline std::string trim(std::string s) {
  auto not_space = [](unsigned char c){ return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

static inline bool to_bool(const std::string& v) {
  std::string x = v;
  std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c){ return std::tolower(c); });
  return (x == "true" || x == "1" || x == "yes" || x == "y");
}

Config load_config_yaml_flat(const std::string& path) {
  Config cfg;
  std::ifstream f(path);
  if (!f.is_open()) return cfg;  // fallback to defaults

  std::string line;
  while (std::getline(f, line)) {
    // Strip comments
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);

    line = trim(line);
    if (line.empty()) continue;

    auto colon = line.find(':');
    if (colon == std::string::npos) continue;

    std::string key = trim(line.substr(0, colon));
    std::string val = trim(line.substr(colon + 1));

    // Remove optional quotes
    if (!val.empty() && (val.front() == '"' || val.front() == '\'')) val.erase(val.begin());
    if (!val.empty() && (val.back() == '"' || val.back() == '\'')) val.pop_back();

    try {
      if (key == "imgsz") cfg.imgsz = std::stoi(val);
      else if (key == "conf_thresh") cfg.conf_thresh = std::stof(val);
      else if (key == "iou_thresh") cfg.iou_thresh = std::stof(val);
      else if (key == "end2end") cfg.end2end = to_bool(val);
      else if (key == "intra_threads") cfg.intra_threads = std::stoi(val);
      else if (key == "inter_threads") cfg.inter_threads = std::stoi(val);
      else if (key == "warmup") cfg.warmup = std::stoi(val);
      else if (key == "repeat") cfg.repeat = std::stoi(val);
      else if (key == "write_json") cfg.write_json = to_bool(val);
      else if (key == "write_overlay") cfg.write_overlay = to_bool(val);
    } catch (...) {
      // ignore parse errors (demo-friendly)
    }
  }
  return cfg;
}

}  // namespace visioncore
