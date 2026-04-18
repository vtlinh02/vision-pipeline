// Purpose: Minimal config loader (flat YAML key: value). Not a full YAML parser.

#pragma once
#include <string>

namespace visioncore {

struct Config {
  int imgsz = 640;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  bool end2end = true;

  int intra_threads = 0;  // 0 = ORT default (physical cores)
  int inter_threads = 1;

  int warmup = 20;
  int repeat = 200;

  bool write_json = true;
  bool write_overlay = true;
};

Config load_config_yaml_flat(const std::string& path);

}  // namespace visioncore
