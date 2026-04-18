// Purpose: Writes JSON outputs (detections + metrics) without external JSON dependency.

#pragma once
#include <string>
#include <vector>
#include "visioncore/types.hpp"
#include "visioncore/metrics.hpp"

namespace visioncore {

struct RunMetrics {
  Stats preprocess;
  Stats inference;
  Stats postprocess;
  Stats end_to_end;
};

void write_detections_json(
  const std::string& path,
  const std::string& image_path,
  int orig_w, int orig_h,
  const std::string& model_path,
  bool end2end,
  float conf_thresh,
  float iou_thresh,
  int intra_threads,
  int inter_threads,
  const std::vector<Detection>& dets
);

void write_metrics_json(
  const std::string& path,
  const RunMetrics& m
);

}  // namespace visioncore
