// Purpose: Postprocessing for YOLO26 end-to-end output and optional classic YOLO decoding + NMS.

#pragma once
#include <vector>
#include "visioncore/types.hpp"
#include "visioncore/ort_session.hpp"

namespace visioncore {

std::vector<Detection> postprocess_yolo26_end2end(
  const OrtOutput& out,
  const LetterboxInfo& lb,
  float conf_thresh
);

std::vector<Detection> postprocess_classic_yolo(
  const OrtOutput& out,
  const LetterboxInfo& lb,
  float conf_thresh,
  float iou_thresh,
  int num_classes
);

}  // namespace visioncore
