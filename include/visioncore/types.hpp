// Purpose: Shared types for detections and letterbox metadata.

#pragma once
#include <string>
#include <vector>

namespace visioncore {

struct Detection {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
  float score = 0.f;
  int class_id = -1;
  std::string class_name;
};

struct LetterboxInfo {
  int top = 0;
  int left = 0;
  float gain = 1.f;  // scale factor used during resize
  int in_w = 0;
  int in_h = 0;
  int orig_w = 0;
  int orig_h = 0;
};

}  // namespace visioncore
