// Purpose: Implements end-to-end parsing (1,300,6) and optional classic YOLO decoding + OpenCV NMS.

#include "visioncore/postprocess.hpp"
#include "visioncore/coco_names.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

namespace visioncore {

static inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(v, hi));
}

std::vector<Detection> postprocess_yolo26_end2end(
  const OrtOutput& out,
  const LetterboxInfo& lb,
  float conf_thresh
) {
  std::vector<Detection> dets;
  if (out.shape.size() != 3 || out.shape[0] != 1 || out.shape[2] != 6) {
    return dets;
  }

  const int64_t max_det = out.shape[1];
  dets.reserve(static_cast<size_t>(max_det));

  const auto& names = coco80_names();

  for (int64_t i = 0; i < max_det; ++i) {
    const size_t base = static_cast<size_t>(i * 6);
    float x1 = out.data[base + 0];
    float y1 = out.data[base + 1];
    float x2 = out.data[base + 2];
    float y2 = out.data[base + 3];
    float conf = out.data[base + 4];
    float cls_f = out.data[base + 5];

    if (conf < conf_thresh) continue;

    // Undo letterbox: subtract padding then divide by gain.
    x1 = (x1 - lb.left) / lb.gain;
    x2 = (x2 - lb.left) / lb.gain;
    y1 = (y1 - lb.top) / lb.gain;
    y2 = (y2 - lb.top) / lb.gain;

    // Clip to image bounds.
    x1 = clampf(x1, 0.f, static_cast<float>(lb.orig_w - 1));
    x2 = clampf(x2, 0.f, static_cast<float>(lb.orig_w - 1));
    y1 = clampf(y1, 0.f, static_cast<float>(lb.orig_h - 1));
    y2 = clampf(y2, 0.f, static_cast<float>(lb.orig_h - 1));

    int cls = static_cast<int>(std::lround(cls_f));
    Detection d;
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    d.score = conf;
    d.class_id = cls;
    if (cls >= 0 && cls < static_cast<int>(names.size())) d.class_name = names[cls];
    dets.push_back(std::move(d));
  }
  return dets;
}

// Classic YOLO: (1, 4+nc, N) or (1, N, 4+nc).
std::vector<Detection> postprocess_classic_yolo(
  const OrtOutput& out,
  const LetterboxInfo& lb,
  float conf_thresh,
  float iou_thresh,
  int num_classes
) {
  std::vector<Detection> dets;
  if (out.shape.size() != 3 || out.shape[0] != 1) return dets;

  const int64_t d1 = out.shape[1];
  const int64_t d2 = out.shape[2];
  const int64_t feat = 4 + num_classes;

  bool layout_c_first = (d1 == feat);     // (1, feat, N)
  bool layout_n_first = (d2 == feat);     // (1, N, feat)
  if (!layout_c_first && !layout_n_first) return dets;

  const int64_t N = layout_c_first ? d2 : d1;

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  boxes.reserve(static_cast<size_t>(N));
  scores.reserve(static_cast<size_t>(N));
  class_ids.reserve(static_cast<size_t>(N));

  auto get_feat = [&](int64_t i, int j) -> float {
    // i in [0,N), j in [0,feat)
    if (layout_n_first) {
      // (1,N,feat): contiguous
      return out.data[static_cast<size_t>(i * feat + j)];
    } else {
      // (1,feat,N): channel-major
      return out.data[static_cast<size_t>(j * N + i)];
    }
  };

  for (int64_t i = 0; i < N; ++i) {
    float x = get_feat(i, 0);
    float y = get_feat(i, 1);
    float w = get_feat(i, 2);
    float h = get_feat(i, 3);

    int best_c = -1;
    float best_s = 0.f;
    for (int c = 0; c < num_classes; ++c) {
      float s = get_feat(i, 4 + c);
      if (s > best_s) { best_s = s; best_c = c; }
    }
    if (best_s < conf_thresh) continue;

    // Undo letterbox using center-based coords like Ultralytics example.
    x -= lb.left;
    y -= lb.top;

    float left = (x - w * 0.5f) / lb.gain;
    float top  = (y - h * 0.5f) / lb.gain;
    float ww   = w / lb.gain;
    float hh   = h / lb.gain;

    int xi = static_cast<int>(std::round(clampf(left, 0.f, static_cast<float>(lb.orig_w - 1))));
    int yi = static_cast<int>(std::round(clampf(top,  0.f, static_cast<float>(lb.orig_h - 1))));
    int wi = static_cast<int>(std::round(clampf(ww,   0.f, static_cast<float>(lb.orig_w - xi))));
    int hi = static_cast<int>(std::round(clampf(hh,   0.f, static_cast<float>(lb.orig_h - yi))));

    boxes.emplace_back(xi, yi, wi, hi);
    scores.push_back(best_s);
    class_ids.push_back(best_c);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_thresh, iou_thresh, indices);

  const auto& names = coco80_names();
  for (int idx : indices) {
    Detection d;
    d.x1 = static_cast<float>(boxes[idx].x);
    d.y1 = static_cast<float>(boxes[idx].y);
    d.x2 = static_cast<float>(boxes[idx].x + boxes[idx].width);
    d.y2 = static_cast<float>(boxes[idx].y + boxes[idx].height);
    d.score = scores[idx];
    d.class_id = class_ids[idx];
    if (d.class_id >= 0 && d.class_id < static_cast<int>(names.size())) d.class_name = names[d.class_id];
    dets.push_back(std::move(d));
  }
  return dets;
}

}  // namespace visioncore
