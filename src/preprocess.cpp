// Purpose: Implements letterbox consistent with Ultralytics ONNXRuntime example (pad value 114).

#include "visioncore/preprocess.hpp"
#include <cmath>

namespace visioncore {

cv::Mat bgr_to_rgb(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

cv::Mat letterbox_rgb(const cv::Mat& rgb, int target_w, int target_h, LetterboxInfo& info) {
  info.orig_w = rgb.cols;
  info.orig_h = rgb.rows;
  info.in_w = target_w;
  info.in_h = target_h;

  const int w = rgb.cols;
  const int h = rgb.rows;
  const float r = std::min(static_cast<float>(target_h) / h, static_cast<float>(target_w) / w);
  info.gain = r;

  const int new_w = static_cast<int>(std::round(w * r));
  const int new_h = static_cast<int>(std::round(h * r));

  cv::Mat resized;
  if (new_w != w || new_h != h) {
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  } else {
    resized = rgb;
  }

  const float dw = (target_w - new_w) / 2.0f;
  const float dh = (target_h - new_h) / 2.0f;

  const int top = static_cast<int>(std::round(dh - 0.1f));
  const int bottom = static_cast<int>(std::round(dh + 0.1f));
  const int left = static_cast<int>(std::round(dw - 0.1f));
  const int right = static_cast<int>(std::round(dw + 0.1f));

  info.top = top;
  info.left = left;

  cv::Mat out;
  cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
  return out;
}

std::vector<float> to_nchw_float32_0_1(const cv::Mat& rgb_letterboxed) {
  const int H = rgb_letterboxed.rows;
  const int W = rgb_letterboxed.cols;
  std::vector<float> out(3 * H * W);

  for (int y = 0; y < H; ++y) {
    const cv::Vec3b* row = rgb_letterboxed.ptr<cv::Vec3b>(y);
    for (int x = 0; x < W; ++x) {
      const cv::Vec3b px = row[x];  // RGB order
      const float r = px[0] / 255.0f;
      const float g = px[1] / 255.0f;
      const float b = px[2] / 255.0f;

      out[0 * H * W + y * W + x] = r;
      out[1 * H * W + y * W + x] = g;
      out[2 * H * W + y * W + x] = b;
    }
  }
  return out;
}

}  // namespace visioncore
