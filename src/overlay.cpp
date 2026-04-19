// Purpose: Implements drawing of boxes/labels using OpenCV.

#include "visioncore/overlay.hpp"
#include <sstream>

namespace visioncore {

cv::Scalar color_for_class(int class_id) {
  // Deterministic palette: simple hashing.
  int r = (class_id * 37) % 255;
  int g = (class_id * 17 + 99) % 255;
  int b = (class_id * 29 + 199) % 255;
  return cv::Scalar(b, g, r);
}

void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets) {
  for (const auto& d : dets) {
    cv::Scalar color = color_for_class(d.class_id);
    cv::rectangle(bgr,
      cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
      color, 2);

    std::ostringstream ss;
    ss << d.class_name << ":" << std::fixed << std::setprecision(2) << d.score;
    std::string label = ss.str();

    int base = 0;
    auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    int x = static_cast<int>(d.x1);
    int y = std::max(0, static_cast<int>(d.y1) - sz.height - 4);

    cv::rectangle(bgr, cv::Rect(x, y, sz.width + 6, sz.height + base + 6), color, cv::FILLED);
    cv::putText(bgr, label, cv::Point(x + 3, y + sz.height + 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);
  }
}

}  // namespace visioncore
