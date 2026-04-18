// Purpose: Draws detections on an image and writes overlay PNG.

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "visioncore/types.hpp"

namespace visioncore {

cv::Scalar color_for_class(int class_id);

void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets);

}  // namespace visioncore
