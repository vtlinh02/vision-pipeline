// Purpose: Preprocessing (BGR->RGB, letterbox padding=114, normalize, NCHW float32).

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "visioncore/types.hpp"

namespace visioncore {

cv::Mat bgr_to_rgb(const cv::Mat& bgr);

cv::Mat letterbox_rgb(const cv::Mat& rgb, int target_w, int target_h, LetterboxInfo& info);

std::vector<float> to_nchw_float32_0_1(const cv::Mat& rgb_letterboxed);

}  // namespace visioncore
