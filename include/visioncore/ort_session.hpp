// Purpose: Small ONNX Runtime session wrapper for CPU inference.

#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace visioncore {

struct OrtOutput {
  std::vector<int64_t> shape;
  std::vector<float> data;  // copied for simplicity (Week One)
};

class OrtSession {
public:
  OrtSession(const std::string& model_path, int intra_threads, int inter_threads);

  OrtOutput run(const std::vector<float>& input_nchw, int64_t n, int64_t c, int64_t h, int64_t w);

private:
  Ort::Env env_;
  Ort::SessionOptions so_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace visioncore
