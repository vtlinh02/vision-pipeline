// Purpose: Implements ORT session creation and inference execution, with threading config.

#include "visioncore/ort_session.hpp"
#include <stdexcept>

namespace visioncore {

OrtSession::OrtSession(const std::string& model_path, int intra_threads, int inter_threads)
  : env_(ORT_LOGGING_LEVEL_WARNING, "visioncore"),
    so_(),
    session_(nullptr) {

  so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (intra_threads >= 0) so_.SetIntraOpNumThreads(intra_threads);
  if (inter_threads >= 0) so_.SetInterOpNumThreads(inter_threads);

  session_ = Ort::Session(env_, model_path.c_str(), so_);

  Ort::AllocatorWithDefaultOptions allocator;
  auto in0 = session_.GetInputNameAllocated(0, allocator);
  auto out0 = session_.GetOutputNameAllocated(0, allocator);
  input_name_ = in0.get();
  output_name_ = out0.get();
}

OrtOutput OrtSession::run(const std::vector<float>& input_nchw, int64_t n, int64_t c, int64_t h, int64_t w) {
  std::vector<int64_t> in_shape = {n, c, h, w};
  const size_t in_size = static_cast<size_t>(n * c * h * w);
  if (input_nchw.size() != in_size) {
    throw std::runtime_error("Input size mismatch for NCHW tensor.");
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    mem, const_cast<float*>(input_nchw.data()), in_size, in_shape.data(), in_shape.size()
  );

  const char* in_names[] = { input_name_.c_str() };
  const char* out_names[] = { output_name_.c_str() };

  auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);

  auto& o0 = outputs[0];
  auto info = o0.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();

  const float* out_data = o0.GetTensorData<float>();
  const size_t out_count = info.GetElementCount();

  OrtOutput out;
  out.shape = shape;
  out.data.assign(out_data, out_data + out_count);
  return out;
}

}  // namespace visioncore
