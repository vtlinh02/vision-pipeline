// Purpose: Implements simple JSON writers for Week One outputs.

#include "visioncore/json_writer.hpp"
#include <fstream>
#include <iomanip>

namespace visioncore {

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
) {
  std::ofstream f(path);
  f << "{\n";
  f << "  \"schema_version\": \"1.0\",\n";
  f << "  \"image\": {\n";
  f << "    \"path\": \"" << json_escape(image_path) << "\",\n";
  f << "    \"orig_size\": [" << orig_w << ", " << orig_h << "]\n";
  f << "  },\n";
  f << "  \"model\": {\n";
  f << "    \"path\": \"" << json_escape(model_path) << "\",\n";
  f << "    \"end2end\": " << (end2end ? "true" : "false") << ",\n";
  f << "    \"conf_thresh\": " << std::fixed << std::setprecision(3) << conf_thresh << ",\n";
  f << "    \"iou_thresh\": " << std::fixed << std::setprecision(3) << iou_thresh << "\n";
  f << "  },\n";
  f << "  \"runtime\": {\n";
  f << "    \"backend\": \"onnxruntime\",\n";
  f << "    \"intra_threads\": " << intra_threads << ",\n";
  f << "    \"inter_threads\": " << inter_threads << "\n";
  f << "  },\n";
  f << "  \"detections\": [\n";
  for (size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    f << "    {\n";
    f << "      \"bbox_xyxy\": [" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "],\n";
    f << "      \"confidence\": " << d.score << ",\n";
    f << "      \"class_id\": " << d.class_id << ",\n";
    f << "      \"class_name\": \"" << json_escape(d.class_name) << "\"\n";
    f << "    }" << (i + 1 < dets.size() ? "," : "") << "\n";
  }
  f << "  ]\n";
  f << "}\n";
}

static void write_stats(std::ofstream& f, const char* key, const Stats& s, bool comma) {
  f << "  \"" << key << "\": {\"mean_ms\": " << s.mean_ms
    << ", \"p50_ms\": " << s.p50_ms << ", \"p95_ms\": " << s.p95_ms << "}"
    << (comma ? "," : "") << "\n";
}

void write_metrics_json(const std::string& path, const RunMetrics& m) {
  std::ofstream f(path);
  f << "{\n";
  write_stats(f, "preprocess_ms",  m.preprocess, true);
  write_stats(f, "inference_ms",   m.inference, true);
  write_stats(f, "postprocess_ms", m.postprocess, true);
  write_stats(f, "end_to_end_ms",  m.end_to_end, false);
  f << "}\n";
}

}  // namespace visioncore
