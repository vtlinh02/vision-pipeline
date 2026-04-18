// Purpose: Week One CLI demo: load image, preprocess, run ORT, postprocess, write JSON+overlay+metrics.

#include "visioncore/config.hpp"
#include "visioncore/preprocess.hpp"
#include "visioncore/ort_session.hpp"
#include "visioncore/postprocess.hpp"
#include "visioncore/json_writer.hpp"
#include "visioncore/overlay.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

static void usage() {
  std::cout <<
    "detect_image --model <model.onnx> --image <image.jpg> --out_dir <dir> [--config demo.yaml]\n"
    "           [--threads N] [--warmup N] [--repeat N] [--conf X] [--iou Y] [--end2end true/false]\n";
}

static std::string arg_or(const std::vector<std::string>& args, const std::string& key, const std::string& def) {
  for (size_t i = 0; i + 1 < args.size(); ++i) if (args[i] == key) return args[i + 1];
  return def;
}

static bool has_arg(const std::vector<std::string>& args, const std::string& key) {
  for (const auto& a : args) if (a == key) return true;
  return false;
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (has_arg(args, "--help") || args.empty()) { usage(); return 0; }

  const std::string cfg_path = arg_or(args, "--config", "");
  visioncore::Config cfg;
  if (!cfg_path.empty()) cfg = visioncore::load_config_yaml_flat(cfg_path);

  const std::string model_path = arg_or(args, "--model", "");
  const std::string image_path = arg_or(args, "--image", "");
  const std::string out_dir    = arg_or(args, "--out_dir", "outputs/run");

  if (model_path.empty() || image_path.empty()) {
    usage();
    std::cerr << "ERROR: --model and --image are required.\n";
    return 2;
  }

  if (has_arg(args, "--threads")) cfg.intra_threads = std::stoi(arg_or(args, "--threads", "0"));
  if (has_arg(args, "--warmup"))  cfg.warmup = std::stoi(arg_or(args, "--warmup", "20"));
  if (has_arg(args, "--repeat"))  cfg.repeat = std::stoi(arg_or(args, "--repeat", "200"));
  if (has_arg(args, "--conf"))    cfg.conf_thresh = std::stof(arg_or(args, "--conf", "0.25"));
  if (has_arg(args, "--iou"))     cfg.iou_thresh = std::stof(arg_or(args, "--iou", "0.45"));
  if (has_arg(args, "--end2end")) cfg.end2end = (arg_or(args, "--end2end", "true") == "true");

  fs::create_directories(out_dir);

  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    std::cerr << "ERROR: failed to read image: " << image_path << "\n";
    return 3;
  }

  // Preprocess
  visioncore::LetterboxInfo lb;
  auto t0 = Clock::now();
  cv::Mat rgb = visioncore::bgr_to_rgb(bgr);
  cv::Mat rgb_lb = visioncore::letterbox_rgb(rgb, cfg.imgsz, cfg.imgsz, lb);
  std::vector<float> input = visioncore::to_nchw_float32_0_1(rgb_lb);
  auto t1 = Clock::now();

  // ORT session
  visioncore::OrtSession sess(model_path, cfg.intra_threads, cfg.inter_threads);

  // Warmup + timing loops
  std::vector<double> preprocess_ms, inference_ms, postprocess_ms, end2end_ms;
  preprocess_ms.reserve(cfg.repeat);
  inference_ms.reserve(cfg.repeat);
  postprocess_ms.reserve(cfg.repeat);
  end2end_ms.reserve(cfg.repeat);

  // One inference result to save outputs
  visioncore::OrtOutput out_saved;
  std::vector<visioncore::Detection> dets_saved;

  for (int i = 0; i < cfg.warmup + cfg.repeat; ++i) {
    auto s0 = Clock::now();

    // (Preprocess is constant for same image, but we record it for realism.)
    auto p0 = Clock::now();
    // reuse `input`
    auto p1 = Clock::now();

    auto i0 = Clock::now();
    auto out = sess.run(input, 1, 3, cfg.imgsz, cfg.imgsz);
    auto i1 = Clock::now();

    auto pp0 = Clock::now();
    std::vector<visioncore::Detection> dets;
    if (cfg.end2end) {
      dets = visioncore::postprocess_yolo26_end2end(out, lb, cfg.conf_thresh);
    } else {
      dets = visioncore::postprocess_classic_yolo(out, lb, cfg.conf_thresh, cfg.iou_thresh, 80);
    }
    auto pp1 = Clock::now();

    auto s1 = Clock::now();

    if (i >= cfg.warmup) {
      double pms = std::chrono::duration<double, std::milli>(p1 - p0).count();
      double ims = std::chrono::duration<double, std::milli>(i1 - i0).count();
      double ppms = std::chrono::duration<double, std::milli>(pp1 - pp0).count();
      double sms = std::chrono::duration<double, std::milli>(s1 - s0).count();
      preprocess_ms.push_back(pms);
      inference_ms.push_back(ims);
      postprocess_ms.push_back(ppms);
      end2end_ms.push_back(sms);

      if (out_saved.data.empty()) {
        out_saved = std::move(out);
        dets_saved = std::move(dets);
      }
    }
  }

  // Write outputs
  const std::string stem = fs::path(image_path).stem().string();
  if (cfg.write_json) {
    visioncore::write_detections_json(
      (fs::path(out_dir) / (stem + ".detections.json")).string(),
      image_path,
      bgr.cols, bgr.rows,
      model_path,
      cfg.end2end,
      cfg.conf_thresh,
      cfg.iou_thresh,
      cfg.intra_threads,
      cfg.inter_threads,
      dets_saved
    );
  }

  if (cfg.write_overlay) {
    cv::Mat overlay = bgr.clone();
    visioncore::draw_detections(overlay, dets_saved);
    cv::imwrite((fs::path(out_dir) / (stem + ".overlay.png")).string(), overlay);
  }

  visioncore::RunMetrics rm;
  rm.preprocess = visioncore::compute_stats_ms(preprocess_ms);
  rm.inference = visioncore::compute_stats_ms(inference_ms);
  rm.postprocess = visioncore::compute_stats_ms(postprocess_ms);
  rm.end_to_end = visioncore::compute_stats_ms(end2end_ms);

  visioncore::write_metrics_json((fs::path(out_dir) / (stem + ".metrics.json")).string(), rm);

  std::cout << "Done. Outputs in: " << out_dir << "\n";
  std::cout << "Inference mean(ms): " << rm.inference.mean_ms << " p50: " << rm.inference.p50_ms
            << " p95: " << rm.inference.p95_ms << "\n";
  return 0;
}
