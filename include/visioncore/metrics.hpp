// Purpose: Simple timing statistics (mean, p50, p95) for benchmark output.

#pragma once
#include <vector>
#include <string>

namespace visioncore {

struct Stats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
};

Stats compute_stats_ms(const std::vector<double>& samples_ms);

std::string json_escape(const std::string& s);

}  // namespace visioncore
