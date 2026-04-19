// Purpose: Implements simple stats and JSON string escaping.

#include "visioncore/metrics.hpp"
#include <algorithm>
#include <numeric>

namespace visioncore {

Stats compute_stats_ms(const std::vector<double>& samples_ms) {
  Stats st;
  if (samples_ms.empty()) return st;

  st.mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();

  std::vector<double> v = samples_ms;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) -> double {
    if (v.empty()) return 0.0;
    double idx = p * (v.size() - 1);
    size_t i0 = static_cast<size_t>(idx);
    size_t i1 = std::min(i0 + 1, v.size() - 1);
    double frac = idx - i0;
    return v[i0] * (1.0 - frac) + v[i1] * frac;
  };
  st.p50_ms = pct(0.50);
  st.p95_ms = pct(0.95);
  return st;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

}  // namespace visioncore
