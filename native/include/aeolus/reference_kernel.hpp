#pragma once

#include <algorithm>
#include <cmath>

namespace aeolus {

// Native parity primitive for the high-throughput port. The Python truth
// simulator is still the canonical semantics; tests should compare this
// function's output before a CUDA/vectorized integration replaces it.
inline double ignition_hazard(double ros_m_per_min, double residual,
                              double treatment_factor, double distance_cells,
                              double cell_size_m) {
  const auto scaled = ros_m_per_min * residual * treatment_factor *
                      distance_cells / std::max(cell_size_m, 1e-9);
  return std::clamp(1.0 - std::exp(-std::max(0.0, scaled)), 0.0, 1.0);
}

}  // namespace aeolus

extern "C" {

double aeolus_ignition_hazard(double ros_m_per_min, double residual,
                              double treatment_factor, double distance_cells,
                              double cell_size_m);

void aeolus_ignition_hazard_batch(const double* ros_m_per_min,
                                  const double* residual,
                                  const double* treatment_factor,
                                  const double* distance_cells,
                                  double cell_size_m, double* output,
                                  unsigned long count);
}
