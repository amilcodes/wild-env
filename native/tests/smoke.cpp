#include <cassert>
#include <cmath>

#include "aeolus/reference_kernel.hpp"

int main() {
  const auto low = aeolus::ignition_hazard(1.0, 1.0, 1.0, 1.0, 60.0);
  const auto high = aeolus::ignition_hazard(8.0, 1.0, 1.0, 1.0, 60.0);
  const auto treated = aeolus::ignition_hazard(8.0, 1.0, 0.2, 1.0, 60.0);
  assert(low > 0.0 && low < high && high < 1.0);
  assert(treated < high);

  const double ros[] = {1.0, 8.0, 8.0};
  const double residual[] = {1.0, 1.0, 1.0};
  const double treatment[] = {1.0, 1.0, 0.2};
  const double distance[] = {1.0, 1.0, 1.0};
  double output[3] = {};
  aeolus_ignition_hazard_batch(ros, residual, treatment, distance, 60.0,
                               output, 3);
  assert(std::abs(output[0] - low) < 1e-12);
  assert(std::abs(output[1] - high) < 1e-12);
  assert(std::abs(output[2] - treated) < 1e-12);
}
