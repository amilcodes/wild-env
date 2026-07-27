#include <cassert>

#include "aeolus/reference_kernel.hpp"

int main() {
  const auto low = aeolus::ignition_hazard(1.0, 1.0, 1.0, 1.0, 60.0);
  const auto high = aeolus::ignition_hazard(8.0, 1.0, 1.0, 1.0, 60.0);
  const auto treated = aeolus::ignition_hazard(8.0, 1.0, 0.2, 1.0, 60.0);
  assert(low > 0.0 && low < high && high < 1.0);
  assert(treated < high);
}
