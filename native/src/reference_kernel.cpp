#include "aeolus/reference_kernel.hpp"

extern "C" double aeolus_ignition_hazard(double ros_m_per_min,
                                          double residual,
                                          double treatment_factor,
                                          double distance_cells,
                                          double cell_size_m) {
  return aeolus::ignition_hazard(ros_m_per_min, residual, treatment_factor,
                                 distance_cells, cell_size_m);
}

extern "C" void aeolus_ignition_hazard_batch(
    const double* ros_m_per_min, const double* residual,
    const double* treatment_factor, const double* distance_cells,
    double cell_size_m, double* output, unsigned long count) {
  for (unsigned long index = 0; index < count; ++index) {
    output[index] = aeolus::ignition_hazard(
        ros_m_per_min[index], residual[index], treatment_factor[index],
        distance_cells[index], cell_size_m);
  }
}
