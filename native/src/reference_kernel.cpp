#include "aeolus/reference_kernel.hpp"

extern "C" double aeolus_ignition_hazard(double ros_m_per_min,
                                            double residual,
                                            double treatment_factor,
                                            double distance_cells,
                                            double cell_size_m) {
  return aeolus::ignition_hazard(ros_m_per_min, residual, treatment_factor,
                                 distance_cells, cell_size_m);
}
