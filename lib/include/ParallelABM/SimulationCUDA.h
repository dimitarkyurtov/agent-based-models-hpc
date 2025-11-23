#ifndef PARALLELABM_SIMULATIONCUDA_H
#define PARALLELABM_SIMULATIONCUDA_H

#include "LocalRegion.h"
#include "ModelCUDA.h"
#include "Simulation.h"

namespace ParallelABM {

/**
 * @class SimulationCUDA
 * @brief CUDA/GPU-specific simulation implementation using NVIDIA GPUs.
 *
 * SimulationCUDA executes agent-based simulations on NVIDIA GPUs by
 * distributing work across multiple GPU devices. Each GPU processes a subregion
 * of agents using the ModelCUDA's interaction rule kernel.
 *
 * GPU EXECUTION MODEL:
 * ====================
 * - Reads GPU count from Environment::GetNumberOfGPUs()
 * - Splits the local region into subregions matching GPU count
 * - For each GPU:
 *   1. Sets the active CUDA device
 *   2. Allocates device memory for agents and neighbors
 *   3. Copies data from host to device
 *   4. Launches the interaction rule kernel with maximum thread blocks/threads
 *   5. Copies updated agents back to host memory
 *   6. Frees device memory
 * - Synchronizes all GPU operations before returning
 *
 * KERNEL CONFIGURATION:
 * =====================
 * The kernel is launched with optimal block and thread counts based on
 * device properties (queried via cudaGetDeviceProperties). The configuration
 * maximizes occupancy by using the maximum threads per block supported by
 * the device and calculating the required number of blocks to cover all agents.
 *
 * @see Simulation Base class providing MPI coordination
 * @see ModelCUDA CUDA interaction rule implementation
 */
class SimulationCUDA : public Simulation<ModelCUDA> {
 public:
  // Inherit base class constructors
  using Simulation<ModelCUDA>::Simulation;

  // Copy constructor - deleted (inherited from base class semantics)
  SimulationCUDA(const SimulationCUDA&) = delete;

  // Copy assignment - deleted
  SimulationCUDA& operator=(const SimulationCUDA&) = delete;

  // Move constructor - deleted (inherited from base class semantics)
  SimulationCUDA(SimulationCUDA&&) = delete;

  // Move assignment - deleted
  SimulationCUDA& operator=(SimulationCUDA&&) = delete;

  /**
   * @brief Execute model computation across available NVIDIA GPUs.
   *
   * Splits the local region into subregions based on available GPUs from the
   * environment. For each subregion, launches the model's interaction rule
   * kernel on the corresponding GPU with maximum available thread blocks
   * and threads.
   *
   * @param local_region Pointer to the local region containing agents
   */
  void LaunchModel(LocalRegion* local_region) override;

  ~SimulationCUDA() override = default;
};

}  // namespace ParallelABM

#endif  // PARALLELABM_SIMULATIONCUDA_H
