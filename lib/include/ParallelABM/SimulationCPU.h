#ifndef PARALLELABM_SIMULATIONCPU_H
#define PARALLELABM_SIMULATIONCPU_H

#include "LocalRegion.h"
#include "ModelCPU.h"
#include "Simulation.h"

namespace ParallelABM {

/**
 * @class SimulationCPU
 * @brief Template-based CPU simulation implementation using multi-threading.
 *
 * @tparam AgentT The concrete agent type for this simulation. Must be:
 *   - Default constructible (for MPI receive operations)
 *   - Copy constructible and copy assignable (for MPI send/receive)
 *   - Preferably trivially copyable for optimal MPI performance
 *
 * SimulationCPU executes agent-based simulations on CPU by distributing
 * work across multiple threads. Each thread processes a subregion of agents
 * independently using the ModelCPU's interaction rule.
 *
 * THREADING MODEL:
 * ================
 * - Reads CPU core count from Environment::GetNumberOfCPUCores()
 * - Splits the local region into subregions matching core count
 * - Spawns one thread per subregion for parallel execution
 * - Synchronizes all threads before returning control
 *
 * @see Simulation Base class providing MPI coordination
 * @see ModelCPU CPU interaction rule implementation
 */
template <typename AgentT>
class SimulationCPU : public Simulation<AgentT, ModelCPU<AgentT>> {
 public:
  // Inherit base class constructors
  using Simulation<AgentT, ModelCPU<AgentT>>::Simulation;

  // Copy constructor - deleted (inherited from base class semantics)
  SimulationCPU(const SimulationCPU&) = delete;

  // Copy assignment - deleted
  SimulationCPU& operator=(const SimulationCPU&) = delete;

  // Move constructor - deleted (inherited from base class semantics)
  SimulationCPU(SimulationCPU&&) = delete;

  // Move assignment - deleted
  SimulationCPU& operator=(SimulationCPU&&) = delete;

  /**
   * @brief Execute model computation across CPU threads.
   *
   * Splits the local region into subregions based on available CPU cores
   * from the environment and processes each subregion in parallel using
   * the model's interaction rule.
   *
   * @param local_region Pointer to the local region containing agents
   */
  void LaunchModel(LocalRegion<AgentT>* local_region) override;

  ~SimulationCPU() override = default;
};

}  // namespace ParallelABM

// Include implementation
#include "SimulationCPU.inl"

#endif  // PARALLELABM_SIMULATIONCPU_H
