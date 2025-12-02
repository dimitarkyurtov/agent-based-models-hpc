#ifndef PARALLELABM_MPICOORDINATOR_H
#define PARALLELABM_MPICOORDINATOR_H

#include <memory>
#include <span>

#include "ParallelABM/MPIWorker.h"
#include "ParallelABM/Space.h"

/**
 * @class MPICoordinator
 * @brief Template-based coordinator (leader) node in the MPI network.
 *
 * @tparam AgentT The concrete agent type for this coordinator. Must be:
 *   - Default constructible (for MPI receive operations)
 *   - Copy constructible and copy assignable (for MPI send/receive)
 *   - Preferably trivially copyable for optimal MPI performance
 *
 * The coordinator is responsible for managing the simulation space,
 * distributing local regions to worker nodes, and collecting results.
 */
template <typename AgentT>
class MPICoordinator : public MPIWorker<AgentT> {
 private:
  int num_processes_;                     // Total number of MPI processes
  std::shared_ptr<Space<AgentT>> space_;  // Shared simulation space
  std::vector<typename Space<AgentT>::Region>
      regions_;  // Cached regions for all processes

 public:
  /**
   * @brief Constructs an MPICoordinator.
   * @param rank The MPI rank of this coordinator (typically 0)
   * @param num_processes The total number of MPI processes
   * @param space Shared pointer to the simulation space
   */
  MPICoordinator(int rank, int num_processes,
                 std::shared_ptr<Space<AgentT>> space);

  /**
   * @brief Sends local regions to all worker nodes via MPI.
   *
   * Distributes the simulation space among worker nodes by sending each
   * worker its designated local region and the size of that region.
   */
  void SendLocalRegionsToWorkers();

  /**
   * @brief Receives local regions from all worker nodes via MPI.
   *
   * Blocks until all worker nodes have sent their local regions back,
   * then updates the simulation space with the received data.
   */
  void ReceiveLocalRegionsFromWorkers();

  /**
   * @brief Sends neighbor agents to all worker nodes via MPI.
   *
   * Sends pre-computed neighbor agents for each worker's region using the
   * cached regions' neighbor data. This is a blocking operation.
   */
  void SendNeighborsToWorkers();
};

// Include implementation
#include "MPICoordinator.inl"

#endif  // PARALLELABM_MPICOORDINATOR_H
