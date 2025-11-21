#ifndef PARALLELABM_MPIWORKER_H
#define PARALLELABM_MPIWORKER_H

#include <memory>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/MPINode.h"

/**
 * @class MPIWorker
 * @brief Represents a worker node in the MPI network.
 *
 * Worker nodes are responsible for processing a local region of agents
 * and communicating with the coordinator node. The worker holds a
 * LocalRegion instance that encapsulates both the agents to process
 * and their neighboring agents.
 */
class MPIWorker : public MPINode {
 protected:
  /// The local region of agents this worker processes
  std::unique_ptr<ParallelABM::LocalRegion> local_region_;
  /// Function to split the region into subregions
  ParallelABM::SplitFunction split_function_;

 public:
  /**
   * @brief Constructs an MPIWorker with a splitting function.
   * @param rank The MPI rank of this worker
   * @param split_function Function to split region into subregions
   */
  MPIWorker(int rank, ParallelABM::SplitFunction split_function);

  /**
   * @brief Constructs an MPIWorker with a specific local region.
   * @param rank The MPI rank of this worker
   * @param region The local region to process
   */
  MPIWorker(int rank, std::unique_ptr<ParallelABM::LocalRegion> region);

  /**
   * @brief Gets a pointer to the local region.
   * @return Non-owning pointer to the LocalRegion, or nullptr if not set
   */
  [[nodiscard]] ParallelABM::LocalRegion* GetLocalRegion() noexcept;

  /**
   * @brief Gets a const pointer to the local region.
   * @return Non-owning const pointer to the LocalRegion, or nullptr if not set
   */
  [[nodiscard]] const ParallelABM::LocalRegion* GetLocalRegion() const noexcept;

  /**
   * @brief Gets a reference to the unique_ptr holding the local region.
   * @return Reference to the unique_ptr managing the LocalRegion
   */
  [[nodiscard]] std::unique_ptr<ParallelABM::LocalRegion>&
  GetLocalRegionPtr() noexcept;

  /**
   * @brief Sets the local region by transferring ownership.
   * @param region Unique pointer to the new LocalRegion
   */
  void SetLocalRegion(std::unique_ptr<ParallelABM::LocalRegion> region);

  /**
   * @brief Receives the local region from the coordinator via MPI.
   *
   * Blocks until the region_id, region size, and agents are received from
   * the coordinator (rank 0) via MPI communication. Creates a default
   * LocalRegion implementation to store the received data. Neighbors must
   * be received separately via ReceiveNeighbors().
   */
  void ReceiveLocalRegion();

  /**
   * @brief Sends the local region back to the coordinator via MPI.
   *
   * Blocking method that sends the agents from the local region back to
   * the coordinator node (rank 0).
   */
  void SendLocalRegionToLeader();

  /**
   * @brief Receives neighbor agents from the coordinator via MPI.
   *
   * Blocks until the neighbor agents are received from the coordinator
   * (rank 0) via MPI communication. Updates the existing LocalRegion's
   * neighbors.
   */
  void ReceiveNeighbors();
};

#endif  // PARALLELABM_MPIWORKER_H
