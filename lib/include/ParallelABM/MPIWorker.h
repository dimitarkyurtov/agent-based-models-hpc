#ifndef PARALLELABM_MPIWORKER_H
#define PARALLELABM_MPIWORKER_H

#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/MPINode.h"

/**
 * @class MPIWorker
 * @brief Represents a worker node in the MPI network.
 *
 * Worker nodes are responsible for processing a local region of agents
 * and communicating with the coordinator node.
 */
class MPIWorker : public MPINode {
 protected:
  std::vector<Agent>
      localRegion;  // The local region of agents this worker processes
  std::vector<Agent> neighbors;  // Neighbor agents for the local region

 public:
  /**
   * @brief Default constructor.
   * @param rank The MPI rank of this worker
   */
  explicit MPIWorker(int rank);

  /**
   * @brief Constructs an MPIWorker with a specific local region.
   * @param rank The MPI rank of this worker
   * @param region The local region of agents to process
   */
  MPIWorker(int rank, const std::vector<Agent>& region);

  /**
   * @brief Gets the local region of agents.
   * @return The local region.
   */
  std::vector<Agent>& GetLocalRegion();

  /**
   * @brief Gets the neighbor agents.
   * @return The neighbor agents.
   */
  std::vector<Agent>& GetNeighbors();

  /**
   * @brief Sets the local region of agents.
   * @param region The new local region
   */
  void SetLocalRegion(const std::vector<Agent>& region);

  /**
   * @brief Receives the local region from the coordinator via MPI.
   *
   * Blocks until the local region and its size are received from the
   * coordinator (rank 0) via MPI communication.
   */
  void ReceiveLocalRegion();

  /**
   * @brief Sends the local region back to the coordinator via MPI.
   *
   * Blocking method that sends the local region back to the coordinator
   * node (rank 0).
   */
  void SendLocalRegionToLeader();

  /**
   * @brief Receives neighbor agents from the coordinator via MPI.
   *
   * Blocks until the neighbor agents are received from the coordinator
   * (rank 0) via MPI communication.
   */
  void ReceiveNeighbors();
};

#endif  // PARALLELABM_MPIWORKER_H
