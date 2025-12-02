#ifndef PARALLELABM_MPIWORKER_INL
#define PARALLELABM_MPIWORKER_INL

#include <mpi.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPINode.h"

template <typename AgentT>
MPIWorker<AgentT>::MPIWorker(int rank)
    : MPINode(rank), local_region_(nullptr) {}

template <typename AgentT>
MPIWorker<AgentT>::MPIWorker(
    int rank, std::unique_ptr<ParallelABM::LocalRegion<AgentT>> region)
    : MPINode(rank), local_region_(std::move(region)) {}

template <typename AgentT>
ParallelABM::LocalRegion<AgentT>* MPIWorker<AgentT>::GetLocalRegion()
    noexcept {
  return local_region_.get();
}

template <typename AgentT>
const ParallelABM::LocalRegion<AgentT>* MPIWorker<AgentT>::GetLocalRegion()
    const noexcept {
  return local_region_.get();
}

template <typename AgentT>
std::unique_ptr<ParallelABM::LocalRegion<AgentT>>&
MPIWorker<AgentT>::GetLocalRegionPtr() noexcept {
  return local_region_;
}

template <typename AgentT>
void MPIWorker<AgentT>::SetLocalRegion(
    std::unique_ptr<ParallelABM::LocalRegion<AgentT>> region) {
  local_region_ = std::move(region);
}

template <typename AgentT>
void MPIWorker<AgentT>::ReceiveLocalRegion() {
  // Receive region ID
  int region_id = 0;
  MPI_Recv(&region_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Received region ID (" +
                                          std::to_string(region_id) +
                                          ") from coordinator");

  // Receive region size
  int region_size = 0;
  MPI_Recv(&region_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Received region size (" +
                                          std::to_string(region_size) +
                                          ") from coordinator");

  // Receive agents directly as contiguous array
  std::vector<AgentT> agents;
  agents.reserve(static_cast<size_t>(region_size));

  if (region_size > 0) {
    agents.resize(region_size);
    MPI_Recv(agents.data(), static_cast<int>(region_size * sizeof(AgentT)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ParallelABM::Logger::GetInstance().Info(
        "MPIWorker: Received local region (" + std::to_string(region_size) +
        " agents) from coordinator");
  }

  // Create LocalRegion with empty neighbors (will be set later via
  // ReceiveNeighbors)
  local_region_ = std::make_unique<ParallelABM::LocalRegion<AgentT>>(
      region_id, std::move(agents), std::vector<AgentT>{});
}

template <typename AgentT>
void MPIWorker<AgentT>::SendLocalRegionToLeader() {
  if (!local_region_) {
    ParallelABM::Logger::GetInstance().Warning(
        "MPIWorker: Attempted to send null local region to coordinator");
    return;
  }

  const std::vector<AgentT>& agents = local_region_->GetAgents();
  const int kRegionSize = static_cast<int>(agents.size());

  MPI_Send(&kRegionSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent region size (" +
                                          std::to_string(kRegionSize) +
                                          ") to coordinator");

  if (kRegionSize > 0) {
    // Send agents directly as contiguous array (zero-copy)
    MPI_Send(agents.data(), static_cast<int>(kRegionSize * sizeof(AgentT)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent local region (" +
                                            std::to_string(kRegionSize) +
                                            " agents) to coordinator");
  }
}

template <typename AgentT>
void MPIWorker<AgentT>::ReceiveNeighbors() {
  // Receive neighbor count
  int num_neighbors = 0;
  MPI_Recv(&num_neighbors, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info(
      "MPIWorker: Received neighbor count (" + std::to_string(num_neighbors) +
      ") from coordinator");

  // Receive neighbors as contiguous array
  std::vector<AgentT> neighbors;
  neighbors.reserve(static_cast<size_t>(num_neighbors));

  if (num_neighbors > 0) {
    neighbors.resize(num_neighbors);
    MPI_Recv(neighbors.data(), static_cast<int>(num_neighbors * sizeof(AgentT)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ParallelABM::Logger::GetInstance().Info("MPIWorker: Received neighbors (" +
                                            std::to_string(num_neighbors) +
                                            " agents) from coordinator");
  }

  // Update the existing LocalRegion with neighbors
  if (local_region_) {
    local_region_->SetNeighbors(std::move(neighbors));
  } else {
    ParallelABM::Logger::GetInstance().Warning(
        "MPIWorker: Received neighbors but local region is null");
  }
}

#endif  // PARALLELABM_MPIWORKER_INL
