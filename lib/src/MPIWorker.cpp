#include "ParallelABM/MPIWorker.h"

#include <mpi.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPINode.h"

MPIWorker::MPIWorker(int rank, ParallelABM::SplitFunction split_function)
    : MPINode(rank),
      local_region_(nullptr),
      split_function_(std::move(split_function)) {}

MPIWorker::MPIWorker(int rank, std::unique_ptr<ParallelABM::LocalRegion> region)
    : MPINode(rank), local_region_(std::move(region)) {}

ParallelABM::LocalRegion* MPIWorker::GetLocalRegion() noexcept {
  return local_region_.get();
}

const ParallelABM::LocalRegion* MPIWorker::GetLocalRegion() const noexcept {
  return local_region_.get();
}

std::unique_ptr<ParallelABM::LocalRegion>&
MPIWorker::GetLocalRegionPtr() noexcept {
  return local_region_;
}

void MPIWorker::SetLocalRegion(
    std::unique_ptr<ParallelABM::LocalRegion> region) {
  local_region_ = std::move(region);
}

void MPIWorker::ReceiveLocalRegion() {
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

  // Receive agents
  std::vector<Agent> agents(region_size);
  if (region_size > 0) {
    MPI_Recv(agents.data(), static_cast<int>(region_size * sizeof(Agent)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info(
        "MPIWorker: Received local region (" + std::to_string(region_size) +
        " agents) from coordinator");
  }

  // Create LocalRegion with empty neighbors (will be set later via
  // ReceiveNeighbors)
  local_region_ = std::make_unique<ParallelABM::LocalRegion>(
      region_id, std::move(agents), std::vector<Agent>{}, split_function_);
}

void MPIWorker::SendLocalRegionToLeader() {
  if (!local_region_) {
    ParallelABM::Logger::GetInstance().Warning(
        "MPIWorker: Attempted to send null local region to coordinator");
    return;
  }

  const std::vector<Agent>& agents = local_region_->GetAgents();
  const int kRegionSize = static_cast<int>(agents.size());

  MPI_Send(&kRegionSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent region size (" +
                                          std::to_string(kRegionSize) +
                                          ") to coordinator");

  if (kRegionSize > 0) {
    MPI_Send(agents.data(), static_cast<int>(kRegionSize * sizeof(Agent)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent local region (" +
                                            std::to_string(kRegionSize) +
                                            " agents) to coordinator");
  }
}

void MPIWorker::ReceiveNeighbors() {
  // Receive neighbor count
  int num_neighbors = 0;
  MPI_Recv(&num_neighbors, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info(
      "MPIWorker: Received neighbor count (" + std::to_string(num_neighbors) +
      ") from coordinator");

  // Receive neighbors
  std::vector<Agent> neighbors(num_neighbors);
  if (num_neighbors > 0) {
    MPI_Recv(neighbors.data(), static_cast<int>(num_neighbors * sizeof(Agent)),
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
