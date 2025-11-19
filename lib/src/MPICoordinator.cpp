#include "ParallelABM/MPICoordinator.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPIWorker.h"
#include "ParallelABM/Space.h"

MPICoordinator::MPICoordinator(int rank, int num_processes,
                               std::unique_ptr<Space> space)
    : MPIWorker(rank), num_processes_(num_processes), space_(std::move(space)) {
  this->space_->Initialize();
}

void MPICoordinator::SendLocalRegionsToWorkers() {
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    const std::span<Agent> kRegion = GetRegionForWorker(worker_rank);

    const int kRegionSize = static_cast<int>(kRegion.size());
    MPI_Send(&kRegionSize, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent region size (" + std::to_string(kRegionSize) +
        ") to worker " + std::to_string(worker_rank));

    MPI_Send(kRegion.data(), static_cast<int>(kRegionSize * sizeof(Agent)),
             MPI_BYTE, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent local region (" + std::to_string(kRegionSize) +
        " agents) to worker " + std::to_string(worker_rank));
  }

  const std::span<Agent> kCoordinatorSpan = GetRegionForWorker(0);
  const std::vector<Agent> kCoordinatorRegion(kCoordinatorSpan.begin(),
                                              kCoordinatorSpan.end());
  SetLocalRegion(kCoordinatorRegion);
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own local region (" +
      std::to_string(kCoordinatorRegion.size()) + " agents)");
}

void MPICoordinator::ReceiveLocalRegionsFromWorkers() {
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    int region_size = 0;
    MPI_Recv(&region_size, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Received region size (" + std::to_string(region_size) +
        ") from worker " + std::to_string(worker_rank));

    const std::span<Agent> kRegion = GetRegionForWorker(worker_rank);

    MPI_Recv(kRegion.data(), static_cast<int>(region_size * sizeof(Agent)),
             MPI_BYTE, worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Received local region (" +
        std::to_string(region_size) + " agents) from worker " +
        std::to_string(worker_rank));
  }

  const std::span<Agent> kCoordinatorRegion = GetRegionForWorker(0);
  std::ranges::copy(localRegion, kCoordinatorRegion.begin());
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Copied own local region (" +
      std::to_string(localRegion.size()) + " agents) to space");
}

void MPICoordinator::SendNeighborsToWorkers() {
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    const std::vector<Agent> kNeighborAgents =
        GetNeighborsForWorker(worker_rank);

    const int kNumNeighbors = static_cast<int>(kNeighborAgents.size());
    MPI_Send(&kNumNeighbors, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent neighbor count (" +
        std::to_string(kNumNeighbors) + ") to worker " +
        std::to_string(worker_rank));

    if (kNumNeighbors > 0) {
      MPI_Send(kNeighborAgents.data(),
               static_cast<int>(kNumNeighbors * sizeof(Agent)), MPI_BYTE,
               worker_rank, 0, MPI_COMM_WORLD);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Sent neighbors (" + std::to_string(kNumNeighbors) +
          " agents) to worker " + std::to_string(worker_rank));
    }
  }

  neighbors = GetNeighborsForWorker(0);
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own neighbors (" + std::to_string(neighbors.size()) +
      " agents)");
}

// Mark: Private Methods

std::vector<Agent> MPICoordinator::GetNeighborsForWorker(int rank) {
  const std::span<Agent> kRegion = GetRegionForWorker(rank);
  std::vector<Agent>& agents = space_->GetAgents();

  const auto kStartIndex =
      // NOLINTNEXTLINE(bugprone-pointer-arithmetic-on-polymorphic-object,cert-ctr56-cpp)
      static_cast<unsigned>(kRegion.data() - agents.data());
  const unsigned kEndIndex =
      kStartIndex + static_cast<unsigned>(kRegion.size());

  return space_->GetNeighborsForRegion(kStartIndex, kEndIndex);
}

std::span<Agent> MPICoordinator::GetRegionForWorker(int rank) {
  const int kTotalAgents = static_cast<int>(this->space_->GetAgents().size());

  const int kAgentsPerProcess = kTotalAgents / num_processes_;
  const int kRemainder = kTotalAgents % num_processes_;

  const int kRoughStart =
      (rank * kAgentsPerProcess) + (rank < kRemainder ? rank : kRemainder);
  const int kRoughEnd =
      kRoughStart + kAgentsPerProcess + (rank < kRemainder ? 1 : 0);

  int start_index = 0;
  int end_index = 0;
  if (rank == 0) {
    start_index = 0;
  } else {
    start_index = this->space_->GetNearestProperSplit(kRoughStart, false);
  }

  if (rank == num_processes_ - 1) {
    end_index = kTotalAgents;
  } else {
    end_index = this->space_->GetNearestProperSplit(kRoughEnd, true);
  }

  std::vector<Agent>& agents = this->space_->GetAgents();
  // NOLINTNEXTLINE(bugprone-pointer-arithmetic-on-polymorphic-object,cert-ctr56-cpp,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return {agents.data() + start_index,
          static_cast<size_t>(end_index - start_index)};
}
