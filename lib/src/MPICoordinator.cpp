#include "ParallelABM/MPICoordinator.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPIWorker.h"
#include "ParallelABM/Space.h"

MPICoordinator::MPICoordinator(int rank, int num_processes,
                               std::unique_ptr<Space> space,
                               ParallelABM::SplitFunction split_function)
    : MPIWorker(rank, std::move(split_function)),
      num_processes_(num_processes),
      space_(std::move(space)) {
  this->space_->Initialize();

  regions_ = space_->SplitIntoRegions(num_processes_);
  ParallelABM::Logger::GetInstance().Info("MPICoordinator: Split space into " +
                                          std::to_string(regions_.size()) +
                                          " regions");
}

void MPICoordinator::SendLocalRegionsToWorkers() {
  std::vector<Agent>& agents = space_->GetAgents();

  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    const Space::Region& region = regions_[worker_rank];
    const std::vector<int>& indices = region.GetIndices();
    const int kRegionId = region.GetRegionId();

    MPI_Send(&kRegionId, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent region id (" + std::to_string(kRegionId) +
        ") to worker " + std::to_string(worker_rank));

    const int kRegionSize = static_cast<int>(indices.size());
    MPI_Send(&kRegionSize, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent region size (" + std::to_string(kRegionSize) +
        ") to worker " + std::to_string(worker_rank));

    if (kRegionSize > 0) {
      std::vector<int> block_lengths(kRegionSize, sizeof(Agent));
      std::vector<int> displacements(kRegionSize);

      for (size_t i = 0; i < indices.size(); ++i) {
        displacements[i] = indices[i] * static_cast<int>(sizeof(Agent));
      }

      MPI_Datatype indexed_type = MPI_DATATYPE_NULL;
      MPI_Type_indexed(kRegionSize, block_lengths.data(), displacements.data(),
                       MPI_BYTE, &indexed_type);
      MPI_Type_commit(&indexed_type);

      MPI_Send(agents.data(), 1, indexed_type, worker_rank, 0, MPI_COMM_WORLD);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Sent local region (" + std::to_string(kRegionSize) +
          " agents) to worker " + std::to_string(worker_rank));

      MPI_Type_free(&indexed_type);
    }
  }

  const Space::Region& coordinator_region = regions_[0];
  const std::vector<int>& coordinator_indices = coordinator_region.GetIndices();
  std::vector<Agent> coordinator_agents;
  coordinator_agents.reserve(coordinator_indices.size());

  for (const int kIndex : coordinator_indices) {
    coordinator_agents.push_back(agents[kIndex]);
  }

  local_region_ = std::make_unique<ParallelABM::LocalRegion>(
      coordinator_region.GetRegionId(), std::move(coordinator_agents),
      std::vector<Agent>{}, split_function_);
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own local region (" +
      std::to_string(coordinator_indices.size()) + " agents)");
}

void MPICoordinator::ReceiveLocalRegionsFromWorkers() {
  std::vector<Agent>& agents = space_->GetAgents();

  // Receive updated regions from workers using MPI_Type_indexed
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    int region_size = 0;
    MPI_Recv(&region_size, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Received region size (" + std::to_string(region_size) +
        ") from worker " + std::to_string(worker_rank));

    if (region_size > 0) {
      const Space::Region& region = regions_[worker_rank];
      const std::vector<int>& indices = region.GetIndices();

      // Create indexed datatype to receive directly into space's agents vector
      std::vector<int> block_lengths(region_size, sizeof(Agent));
      std::vector<int> displacements(region_size);

      for (size_t i = 0; i < indices.size(); ++i) {
        displacements[i] = indices[i] * static_cast<int>(sizeof(Agent));
      }

      MPI_Datatype indexed_type = MPI_DATATYPE_NULL;
      MPI_Type_indexed(region_size, block_lengths.data(), displacements.data(),
                       MPI_BYTE, &indexed_type);
      MPI_Type_commit(&indexed_type);

      MPI_Recv(agents.data(), 1, indexed_type, worker_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Received local region (" +
          std::to_string(region_size) + " agents) from worker " +
          std::to_string(worker_rank));

      MPI_Type_free(&indexed_type);
    }
  }

  // Copy coordinator's local region back into space
  const Space::Region& coordinator_region = regions_[0];
  const std::vector<int>& coordinator_indices = coordinator_region.GetIndices();
  const std::vector<Agent>& coordinator_local_agents =
      local_region_->GetAgents();

  for (size_t i = 0; i < coordinator_indices.size(); ++i) {
    agents[coordinator_indices[i]] = coordinator_local_agents[i];
  }

  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Copied own local region (" +
      std::to_string(coordinator_local_agents.size()) + " agents) to space");
}

void MPICoordinator::SendNeighborsToWorkers() {
  // Send neighbor agents to each worker using their pre-computed regions
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    const Space::Region& region = regions_[worker_rank];
    const std::vector<Agent>& neighbor_agents = region.GetNeighbors();

    const int kNumNeighbors = static_cast<int>(neighbor_agents.size());
    MPI_Send(&kNumNeighbors, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent neighbor count (" +
        std::to_string(kNumNeighbors) + ") to worker " +
        std::to_string(worker_rank));

    if (kNumNeighbors > 0) {
      MPI_Send(neighbor_agents.data(),
               static_cast<int>(kNumNeighbors * sizeof(Agent)), MPI_BYTE,
               worker_rank, 0, MPI_COMM_WORLD);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Sent neighbors (" + std::to_string(kNumNeighbors) +
          " agents) to worker " + std::to_string(worker_rank));
    }
  }

  // Set coordinator's own neighbors from region 0
  const std::vector<Agent>& coordinator_neighbors = regions_[0].GetNeighbors();
  local_region_->SetNeighbors(coordinator_neighbors);
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own neighbors (" +
      std::to_string(coordinator_neighbors.size()) + " agents)");
}
