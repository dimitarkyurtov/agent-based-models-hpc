#ifndef PARALLELABM_MPICOORDINATOR_INL
#define PARALLELABM_MPICOORDINATOR_INL

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPIWorker.h"
#include "ParallelABM/Space.h"

template <typename AgentT>
MPICoordinator<AgentT>::MPICoordinator(int rank, int num_processes,
                                       std::shared_ptr<Space<AgentT>> space)
    : MPIWorker<AgentT>(rank), num_processes_(num_processes), space_(space) {
  this->space_->Initialize();

  regions_ = space_->SplitIntoRegions(num_processes_);
  ParallelABM::Logger::GetInstance().Info("MPICoordinator: Split space into " +
                                          std::to_string(regions_.size()) +
                                          " regions");
}

template <typename AgentT>
void MPICoordinator<AgentT>::SendLocalRegionsToWorkers() {
  std::vector<AgentT>& agents = space_->GetAgents();

  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    const typename Space<AgentT>::Region& region = regions_[worker_rank];
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
      // Gather selected agents into contiguous buffer
      std::vector<AgentT> temp_agents;
      temp_agents.reserve(indices.size());
      for (const int kIndex : indices) {
        temp_agents.push_back(agents[kIndex]);
      }

      // Send contiguous buffer (zero-copy from temp buffer)
      MPI_Send(temp_agents.data(),
               static_cast<int>(temp_agents.size() * sizeof(AgentT)), MPI_BYTE,
               worker_rank, 0, MPI_COMM_WORLD);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Sent local region (" + std::to_string(kRegionSize) +
          " agents) to worker " + std::to_string(worker_rank));
    }
  }

  const typename Space<AgentT>::Region& coordinator_region = regions_[0];
  const std::vector<int>& coordinator_indices =
      coordinator_region.GetIndices();
  std::vector<AgentT> coordinator_agents;
  coordinator_agents.reserve(coordinator_indices.size());

  for (const int kIndex : coordinator_indices) {
    coordinator_agents.push_back(agents[kIndex]);
  }

  this->local_region_ = std::make_unique<ParallelABM::LocalRegion<AgentT>>(
      coordinator_region.GetRegionId(), std::move(coordinator_agents),
      std::vector<AgentT>{});
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own local region (" +
      std::to_string(coordinator_indices.size()) + " agents)");
}

template <typename AgentT>
void MPICoordinator<AgentT>::ReceiveLocalRegionsFromWorkers() {
  std::vector<AgentT>& agents = space_->GetAgents();

  // Receive updated regions from workers
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    int region_size = 0;
    MPI_Recv(&region_size, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Received region size (" + std::to_string(region_size) +
        ") from worker " + std::to_string(worker_rank));

    if (region_size > 0) {
      const typename Space<AgentT>::Region& region = regions_[worker_rank];
      const std::vector<int>& indices = region.GetIndices();

      // Receive agents as contiguous array (zero-copy)
      std::vector<AgentT> temp_agents(region_size);
      MPI_Recv(temp_agents.data(),
               static_cast<int>(region_size * sizeof(AgentT)), MPI_BYTE,
               worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Copy received agents back into space's agents vector
      for (size_t i = 0; i < indices.size(); ++i) {
        agents[indices[i]] = temp_agents[i];
      }

      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Received local region (" +
          std::to_string(region_size) + " agents) from worker " +
          std::to_string(worker_rank));
    }
  }

  // Copy coordinator's local region back into space
  const typename Space<AgentT>::Region& coordinator_region = regions_[0];
  const std::vector<int>& coordinator_indices =
      coordinator_region.GetIndices();
  const std::vector<AgentT>& coordinator_local_agents =
      this->local_region_->GetAgents();

  for (size_t i = 0; i < coordinator_indices.size(); ++i) {
    agents[coordinator_indices[i]] = coordinator_local_agents[i];
  }

  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Copied own local region (" +
      std::to_string(coordinator_local_agents.size()) + " agents) to space");
}

template <typename AgentT>
void MPICoordinator<AgentT>::SendNeighborsToWorkers() {
  // Update neighbor data for each region based on current simulation state
  // and send to workers
  for (int worker_rank = 1; worker_rank < num_processes_; ++worker_rank) {
    typename Space<AgentT>::Region& region = regions_[worker_rank];

    // Retrieve fresh neighbor data from space based on current agent states
    std::vector<AgentT> updated_neighbors =
        space_->GetRegionNeighbours(region);

    // Update the region's stored neighbors with current data
    region.SetNeighbors(updated_neighbors);

    const int kNumNeighbors = static_cast<int>(updated_neighbors.size());
    MPI_Send(&kNumNeighbors, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
    ParallelABM::Logger::GetInstance().Info(
        "MPICoordinator: Sent neighbor count (" +
        std::to_string(kNumNeighbors) + ") to worker " +
        std::to_string(worker_rank));

    if (kNumNeighbors > 0) {
      // Send neighbors as contiguous array (zero-copy)
      MPI_Send(updated_neighbors.data(),
               static_cast<int>(kNumNeighbors * sizeof(AgentT)), MPI_BYTE,
               worker_rank, 0, MPI_COMM_WORLD);
      ParallelABM::Logger::GetInstance().Info(
          "MPICoordinator: Sent neighbors (" + std::to_string(kNumNeighbors) +
          " agents) to worker " + std::to_string(worker_rank));
    }
  }

  // Update and set coordinator's own neighbors from region 0
  typename Space<AgentT>::Region& coordinator_region = regions_[0];
  std::vector<AgentT> coordinator_neighbors =
      space_->GetRegionNeighbours(coordinator_region);

  // Update the coordinator's region with fresh neighbors
  coordinator_region.SetNeighbors(coordinator_neighbors);

  this->local_region_->SetNeighbors(std::move(coordinator_neighbors));
  ParallelABM::Logger::GetInstance().Debug(
      "MPICoordinator: Set own neighbors (" +
      std::to_string(this->local_region_->GetNeighbors().size()) +
      " agents)");
}

#endif  // PARALLELABM_MPICOORDINATOR_INL
