#ifndef PARALLELABM_SIMULATIONCPU_INL
#define PARALLELABM_SIMULATIONCPU_INL

#include <functional>
#include <future>
#include <vector>

namespace ParallelABM {

template <typename AgentT>
void SimulationCPU<AgentT>::Setup() {
  const auto kNumCores = this->environment.GetNumberOfCPUCores();
  LocalRegion<AgentT>* local_region = this->mpi_worker_->GetLocalRegion();

  subregions_ = this->space_->SplitLocalRegion(
      *local_region, static_cast<int>(kNumCores));

  thread_pool_ = std::make_unique<BS::thread_pool<>>(kNumCores);
}

template <typename AgentT>
void SimulationCPU<AgentT>::LaunchModel(
    LocalRegion<AgentT>* /*local_region*/) {
  // Pre-compute all neighbor data sequentially to avoid race conditions
  std::vector<std::vector<AgentT>> precomputed_neighbors;
  precomputed_neighbors.reserve(subregions_.size());

  for (auto& subregion : subregions_) {
    LocalRegion<AgentT>* parent = subregion.GetLocalRegion();
    std::vector<AgentT>& agents = parent->GetAgents();
    const auto& parent_neighbors = parent->GetNeighbors();
    const auto& neighbor_indices = subregion.GetNeighborIndices();

    std::vector<AgentT> neighbors;
    neighbors.reserve(neighbor_indices.size());
    for (const int kNeighborIdx : neighbor_indices) {
      if (kNeighborIdx >= 0) {
        // Positive index: reference to agent in parent region
        neighbors.push_back(agents[kNeighborIdx]);
      } else {
        // Negative index: reference to parent's neighbor
        // -1 maps to parent_neighbors[0], -2 to parent_neighbors[1], etc.
        const auto kParentNeighborIdx = static_cast<size_t>(-kNeighborIdx - 1);
        neighbors.push_back(parent_neighbors[kParentNeighborIdx]);
      }
    }

    precomputed_neighbors.push_back(std::move(neighbors));
  }

  // Submit tasks to thread pool using pre-computed data
  thread_pool_->detach_loop(
      static_cast<size_t>(0), subregions_.size(),
      [this, &precomputed_neighbors](const size_t kI) {
        auto& subregion = subregions_[kI];
        const auto& indices = subregion.GetIndices();
        LocalRegion<AgentT>* parent = subregion.GetLocalRegion();
        std::vector<AgentT>& agents = parent->GetAgents();

        // Create reference wrappers for agents
        std::vector<std::reference_wrapper<AgentT>> agent_refs;
        agent_refs.reserve(indices.size());
        for (const int kIdx : indices) {
          agent_refs.emplace_back(std::ref(agents[kIdx]));
        }

        ParallelABM::Logger::GetInstance().Info(
            "Subregion " + std::to_string(kI) + ": Processing " +
            std::to_string(agent_refs.size()) + " agents with " +
            std::to_string(precomputed_neighbors[kI].size()) + " neighbors");

        this->model->ComputeInteractions(agent_refs, precomputed_neighbors[kI]);
      });

  // Wait for all tasks to complete
  thread_pool_->wait();
}

}  // namespace ParallelABM

#endif  // PARALLELABM_SIMULATIONCPU_INL
