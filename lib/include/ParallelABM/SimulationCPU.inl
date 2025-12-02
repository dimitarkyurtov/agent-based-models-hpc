#ifndef PARALLELABM_SIMULATIONCPU_INL
#define PARALLELABM_SIMULATIONCPU_INL

#include <functional>
#include <thread>
#include <vector>

namespace ParallelABM {

template <typename AgentT>
void SimulationCPU<AgentT>::LaunchModel(LocalRegion<AgentT>* local_region) {
  const auto kNumCores = this->environment.GetNumberOfCPUCores();

  std::vector<LocalSubRegion<AgentT>> subregions =
      this->space_->SplitLocalRegion(*local_region, static_cast<int>(kNumCores));

  std::vector<std::thread> threads;
  threads.reserve(subregions.size());

  for (auto& subregion : subregions) {
    threads.emplace_back([this, &subregion]() {
      const auto& indices = subregion.GetIndices();
      LocalRegion<AgentT>* parent = subregion.GetLocalRegion();
      std::vector<AgentT>& agents = parent->GetAgents();
      const auto& neighbors = subregion.GetNeighbors();

      std::vector<std::reference_wrapper<AgentT>> agent_refs;
      agent_refs.reserve(indices.size());
      for (const int kIdx : indices) {
        agent_refs.emplace_back(std::ref(agents[kIdx]));
      }

      this->model->ComputeInteractions(agent_refs, neighbors);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace ParallelABM

#endif  // PARALLELABM_SIMULATIONCPU_INL
