#include "ParallelABM/SimulationCPU.h"

#include <functional>
#include <thread>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/LocalRegion.h"

namespace ParallelABM {

void SimulationCPU::LaunchModel(LocalRegion* local_region) {
  const auto kNumCores = environment.GetNumberOfCPUCores();

  std::vector<LocalSubRegion> subregions =
      local_region->SplitIntoSubRegions(static_cast<int>(kNumCores));

  std::vector<std::thread> threads;
  threads.reserve(subregions.size());

  for (auto& subregion : subregions) {
    threads.emplace_back([this, &subregion]() {
      const auto& indices = subregion.GetIndices();
      LocalRegion* parent = subregion.GetLocalRegion();
      std::vector<Agent>& agents = parent->GetAgents();
      const auto& neighbors = subregion.GetNeighbors();

      std::vector<std::reference_wrapper<Agent>> agent_refs;
      agent_refs.reserve(indices.size());
      for (const int kIdx : indices) {
        agent_refs.emplace_back(std::ref(agents[kIdx]));
      }

      model.interaction_rule_(agent_refs, neighbors);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace ParallelABM
