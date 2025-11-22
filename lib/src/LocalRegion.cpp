#include "ParallelABM/LocalRegion.h"

#include <utility>
#include <vector>

#include "ParallelABM/Agent.h"

namespace ParallelABM {

// ============================================================================
// LocalRegion Implementation
// ============================================================================

LocalRegion::LocalRegion(int region_id, std::vector<Agent> agents,
                         std::vector<Agent> neighbors,
                         SplitFunction split_function)
    : region_id_(region_id),
      agents_(std::move(agents)),
      neighbors_(std::move(neighbors)),
      split_function_(std::move(split_function)) {}

int LocalRegion::GetRegionId() const noexcept { return region_id_; }

const std::vector<Agent>& LocalRegion::GetAgents() const noexcept {
  return agents_;
}

std::vector<Agent>& LocalRegion::GetAgents() noexcept { return agents_; }

const std::vector<Agent>& LocalRegion::GetNeighbors() const noexcept {
  return neighbors_;
}

void LocalRegion::SetNeighbors(std::vector<Agent> neighbors) {
  neighbors_ = std::move(neighbors);
}

std::vector<LocalSubRegion> LocalRegion::SplitIntoSubRegions(
    int num_subregions) {
  return split_function_(*this, num_subregions);
}

// ============================================================================
// LocalSubRegion Implementation
// ============================================================================

LocalSubRegion::LocalSubRegion(std::vector<int> indices,
                               LocalRegion* local_region,
                               std::vector<Agent> neighbors)
    : indices_(std::move(indices)),
      local_region_(local_region),
      neighbors_(std::move(neighbors)) {}

const std::vector<int>& LocalSubRegion::GetIndices() const noexcept {
  return indices_;
}

const LocalRegion* LocalSubRegion::GetLocalRegion() const noexcept {
  return local_region_;
}

LocalRegion* LocalSubRegion::GetLocalRegion() noexcept { return local_region_; }

const std::vector<Agent>& LocalSubRegion::GetNeighbors() const noexcept {
  return neighbors_;
}

}  // namespace ParallelABM
