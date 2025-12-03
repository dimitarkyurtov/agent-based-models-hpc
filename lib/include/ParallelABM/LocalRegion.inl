#ifndef PARALLELABM_LOCALREGION_INL
#define PARALLELABM_LOCALREGION_INL

namespace ParallelABM {

// LocalRegion implementation
template <typename AgentT>
LocalRegion<AgentT>::LocalRegion(int region_id, std::vector<AgentT> agents,
                                  std::vector<AgentT> neighbors)
    : region_id_(region_id),
      agents_(std::move(agents)),
      neighbors_(std::move(neighbors)) {}

template <typename AgentT>
int LocalRegion<AgentT>::GetRegionId() const noexcept {
  return region_id_;
}

template <typename AgentT>
const std::vector<AgentT>& LocalRegion<AgentT>::GetAgents() const noexcept {
  return agents_;
}

template <typename AgentT>
std::vector<AgentT>& LocalRegion<AgentT>::GetAgents() noexcept {
  return agents_;
}

template <typename AgentT>
const std::vector<AgentT>& LocalRegion<AgentT>::GetNeighbors() const noexcept {
  return neighbors_;
}

template <typename AgentT>
void LocalRegion<AgentT>::SetNeighbors(std::vector<AgentT> neighbors) {
  neighbors_ = std::move(neighbors);
}

// LocalSubRegion implementation
template <typename AgentT>
LocalSubRegion<AgentT>::LocalSubRegion(
    std::vector<int> indices, LocalRegion<AgentT>* local_region,
    std::vector<int> neighbor_indices)
    : indices_(std::move(indices)),
      local_region_(local_region),
      neighbor_indices_(std::move(neighbor_indices)) {}

template <typename AgentT>
const std::vector<int>& LocalSubRegion<AgentT>::GetIndices() const noexcept {
  return indices_;
}

template <typename AgentT>
const LocalRegion<AgentT>* LocalSubRegion<AgentT>::GetLocalRegion()
    const noexcept {
  return local_region_;
}

template <typename AgentT>
LocalRegion<AgentT>* LocalSubRegion<AgentT>::GetLocalRegion() noexcept {
  return local_region_;
}

template <typename AgentT>
const std::vector<int>&
LocalSubRegion<AgentT>::GetNeighborIndices() const noexcept {
  return neighbor_indices_;
}

}  // namespace ParallelABM

#endif  // PARALLELABM_LOCALREGION_INL
