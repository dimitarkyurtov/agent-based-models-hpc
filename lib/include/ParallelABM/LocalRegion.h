#ifndef PARALLELABM_LOCALREGION_H
#define PARALLELABM_LOCALREGION_H

#include <functional>
#include <vector>

#include "ParallelABM/Agent.h"

namespace ParallelABM {

// Forward declaration for the nested class
class LocalSubRegion;
class LocalRegion;

// Function type for splitting a region into subregions
using SplitFunction =
    std::function<std::vector<LocalSubRegion>(const LocalRegion&)>;

/**
 * @class LocalRegion
 * @brief Represents a local region for parallel agent-based modeling with
 * configurable splitting strategy.
 *
 * LocalRegion encapsulates a subdivision of the global simulation space
 * assigned to a computational node. It contains:
 * - The agents that this region is responsible for processing and updating
 * - Neighboring agents from adjacent regions (read-only for interaction
 * calculations)
 * - A region identifier assigned by the coordinator
 * - A splitting function for subdividing the region into subregions
 *
 * PURPOSE:
 * ========
 * The LocalRegion serves as the fundamental unit of work distribution in the
 * parallel simulation:
 * 1. Receives agent data from the MPI coordinator
 * 2. Provides an interface for further local subdivisions (CPU threads, GPU
 * devices)
 * 3. Enables users to provide custom workload splitting strategies via function
 *
 * USER EXTENSIBILITY:
 * ===================
 * Users provide a SplitFunction during construction to define how the region's
 * workload should be subdivided for local parallel processing. This enables
 * flexible strategies such as:
 * - Splitting by agent count for CPU thread pools
 * - Spatial partitioning for GPU device distribution
 * - Hybrid approaches combining different criteria
 *
 * AGENT OWNERSHIP:
 * ================
 * - agents_: These agents belong to this region and MUST be updated
 * - neighbors_: Read-only agents from adjacent regions for interaction
 * calculations
 *
 * THREADING CONSIDERATIONS:
 * =========================
 * This class itself is not thread-safe. Synchronization must be handled by:
 * - The caller when accessing shared LocalRegion instances
 * - The provided splitting function when called in parallel contexts
 * - The LocalSubRegion instances during parallel execution
 *
 * EXAMPLE USAGE:
 * ==============
 * auto splitter = [](const LocalRegion& region) {
 *   std::vector<LocalSubRegion> subregions;
 *   const int kThreadCount = 4;
 *   const auto& agents = region.GetAgents();
 *   const int kAgentsPerThread = agents.size() / kThreadCount;
 *
 *   for (int i = 0; i < kThreadCount; ++i) {
 *     std::vector<int> indices;
 *     for (int j = i * kAgentsPerThread;
 *          j < (i + 1) * kAgentsPerThread && j < agents.size(); ++j) {
 *       indices.push_back(j);
 *     }
 *     subregions.emplace_back(indices, &region, region.GetNeighbors());
 *   }
 *   return subregions;
 * };
 * LocalRegion region(0, std::move(agents), std::move(neighbors), splitter);
 */
class LocalRegion {
 public:
  /**
   * @brief Construct a LocalRegion with specified parameters
   * @param region_id Unique identifier for this region assigned by coordinator
   * @param agents Vector of agents this region is responsible for
   * @param neighbors Vector of neighboring agents from adjacent regions
   * @param split_function Function to split region into subregions
   */
  LocalRegion(int region_id, std::vector<Agent> agents,
              std::vector<Agent> neighbors, SplitFunction split_function);

  /**
   * @brief Default copy constructor
   */
  LocalRegion(const LocalRegion&) = default;

  /**
   * @brief Default copy assignment operator
   */
  LocalRegion& operator=(const LocalRegion&) = default;

  /**
   * @brief Default move constructor
   */
  LocalRegion(LocalRegion&&) = default;

  /**
   * @brief Default move assignment operator
   */
  LocalRegion& operator=(LocalRegion&&) = default;

  /**
   * @brief Default destructor
   */
  ~LocalRegion() = default;

  /**
   * @brief Split this region into subregions for local parallel processing
   *
   * This method invokes the user-provided splitting function to define
   * the workload splitting strategy. The function should:
   * - Analyze the agents_ vector to determine optimal subdivision
   * - Create LocalSubRegion instances with appropriate index ranges
   * - Distribute neighbors_ to subregions as needed for calculations
   * - Return a vector of LocalSubRegion instances ready for parallel execution
   *
   * @return Vector of LocalSubRegion instances representing the subdivision
   */
  [[nodiscard]] std::vector<LocalSubRegion> SplitIntoSubRegions() const;

  /**
   * @brief Get the region ID
   * @return Region identifier assigned by the coordinator
   */
  [[nodiscard]] int GetRegionId() const noexcept;

  /**
   * @brief Get the agents vector
   * @return Const reference to the agents this region processes
   */
  [[nodiscard]] const std::vector<Agent>& GetAgents() const noexcept;

  /**
   * @brief Get mutable access to the agents vector
   * @return Mutable reference to the agents this region processes
   */
  [[nodiscard]] std::vector<Agent>& GetAgents() noexcept;

  /**
   * @brief Get the neighboring agents vector
   * @return Const reference to the neighboring agents (read-only)
   */
  [[nodiscard]] const std::vector<Agent>& GetNeighbors() const noexcept;

  /**
   * @brief Update the neighboring agents vector
   * @param neighbors New vector of neighboring agents from adjacent regions
   */
  void SetNeighbors(std::vector<Agent> neighbors);

 protected:
  int region_id_;                 ///< Region identifier from coordinator
  std::vector<Agent> agents_;     ///< Agents to process and update
  std::vector<Agent> neighbors_;  ///< Neighboring agents (read-only)
  SplitFunction split_function_;  ///< Function to split region into subregions
};

/**
 * @class LocalSubRegion
 * @brief Represents a subdivision of a LocalRegion for fine-grained parallel
 * processing.
 *
 * LocalSubRegion is created by LocalRegion::SplitIntoSubRegions() to enable
 * parallel execution at the local level (e.g., CPU threads, GPU devices).
 * Each subregion:
 * - Contains indices into the parent LocalRegion's agent vector
 * - Maintains a reference to the parent for accessing agent data
 * - Has its own copy of relevant neighbors for interaction calculations
 *
 * PURPOSE:
 * ========
 * LocalSubRegion enables fine-grained parallelism by:
 * 1. Dividing the parent region's agents among multiple execution units
 * 2. Providing isolated access to agent subsets for thread-safe processing
 * 3. Supporting flexible distribution strategies (spatial, count-based, etc.)
 *
 * MEMORY MANAGEMENT:
 * ==================
 * - indices_: Owned by this instance, defines which agents to process
 * - local_region_: Non-owning pointer to parent (must outlive this instance)
 * - neighbors_: Owned copy of neighbors needed for this subregion's
 * calculations
 *
 * LIFETIME CONSIDERATIONS:
 * ========================
 * LocalSubRegion instances MUST NOT outlive their parent LocalRegion.
 * The parent LocalRegion must remain valid for the entire lifetime of all
 * LocalSubRegion instances created from it.
 *
 * THREAD SAFETY:
 * ==============
 * Multiple LocalSubRegion instances can be processed in parallel safely if:
 * - Their indices_ ranges do not overlap (ensured by proper splitting)
 * - The parent LocalRegion remains unmodified during parallel execution
 * - Synchronization is applied when merging results back to the parent
 */
class LocalSubRegion {
 public:
  /**
   * @brief Construct a LocalSubRegion with specified parameters
   * @param indices Agent indices within the parent region to process
   * @param local_region Non-owning pointer to the parent LocalRegion
   * @param neighbors Neighboring agents relevant to this subregion
   */
  LocalSubRegion(std::vector<int> indices, const LocalRegion* local_region,
                 std::vector<Agent> neighbors);

  /**
   * @brief Default copy constructor
   */
  LocalSubRegion(const LocalSubRegion&) = default;

  /**
   * @brief Default copy assignment operator
   */
  LocalSubRegion& operator=(const LocalSubRegion&) = default;

  /**
   * @brief Default move constructor
   */
  LocalSubRegion(LocalSubRegion&&) noexcept = default;

  /**
   * @brief Default move assignment operator
   */
  LocalSubRegion& operator=(LocalSubRegion&&) noexcept = default;

  /**
   * @brief Default destructor
   */
  ~LocalSubRegion() = default;

  /**
   * @brief Get the agent indices to process
   * @return Const reference to the indices vector
   */
  [[nodiscard]] const std::vector<int>& GetIndices() const noexcept;

  /**
   * @brief Get the parent LocalRegion
   * @return Non-owning pointer to the parent region
   */
  [[nodiscard]] const LocalRegion* GetLocalRegion() const noexcept;

  /**
   * @brief Get the neighboring agents for this subregion
   * @return Const reference to the neighbors vector
   */
  [[nodiscard]] const std::vector<Agent>& GetNeighbors() const noexcept;

 private:
  std::vector<int> indices_;         ///< Indices into parent's agent vector
  const LocalRegion* local_region_;  ///< Non-owning pointer to parent region
  std::vector<Agent> neighbors_;     ///< Neighboring agents for calculations
};

}  // namespace ParallelABM

#endif  // PARALLELABM_LOCALREGION_H
