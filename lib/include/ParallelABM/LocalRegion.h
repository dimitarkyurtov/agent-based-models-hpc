#ifndef PARALLELABM_LOCALREGION_H
#define PARALLELABM_LOCALREGION_H

#include <functional>
#include <vector>

namespace ParallelABM {

// Forward declaration for the nested class
template <typename AgentT>
class LocalSubRegion;
template <typename AgentT>
class LocalRegion;

/**
 * @class LocalRegion
 * @brief Template class representing a local region for parallel agent-based
 * modeling.
 *
 * @tparam AgentT The concrete agent type stored in this region
 *
 * LocalRegion encapsulates a subdivision of the global simulation space
 * assigned to a computational node. It contains:
 * - The agents that this region is responsible for processing and updating
 * - Neighboring agents from adjacent regions (read-only for interaction
 * calculations)
 * - A region identifier assigned by the coordinator
 *
 * PURPOSE:
 * ========
 * The LocalRegion serves as the fundamental unit of work distribution in the
 * parallel simulation:
 * 1. Receives agent data from the MPI coordinator
 * 2. Stores agents and their neighbors for local processing
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
 * - The LocalSubRegion instances during parallel execution
 *
 * NOTE: LocalRegion does NOT perform splitting. All splitting logic is
 * delegated to the Space class via Space::SplitLocalRegion().
 *
 * TEMPLATE BENEFITS:
 * ==================
 * - Direct value storage for agents (no pointer indirection)
 * - Efficient MPI communication (send/recv contiguous arrays)
 * - Type safety prevents object slicing
 */
template <typename AgentT>
class LocalRegion {
 public:
  /**
   * @brief Construct a LocalRegion with specified parameters
   * @param region_id Unique identifier for this region assigned by coordinator
   * @param agents Vector of agents this region is responsible for
   * @param neighbors Vector of neighboring agents from adjacent regions
   */
  LocalRegion(int region_id, std::vector<AgentT> agents,
              std::vector<AgentT> neighbors);

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
  LocalRegion(LocalRegion&&) noexcept = default;

  /**
   * @brief Default move assignment operator
   */
  LocalRegion& operator=(LocalRegion&&) noexcept = default;

  /**
   * @brief Default destructor
   */
  ~LocalRegion() = default;

  /**
   * @brief Get the region ID
   * @return Region identifier assigned by the coordinator
   */
  [[nodiscard]] int GetRegionId() const noexcept;

  /**
   * @brief Get the agents vector
   * @return Const reference to the agents this region processes
   */
  [[nodiscard]] const std::vector<AgentT>& GetAgents() const noexcept;

  /**
   * @brief Get mutable access to the agents vector
   * @return Mutable reference to the agents this region processes
   */
  [[nodiscard]] std::vector<AgentT>& GetAgents() noexcept;

  /**
   * @brief Get the neighboring agents vector
   * @return Const reference to the neighboring agents (read-only)
   */
  [[nodiscard]] const std::vector<AgentT>& GetNeighbors() const noexcept;

  /**
   * @brief Update the neighboring agents vector
   * @param neighbors New vector of neighboring agents from adjacent regions
   */
  void SetNeighbors(std::vector<AgentT> neighbors);

 protected:
  int region_id_;                  ///< Region identifier from coordinator
  std::vector<AgentT> agents_;     ///< Agents to process and update
  std::vector<AgentT> neighbors_;  ///< Neighboring agents (read-only)
};

/**
 * @class LocalSubRegion
 * @brief Template class representing a subdivision of a LocalRegion for
 * fine-grained parallel processing.
 *
 * @tparam AgentT The concrete agent type stored in this subregion
 *
 * LocalSubRegion is created by Space::SplitLocalRegion() to enable parallel
 * execution at the local level (e.g., CPU threads, GPU devices). Each
 * subregion:
 * - Contains indices into the parent LocalRegion's agent vector
 * - Maintains a reference to the parent for accessing agent data
 * - Has its own references to relevant neighbors for interaction calculations
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
 * - neighbors_: References to neighboring agents for calculations
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
template <typename AgentT>
class LocalSubRegion {
 public:
  /**
   * @brief Construct a LocalSubRegion with specified parameters
   * @param indices Agent indices within the parent region to process
   * @param local_region Non-owning pointer to the parent LocalRegion
   * @param neighbor_indices Indices to neighboring agents for this subregion.
   *        Positive indices (0, 1, 2, ...) refer to agents in the parent
   * region. Negative indices (-1, -2, -3, ...) refer to parent's neighbors
   * where -1 = parent_neighbors[0], -2 = parent_neighbors[1], etc.
   */
  LocalSubRegion(std::vector<int> indices, LocalRegion<AgentT>* local_region,
                 std::vector<int> neighbor_indices);

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
   * @brief Get the parent LocalRegion (const)
   * @return Non-owning const pointer to the parent region
   */
  [[nodiscard]] const LocalRegion<AgentT>* GetLocalRegion() const noexcept;

  /**
   * @brief Get the parent LocalRegion (mutable)
   * @return Non-owning pointer to the parent region
   */
  [[nodiscard]] LocalRegion<AgentT>* GetLocalRegion() noexcept;

  /**
   * @brief Get the neighbor indices for this subregion
   * @return Const reference to the neighbor_indices vector.
   *         Positive indices refer to parent region agents.
   *         Negative indices refer to parent region neighbors.
   */
  [[nodiscard]] const std::vector<int>& GetNeighborIndices() const noexcept;

 private:
  std::vector<int> indices_;           ///< Indices into parent's agent vector
  LocalRegion<AgentT>* local_region_;  ///< Non-owning pointer to parent region
  std::vector<int> neighbor_indices_;  ///< Indices to neighbors (+ = local
                                       ///< agents, - = parent neighbors)
};

}  // namespace ParallelABM

// Include implementation
#include "LocalRegion.inl"

#endif  // PARALLELABM_LOCALREGION_H
