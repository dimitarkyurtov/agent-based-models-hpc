#ifndef PARALLELABM_SPACE_H
#define PARALLELABM_SPACE_H

#include <vector>

namespace ParallelABM {
template <typename AgentT>
class LocalRegion;
template <typename AgentT>
class LocalSubRegion;
}  // namespace ParallelABM

/**
 * @class Space
 * @brief Template class defining the spatial organization of agents in the
 * simulation.
 *
 * @tparam AgentT The concrete agent type stored in this space. Must be:
 *   - Default constructible (for MPI receive operations)
 *   - Copy constructible and copy assignable (for MPI send/receive)
 *   - Preferably trivially copyable for optimal MPI performance
 *
 * The Space class represents the spatial structure where agents reside and
 * defines how they are organized positionally with respect to one another.
 * Users should inherit from this class and implement the pure virtual methods
 * according to their specific spatial requirements (e.g., 2D grid, 3D grid,
 * continuous space, network topology, etc.).
 *
 * The library uses this interface to efficiently distribute the simulation
 * workload across multiple computational nodes via MPI while minimizing
 * communication overhead between regions.
 *
 * Template Benefits:
 * - Direct value storage eliminates pointer indirection overhead
 * - Enables zero-copy MPI communication (send/recv contiguous arrays)
 * - Compile-time type safety prevents object slicing
 * - Better cache locality from contiguous memory layout
 */
template <typename AgentT>
class Space {
 public:
  /**
   * @class Region
   * @brief Represents a partition of the space containing a subset of agents.
   *
   * A Region defines a contiguous or logical subset of agents within the space
   * that can be processed independently. Regions are used for distributing
   * workload across computational nodes in parallel simulations.
   */
  class Region {
   private:
    /// Indices of agents in the space's agents vector that belong to this
    /// region
    std::vector<int> indices_;

    /// Unique identifier for this region
    int region_id_ = 0;

    /// Neighbor agents from other regions that interact with agents in this
    /// region (stored by value for efficient MPI communication)
    std::vector<AgentT> neighbors_;

   public:
    /**
     * @brief Default constructor.
     */
    Region() = default;

    /**
     * @brief Constructs a region with specified ID and agent indices.
     * @param region_id The unique identifier for this region
     * @param agent_indices The indices of agents that belong to this region
     */
    Region(int region_id, std::vector<int> agent_indices)
        : indices_(std::move(agent_indices)),
          region_id_(region_id),
          neighbors_() {}

    /**
     * @brief Copy constructor.
     */
    Region(const Region&) = default;

    /**
     * @brief Move constructor.
     */
    Region(Region&&) noexcept = default;

    /**
     * @brief Copy assignment operator.
     */
    Region& operator=(const Region&) = default;

    /**
     * @brief Move assignment operator.
     */
    Region& operator=(Region&&) noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Region() = default;

    /**
     * @brief Retrieves the agent indices belonging to this region.
     * @return Const reference to the vector of agent indices
     */
    [[nodiscard]] const std::vector<int>& GetIndices() const {
      return indices_;
    }

    /**
     * @brief Retrieves the unique identifier of this region.
     * @return The region ID
     */
    [[nodiscard]] int GetRegionId() const { return region_id_; }

    /**
     * @brief Retrieves the neighbor agents from other regions.
     * @return Const reference to the vector of neighbor agents
     */
    [[nodiscard]] const std::vector<AgentT>& GetNeighbors() const {
      return neighbors_;
    }

    /**
     * @brief Updates the neighbor agents with new values.
     * @param new_neighbors Vector of updated neighbor agents
     *
     * This method allows the Space implementation to refresh neighbor data
     * dynamically, ensuring that neighbor information reflects the current
     * simulation state rather than stale data from region creation.
     */
    void SetNeighbors(std::vector<AgentT> new_neighbors) {
      neighbors_ = std::move(new_neighbors);
    }
  };

 protected:
  /**
   * @brief Collection of all agents in the simulation.
   *
   * This vector contains agents stored by value in contiguous memory.
   * The ordering of agents in this vector is significant, as it determines how
   * the space can be partitioned for distributed computation.
   *
   * Benefits of value storage:
   * - Zero pointer indirection - better cache performance
   * - Contiguous memory layout - optimal for vectorization
   * - Direct MPI communication - no serialization overhead
   *
   * The specific ordering strategy depends on the spatial implementation:
   * - For 2D grids: typically row-major or column-major order
   * - For 3D grids: layer-by-layer, row-by-row ordering
   * - For other structures: any ordering that allows efficient neighbor lookups
   */
  std::vector<AgentT> agents{};

 public:
  /**
   * @brief Default constructor.
   */
  Space() = default;

  /**
   * @brief Copy constructor.
   */
  Space(const Space&) = default;

  /**
   * @brief Copy assignment operator.
   */
  Space& operator=(const Space&) = default;

  /**
   * @brief Move constructor.
   */
  Space(Space&&) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  Space& operator=(Space&&) noexcept = default;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~Space() = default;

  /**
   * @brief Populate the agents vector with initial simulation data.
   *
   * This method should be implemented by derived classes to fill the agents
   * vector with the initial population of agents for the simulation.
   * Implementation strategies may include reading from files, generating
   * procedurally, or other initialization logic.
   *
   * @note This method is called only on the leader process (rank 0) during
   * simulation setup.
   */
  virtual void Initialize() = 0;

  /**
   * @brief Partitions the space into multiple regions for distributed
   * processing.
   *
   * This method divides the space and its agents into the specified number of
   * regions, where each region can be processed independently on separate
   * computational nodes. The implementation should strive to create regions
   * that minimize cross-region communication while maintaining balanced
   * workload distribution.
   *
   * @param num_regions The number of regions to create (typically matches the
   *                    number of MPI processes)
   *
   * @return A vector of Region objects, where each Region contains:
   *         - A unique region_id (0 to num_regions-1)
   *         - A vector of indices referencing agents in the space's agents
   * vector
   *         - Copies of neighbor agents for interaction calculations
   *
   * @details
   * The goal is to partition the space such that agents within the same region
   * have high interaction locality (many neighbors within the same region) and
   * minimal dependencies on agents in other regions. This reduces the amount of
   * data that must be communicated between MPI processes during simulation.
   *
   * Implementation considerations:
   * - For grid-based spaces: partition along natural boundaries (rows, columns,
   *   layers) to minimize edge-cutting
   * - For continuous spaces: use spatial decomposition techniques (e.g.,
   * recursive bisection, k-d trees)
   * - For networks: apply graph partitioning algorithms to minimize edge cuts
   * - Balance region sizes to ensure approximately equal workload per process
   *
   * Example - 2D Grid (10x10, row-major order, 4 regions):
   * - Region 0: rows 0-2 (indices 0-29)
   * - Region 1: rows 3-4 (indices 30-49)
   * - Region 2: rows 5-7 (indices 50-79)
   * - Region 3: rows 8-9 (indices 80-99)
   *
   * @note The sum of all agent indices across all regions should equal the
   * total number of agents, with each agent appearing in exactly one region.
   */
  [[nodiscard]] virtual std::vector<Region> SplitIntoRegions(
      int num_regions) const = 0;

  /**
   * @brief Subdivide a local region for parallel processing.
   *
   * Splits a LocalRegion into subregions for fine-grained parallelism
   * (e.g., CPU threads, GPU devices). The splitting strategy should
   * distribute agents efficiently based on the spatial structure.
   *
   * @param region The local region to subdivide
   * @param num_subregions Number of subregions to create
   * @return Vector of LocalSubRegion objects for parallel processing
   */
  virtual std::vector<ParallelABM::LocalSubRegion<AgentT>> SplitLocalRegion(
      ParallelABM::LocalRegion<AgentT>& region, int num_subregions) const = 0;

  /**
   * @brief Retrieves current neighbor agents for a specific region.
   *
   * This method calculates and returns the neighbor agents for the specified
   * region based on the current state of the simulation. Unlike the neighbors
   * stored in Region during SplitIntoRegions, this method provides fresh
   * neighbor data that reflects the current agent states.
   *
   * @param region Reference to the region whose neighbors should be retrieved
   * @return Vector of neighbor agents with current state
   *
   * @details
   * The implementation should:
   * 1. Identify which agents from other regions interact with the target region
   * 2. Collect current agent data for those neighbors from the agents vector
   * 3. Return a new vector containing copies of those neighbor agents
   *
   * This method is called each timestep by the MPICoordinator to ensure
   * that distributed computations have access to up-to-date neighbor
   * information for accurate simulation results.
   */
  [[nodiscard]] virtual std::vector<AgentT> GetRegionNeighbours(
      const Region& region) const = 0;

  /**
   * @brief Provides access to the agents vector for derived classes and the
   * library.
   *
   * @return Reference to the vector containing all agents in the simulation.
   */
  std::vector<AgentT>& GetAgents() { return agents; }

  /**
   * @brief Provides const access to the agents vector.
   *
   * @return Const reference to the vector containing all agents in the
   * simulation.
   */
  [[nodiscard]] const std::vector<AgentT>& GetAgents() const { return agents; }
};

#endif  // PARALLELABM_SPACE_H
