#ifndef PARALLELABM_SPACE_H
#define PARALLELABM_SPACE_H

#include <vector>

#include "ParallelABM/Agent.h"

/**
 * @class Space
 * @brief Abstract interface defining the spatial organization of agents in the
 * simulation.
 *
 * The Space class represents the spatial structure where agents reside and
 * defines how they are organized positionally with respect to one another. This
 * is an abstract interface that users must implement based on their specific
 * spatial requirements (e.g., 2D grid, 3D grid, continuous space, network
 * topology, etc.).
 *
 * The library uses this interface to efficiently distribute the simulation
 * workload across multiple computational nodes via MPI while minimizing
 * communication overhead between regions.
 *
 * Users should inherit from this class and implement the pure virtual methods
 * according to their specific spatial structure requirements.
 */
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
    int region_id_;

    /// Neighbor agents from other regions that interact with agents in this
    /// region
    std::vector<Agent> neighbors_;

   public:
    /**
     * @brief Default constructor.
     */
    Region() : region_id_(0) {}

    /**
     * @brief Constructs a region with specified ID, agent indices, and neighbor
     * agents.
     * @param region_id The unique identifier for this region
     * @param agent_indices The indices of agents that belong to this region
     * @param neighbor_agents Agents from other regions that interact with this
     * region's agents
     */
    Region(int region_id, std::vector<int> agent_indices,
           std::vector<Agent> neighbor_agents)
        : indices_(std::move(agent_indices)),
          region_id_(region_id),
          neighbors_(std::move(neighbor_agents)) {}

    /**
     * @brief Copy constructor.
     */
    Region(const Region&) = default;

    /**
     * @brief Move constructor.
     */
    Region(Region&&) = default;

    /**
     * @brief Copy assignment operator.
     */
    Region& operator=(const Region&) = default;

    /**
     * @brief Move assignment operator.
     */
    Region& operator=(Region&&) = default;

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
    [[nodiscard]] const std::vector<Agent>& GetNeighbors() const {
      return neighbors_;
    }
  };

 protected:
  /**
   * @brief Collection of all agents in the simulation.
   *
   * This vector contains all agents that exist within this spatial structure.
   * The ordering of agents in this vector is significant, as it determines how
   * the space can be partitioned for distributed computation.
   *
   * The specific ordering strategy depends on the spatial implementation:
   * - For 2D grids: typically row-major or column-major order
   * - For 3D grids: layer-by-layer, row-by-row ordering
   * - For other structures: any ordering that allows efficient neighbor lookups
   */
  std::vector<Agent> agents;

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
  Space(Space&&) = default;

  /**
   * @brief Move assignment operator.
   */
  Space& operator=(Space&&) = default;

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
   * @brief Provides access to the agents vector for derived classes and the
   * library.
   *
   * @return Reference to the vector containing all agents in the simulation.
   */
  std::vector<Agent>& GetAgents() { return agents; }

  /**
   * @brief Provides const access to the agents vector.
   *
   * @return Const reference to the vector containing all agents in the
   * simulation.
   */
  [[nodiscard]] const std::vector<Agent>& GetAgents() const { return agents; }
};

#endif  // PARALLELABM_SPACE_H
