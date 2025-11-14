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
   * @brief Finds the nearest appropriate index to split the space for workload
   * distribution.
   *
   * When distributing the simulation across multiple computational nodes, the
   * library needs to partition the space into regions. However, an arbitrary
   * split might not be optimal for the spatial structure. This method allows
   * the Space implementation to suggest the nearest "natural" split point that
   * minimizes the number of cross-region neighbor connections.
   *
   * @param index The desired split index in the agents vector where the library
   * wants to partition the space. This index is treated as a position in the
   * agents vector where agents[0..index-1] would go to one partition and
   * agents[index..end] would go to another.
   *
   * @param forward If true, search for the nearest proper split forward
   * (towards higher indices). If false, search backward (towards lower
   * indices). Default is false.
   *
   * @return The nearest index where the space can be appropriately split. This
   * should be an index that results in minimal cross-region neighbor
   * relationships while remaining as close as possible to the requested index.
   *
   * @details
   * The implementation should choose a split point that creates "clean"
   * boundaries in the spatial structure. What constitutes a clean boundary
   * depends on the space topology:
   *
   * Example 1 - 2D Grid (10x10, stored row-major):
   * - Request: split at index 15 (middle of row 1)
   * - Better options: index 10 (end of row 0) or index 20 (end of row 1)
   * - This ensures the split occurs at row boundaries, minimizing neighbors
   * across regions
   *
   * Example 2 - 3D Grid:
   * - Prefer splits at layer or plane boundaries to minimize cross-region
   * communication
   *
   * Example 3 - Already optimal split:
   * - If index already represents a natural boundary, simply return the same
   * index
   *
   * The implementation should balance:
   * 1. Proximity to the requested index (important for load balancing)
   * 2. Minimizing cross-region neighbor connections (important for
   * communication efficiency)
   *
   * @note This method is called during the initial workload distribution phase
   * and potentially during dynamic load rebalancing.
   */
  [[nodiscard]] virtual int GetNearestProperSplit(unsigned index,
                                                  bool forward) const = 0;

  /**
   * @brief Retrieves the neighbor agents for a specific region that lie outside
   * that region.
   *
   * When a computational node processes a region of the space, it needs to know
   * about agents that are neighbors to agents within the region but are not
   * themselves part of the region. These "boundary neighbors" or "ghost agents"
   * are necessary for correctly computing agent interactions and state updates.
   *
   * @param startIndex The starting index (inclusive) of the region in the
   * agents vector. This marks the beginning of the region being queried.
   *
   * @param endIndex The ending index (exclusive) of the region in the agents
   * vector. The region includes agents[startIndex] through agents[endIndex-1].
   *                 Following STL container conventions, endIndex is one past
   * the last element.
   *
   * @return A vector of Agent objects (copies) that are neighbors of the
   * specified region but not part of it. These represent the "halo" or "ghost
   * zone" around the region. The returned agents satisfy two conditions:
   *         1. They are neighbors of at least one agent in [startIndex,
   * endIndex)
   *         2. They are NOT in the range [startIndex, endIndex)
   *
   * @details
   * The exact definition of "neighbor" depends on the spatial structure and
   * interaction model:
   * - 2D Grid with 4-connectivity: agents in cardinal directions (N, S, E, W)
   * - 2D Grid with 8-connectivity: includes diagonal neighbors
   * - 3D Grid: up to 26 neighbors (6-face, 18-edge, or 26-vertex connectivity)
   * - Continuous space: agents within a certain interaction radius
   * - Network/Graph: nodes connected by edges
   *
   * Example - 2D Grid (10x10, row-major order):
   * - Region: agents[10..20) represents row 1 (indices 10-19)
   * - Returned neighbors: agents from row 0 (indices 0-9) and row 2 (indices
   * 20-29) that are adjacent to row 1
   * - For 4-connectivity: all agents in rows 0 and 2
   * - For 8-connectivity: would also include diagonal neighbors from adjacent
   * rows
   *
   * This method is critical for:
   * 1. Computing agent interactions at region boundaries
   * 2. Determining which agent data must be communicated between MPI nodes
   * 3. Ensuring consistency in distributed simulation (ghost zone
   * synchronization)
   * 4. Maintaining correctness of agent state updates across partition
   * boundaries
   *
   * @note The library will use this information to determine communication
   * patterns and setup MPI message passing for ghost zone synchronization.
   */
  [[nodiscard]] virtual std::vector<Agent> GetNeighborsForRegion(
      unsigned startIndex, unsigned endIndex) const = 0;

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
