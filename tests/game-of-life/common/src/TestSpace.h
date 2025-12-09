#ifndef TEST_TESTSPACE_H
#define TEST_TESTSPACE_H

#include <ParallelABM/Space.h>

#include <cstdint>
#include <iosfwd>
#include <vector>

#include "TestAgent.h"

/**
 * @class TestSpace
 * @brief 2D grid space implementation for test simulations.
 *
 * Implements the Space<TestAgent> interface to provide a rectangular grid where
 * each agent is positioned at integer coordinates. Agents are stored in
 * row-major order for efficient spatial partitioning and zero-copy MPI.
 */
class TestSpace : public Space<TestAgent> {
 public:
  /**
   * @brief Construct a test grid with specified dimensions.
   * @param width Number of columns in the grid
   * @param height Number of rows in the grid
   */
  TestSpace(int width, int height);

  /**
   * @brief Copy constructor.
   */
  TestSpace(const TestSpace&) = default;

  /**
   * @brief Copy assignment operator.
   */
  TestSpace& operator=(const TestSpace&) = default;

  /**
   * @brief Move constructor.
   */
  TestSpace(TestSpace&&) = default;

  /**
   * @brief Move assignment operator.
   */
  TestSpace& operator=(TestSpace&&) = default;

  /**
   * @brief Virtual destructor.
   */
  ~TestSpace() override = default;

  /**
   * @brief Populate the grid with agents using predefined patterns.
   *
   * Creates width * height agents arranged in a 2D grid with positions
   * assigned in row-major order. Each agent's initial alive state is
   * determined using predefined patterns (gliders, oscillators, etc.)
   * for reproducible testing.
   */
  void Initialize() override;

  /**
   * @brief Split the grid into horizontal bands for MPI distribution.
   *
   * Divides the grid into approximately equal horizontal regions for
   * parallel processing. Each region contains complete rows to minimize
   * inter-region communication.
   *
   * @param num_regions Number of regions to create
   * @return Vector of Region objects containing agent indices
   */
  [[nodiscard]] std::vector<Region> SplitIntoRegions(
      int num_regions) const override;

  /**
   * @brief Split local region into subregions for parallel processing.
   *
   * Distributes agents evenly among the requested number of subregions
   * for fine-grained parallelism (CPU threads, GPU devices).
   *
   * @param region The local region to subdivide
   * @param num_subregions Number of subregions to create
   * @return Vector of subregions with evenly distributed agents
   */
  [[nodiscard]] std::vector<ParallelABM::LocalSubRegion<TestAgent>>
  SplitLocalRegion(ParallelABM::LocalRegion<TestAgent>& region,
                   int num_subregions) const override;

  /**
   * @brief Retrieves current neighbor agents for the specified region.
   *
   * Calculates and returns the boundary agents that neighbor the given region,
   * based on the current state of the grid. For a toroidal grid, this includes
   * the row above and the row below the region's boundary, with wrap-around.
   *
   * @param region Reference to the region whose neighbors are needed
   * @return Vector of neighbor agents with current state
   */
  [[nodiscard]] std::vector<TestAgent> GetRegionNeighbours(
      const Region& region) const override;

  /**
   * @brief Get the grid width.
   * @return Number of columns in the grid
   */
  [[nodiscard]] int GetWidth() const noexcept { return width_; }

  /**
   * @brief Get the grid height.
   * @return Number of rows in the grid
   */
  [[nodiscard]] int GetHeight() const noexcept { return height_; }

  /**
   * @brief Get the agent at specified grid coordinates.
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Reference to the agent at (x, y)
   */
  [[nodiscard]] TestAgent& GetAgentAt(int x, int y);

  /**
   * @brief Get the agent at specified grid coordinates (const version).
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Const reference to the agent at (x, y)
   */
  [[nodiscard]] const TestAgent& GetAgentAt(int x, int y) const;

  /**
   * @brief Serialize the current grid state to an output stream.
   *
   * Outputs a text-based checkpoint containing the simulation step number
   * and SHA-256 hash of the binary grid state (0/1 for each agent).
   *
   * @param os Output stream to write the serialized data
   * @param step Current simulation step number
   */
  void Serialize(std::ostream& os, int step) const;

 private:
  int width_;   ///< Grid width (number of columns)
  int height_;  ///< Grid height (number of rows)

  /**
   * @brief Convert 2D coordinates to linear index in agents vector.
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Linear index in row-major order
   */
  [[nodiscard]] int CoordToIndex(int x, int y) const noexcept;
};

#endif  // TEST_TESTSPACE_H
