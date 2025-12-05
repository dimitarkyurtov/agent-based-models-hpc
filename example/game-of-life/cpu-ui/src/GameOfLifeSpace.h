#ifndef GAMEOFLIFE_GAMEOFLIFESPACE_H
#define GAMEOFLIFE_GAMEOFLIFESPACE_H

#include <ParallelABM/Space.h>

#include <random>
#include <vector>

#include "Cell.h"

/**
 * @class GameOfLifeSpace
 * @brief 2D grid space implementation for Conway's Game of Life.
 *
 * Implements the Space<Cell> interface to provide a rectangular grid where
 * each cell is positioned at integer coordinates. Cells are stored in
 * row-major order for efficient spatial partitioning and zero-copy MPI.
 */
class GameOfLifeSpace : public Space<Cell> {
 public:
  /**
   * @brief Construct a Game of Life grid with specified dimensions.
   * @param width Number of columns in the grid
   * @param height Number of rows in the grid
   * @param density Initial density of alive cells (0.0 to 1.0)
   */
  GameOfLifeSpace(int width, int height, double density = 0.3);

  /**
   * @brief Copy constructor.
   */
  GameOfLifeSpace(const GameOfLifeSpace&) = default;

  /**
   * @brief Copy assignment operator.
   */
  GameOfLifeSpace& operator=(const GameOfLifeSpace&) = default;

  /**
   * @brief Move constructor.
   */
  GameOfLifeSpace(GameOfLifeSpace&&) = default;

  /**
   * @brief Move assignment operator.
   */
  GameOfLifeSpace& operator=(GameOfLifeSpace&&) = default;

  /**
   * @brief Virtual destructor.
   */
  ~GameOfLifeSpace() override = default;

  /**
   * @brief Populate the grid with cells having random initial states.
   *
   * Creates width * height cells arranged in a 2D grid with positions
   * assigned in row-major order. Each cell's initial alive state is
   * determined randomly based on the configured density.
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
  [[nodiscard]] std::vector<ParallelABM::LocalSubRegion<Cell>> SplitLocalRegion(
      ParallelABM::LocalRegion<Cell>& region,
      int num_subregions) const override;

  /**
   * @brief Retrieves current neighbor agents for the specified region.
   *
   * Calculates and returns the boundary cells that neighbor the given region,
   * based on the current state of the grid. For a toroidal grid, this includes
   * the row above and the row below the region's boundary, with wrap-around.
   *
   * @param region Reference to the region whose neighbors are needed
   * @return Vector of neighbor cells with current state
   */
  [[nodiscard]] std::vector<Cell> GetRegionNeighbours(
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
   * @brief Get the cell at specified grid coordinates.
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Reference to the cell at (x, y)
   */
  [[nodiscard]] Cell& GetCellAt(int x, int y);

  /**
   * @brief Get the cell at specified grid coordinates (const version).
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Const reference to the cell at (x, y)
   */
  [[nodiscard]] const Cell& GetCellAt(int x, int y) const;

 private:
  int width_;         ///< Grid width (number of columns)
  int height_;        ///< Grid height (number of rows)
  double density_;    ///< Initial alive cell density
  std::mt19937 rng_;  ///< Random number generator for initialization

  /**
   * @brief Convert 2D coordinates to linear index in agents vector.
   * @param x X coordinate (column)
   * @param y Y coordinate (row)
   * @return Linear index in row-major order
   */
  [[nodiscard]] int CoordToIndex(int x, int y) const noexcept;
};

#endif  // GAMEOFLIFE_GAMEOFLIFESPACE_H
