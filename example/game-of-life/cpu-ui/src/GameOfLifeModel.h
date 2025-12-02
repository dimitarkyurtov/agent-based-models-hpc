#ifndef GAMEOFLIFE_GAMEOFLIFEMODEL_H
#define GAMEOFLIFE_GAMEOFLIFEMODEL_H

#include <ParallelABM/ModelCPU.h>

#include <functional>
#include <vector>

#include "Cell.h"

/**
 * @class GameOfLifeModel
 * @brief CPU implementation of Conway's Game of Life rules.
 *
 * Derives from ModelCPU<Cell> and implements the ComputeInteractions virtual
 * method to define the Game of Life logic with type-safe Cell references.
 */
class GameOfLifeModel : public ModelCPU<Cell> {
 public:
  /**
   * @brief Constructs a GameOfLifeModel with specified grid dimensions.
   * @param width Grid width for neighbor calculation
   * @param height Grid height for neighbor calculation
   */
  GameOfLifeModel(int width, int height);

  /**
   * @brief Copy constructor.
   */
  GameOfLifeModel(const GameOfLifeModel&) = default;

  /**
   * @brief Copy assignment operator.
   */
  GameOfLifeModel& operator=(const GameOfLifeModel&) = default;

  /**
   * @brief Move constructor.
   */
  GameOfLifeModel(GameOfLifeModel&&) = default;

  /**
   * @brief Move assignment operator.
   */
  GameOfLifeModel& operator=(GameOfLifeModel&&) = default;

  /**
   * @brief Virtual destructor.
   */
  ~GameOfLifeModel() override = default;

  /**
   * @brief Compute agent interactions implementing Game of Life rules.
   *
   * Implements Conway's Game of Life rules:
   * - Any live cell with 2 or 3 live neighbors survives
   * - Any dead cell with exactly 3 live neighbors becomes alive
   * - All other cells die or stay dead
   *
   * @param agents Vector of references to cells to process (mutable)
   * @param neighbors Vector of references to neighbor cells from adjacent
   * regions (read-only)
   */
  void ComputeInteractions(
      std::vector<std::reference_wrapper<Cell>>& agents,
      const std::vector<std::reference_wrapper<const Cell>>& neighbors)
      override;

 private:
  int width_;   ///< Grid width for neighbor calculation
  int height_;  ///< Grid height for neighbor calculation
};

#endif  // GAMEOFLIFE_GAMEOFLIFEMODEL_H
