#ifndef GAMEOFLIFE_GPU_GAMEOFLIFEMODEL_H
#define GAMEOFLIFE_GPU_GAMEOFLIFEMODEL_H

#include <ParallelABM/ModelCUDA.h>

#include "Cell.h"

/**
 * @class GameOfLifeModel
 * @brief CUDA-based Game of Life model implementing Conway's rules on GPU.
 *
 * Implements ModelCUDA<Cell> to provide GPU-accelerated Game of Life
 * computation using CUDA kernels. Each cell's next state is computed
 * in parallel based on its 8 neighbors following Conway's rules:
 * - Alive cell with 2-3 neighbors survives
 * - Dead cell with exactly 3 neighbors becomes alive
 * - All other cells die or remain dead
 */
class GameOfLifeModel : public ModelCUDA<Cell> {
 public:
  /**
   * @brief Construct Game of Life model with grid dimensions.
   * @param width Grid width (number of columns)
   * @param height Grid height (number of rows)
   */
  GameOfLifeModel(int width, int height);

  /**
   * @brief Get the CUDA kernel implementing Game of Life rules.
   * @return Function pointer to the CUDA kernel
   */
  [[nodiscard]] InteractionRuleCUDA GetInteractionKernel() const override;

 private:
  int width_;   ///< Grid width
  int height_;  ///< Grid height
};

#endif  // GAMEOFLIFE_GPU_GAMEOFLIFEMODEL_H
