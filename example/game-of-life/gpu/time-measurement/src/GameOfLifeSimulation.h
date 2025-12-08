#ifndef GAMEOFLIFE_GPU_GAMEOFLIFESIMULATION_H
#define GAMEOFLIFE_GPU_GAMEOFLIFESIMULATION_H

#include <ParallelABM/SimulationCUDA.h>

#include "Cell.h"

/**
 * @class GameOfLifeSimulation
 * @brief GPU simulation specialized for Game of Life time measurement.
 *
 * Extends SimulationCUDA<Cell> for performance measurement without
 * visualization using CUDA-accelerated computation.
 */
class GameOfLifeSimulation : public ParallelABM::SimulationCUDA<Cell> {
 public:
  /**
   * @brief Construct a Game of Life GPU simulation for time measurement.
   * @param argc Command line argument count (passed to MPI)
   * @param argv Command line arguments (passed to MPI)
   * @param space The game space containing cells
   * @param model Shared pointer to the CUDA model with interaction rules
   * @param environment Compute environment configuration
   */
  GameOfLifeSimulation(int& argc, char**& argv,
                       std::unique_ptr<Space<Cell>> space,
                       std::shared_ptr<ModelCUDA<Cell>> model,
                       ParallelABM::Environment& environment);

  /**
   * @brief Virtual destructor.
   */
  ~GameOfLifeSimulation() override = default;
};

#endif  // GAMEOFLIFE_GPU_GAMEOFLIFESIMULATION_H
