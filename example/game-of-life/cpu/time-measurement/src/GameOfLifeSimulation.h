#ifndef GAMEOFLIFE_GAMEOFLIFESIMULATION_H
#define GAMEOFLIFE_GAMEOFLIFESIMULATION_H

#include <ParallelABM/SimulationCPU.h>

#include "Cell.h"

/**
 * @class GameOfLifeSimulation
 * @brief CPU simulation specialized for Game of Life time measurement.
 *
 * Extends SimulationCPU<Cell> for performance measurement without
 * visualization.
 */
class GameOfLifeSimulation : public ParallelABM::SimulationCPU<Cell> {
 public:
  /**
   * @brief Construct a Game of Life simulation for time measurement.
   * @param argc Command line argument count (passed to MPI)
   * @param argv Command line arguments (passed to MPI)
   * @param space The game space containing cells
   * @param model Shared pointer to the CPU model with interaction rules
   * @param environment Compute environment configuration
   * @param sync_regions_every_timestep Whether to sync regions every timestep
   */
  GameOfLifeSimulation(int& argc, char**& argv,
                       std::unique_ptr<Space<Cell>> space,
                       std::shared_ptr<ModelCPU<Cell>> model,
                       ParallelABM::Environment& environment,
                       bool sync_regions_every_timestep = true);

  /**
   * @brief Virtual destructor.
   */
  ~GameOfLifeSimulation() override = default;
};

#endif  // GAMEOFLIFE_GAMEOFLIFESIMULATION_H
