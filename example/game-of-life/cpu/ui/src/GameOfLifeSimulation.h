#ifndef GAMEOFLIFE_GAMEOFLIFESIMULATION_H
#define GAMEOFLIFE_GAMEOFLIFESIMULATION_H

#include <ParallelABM/SimulationCPU.h>

#include <chrono>
#include <fstream>
#include <string>

#include "Cell.h"
#include "GameOfLifeSpace.h"
#include "Renderer.h"

/**
 * @class GameOfLifeSimulation
 * @brief CPU simulation specialized for Game of Life with terminal rendering.
 *
 * Extends SimulationCPU<Cell> to add visualization capabilities by rendering
 * the grid state after each timestep completes.
 */
class GameOfLifeSimulation : public ParallelABM::SimulationCPU<Cell> {
 public:
  /**
   * @brief Construct a Game of Life simulation with rendering.
   * @param argc Command line argument count (passed to MPI)
   * @param argv Command line arguments (passed to MPI)
   * @param space The game space containing cells
   * @param model Shared pointer to the CPU model with interaction rules
   * @param environment Compute environment configuration
   * @param renderer Renderer for visualization
   * @param frame_delay Delay between frames in milliseconds
   * @param checkpoint_file_path Path to the checkpoint file
   */
  GameOfLifeSimulation(int& argc, char**& argv,
                       std::unique_ptr<Space<Cell>> space,
                       std::shared_ptr<ModelCPU<Cell>> model,
                       ParallelABM::Environment& environment,
                       Renderer& renderer,
                       std::chrono::milliseconds frame_delay,
                       const std::string& checkpoint_file_path =
                           "checkpoints/cpu/checkpoints.dat");

  /**
   * @brief Called after each timestep to render the current state.
   * @param timestep The completed timestep number
   */
  void OnTimeStepCompleted(unsigned int timestep) override;

  /**
   * @brief Virtual destructor.
   */
  ~GameOfLifeSimulation() override = default;

 private:
  Renderer& renderer_;                     ///< Reference to the renderer
  std::chrono::milliseconds frame_delay_;  ///< Delay between frames
  std::string checkpoint_file_path_;       ///< Path to checkpoint file
  std::ofstream checkpoint_file_;  ///< Single checkpoint file for all steps
  static constexpr int kCheckpointInterval = 50;  ///< Serialize every 50 steps
};

#endif  // GAMEOFLIFE_GAMEOFLIFESIMULATION_H
