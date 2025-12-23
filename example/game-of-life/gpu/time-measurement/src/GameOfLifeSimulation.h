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
   * @param sync_regions_every_timestep Whether to sync regions every timestep
   */
  GameOfLifeSimulation(int& argc, char**& argv,
                       std::unique_ptr<Space<Cell>> space,
                       std::shared_ptr<ModelCUDA<Cell>> model,
                       ParallelABM::Environment& environment,
                       bool sync_regions_every_timestep = true);

  /**
   * @brief Setup GPU contexts before simulation starts.
   */
  void Setup() override;

  /**
   * @brief Launch GPU computation with timing
   */
  void LaunchModel(ParallelABM::LocalRegion<Cell>* local_region) override;

  /**
   * @brief Called after each timestep to print checkpoint hash every 50 steps.
   * @param timestep The completed timestep number
   */
  void OnTimeStepCompleted(unsigned int timestep) override;

  /**
   * @brief Virtual destructor - prints timing breakdown
   */
  ~GameOfLifeSimulation() override;

 private:
  static constexpr int kCheckpointInterval = 50;  ///< Serialize every 50 steps

  double computation_time_ms_;  ///< Total GPU computation time
  int checkpoint_count_;        ///< Number of checkpoints written
  std::chrono::high_resolution_clock::time_point setup_complete_time_;
};

#endif  // GAMEOFLIFE_GPU_GAMEOFLIFESIMULATION_H
