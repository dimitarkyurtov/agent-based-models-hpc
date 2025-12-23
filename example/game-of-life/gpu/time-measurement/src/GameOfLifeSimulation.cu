#include <mpi.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <utility>

#include "GameOfLifeSimulation.h"
#include "GameOfLifeSpace.h"

GameOfLifeSimulation::GameOfLifeSimulation(
    int& argc, char**& argv, std::unique_ptr<Space<Cell>> space,
    std::shared_ptr<ModelCUDA<Cell>> model,
    ParallelABM::Environment& environment, bool sync_regions_every_timestep)
    : ParallelABM::SimulationCUDA<Cell>(argc, argv, std::move(space), model,
                                        environment,
                                        sync_regions_every_timestep),
      computation_time_ms_(0.0),
      checkpoint_count_(0) {}

void GameOfLifeSimulation::Setup() {
  // Call parent class Setup to initialize GPU contexts
  ParallelABM::SimulationCUDA<Cell>::Setup();
  setup_complete_time_ = std::chrono::high_resolution_clock::now();
}

void GameOfLifeSimulation::LaunchModel(
    ParallelABM::LocalRegion<Cell>* local_region) {
  auto start = std::chrono::high_resolution_clock::now();

  // Call parent GPU computation
  ParallelABM::SimulationCUDA<Cell>::LaunchModel(local_region);

  auto end = std::chrono::high_resolution_clock::now();
  computation_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
}

void GameOfLifeSimulation::OnTimeStepCompleted(unsigned int timestep) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Sync regions at checkpoint intervals (must be called by ALL ranks)
  if (timestep % kCheckpointInterval == 0) {
    SyncRegions();

    // Only coordinator serializes the data
    if (rank == 0) {
      auto* game_space = dynamic_cast<GameOfLifeSpace*>(space_.get());
      if (game_space != nullptr) {
        game_space->Serialize(std::cout, static_cast<int>(timestep));
        std::cout.flush();
        checkpoint_count_++;
      }
    }
  }
}

GameOfLifeSimulation::~GameOfLifeSimulation() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "\n=== Performance Breakdown (Rank " << rank << ") ===\n"
              << "GPU Computation Time: " << computation_time_ms_ << " ms\n"
              << "Checkpoints Written: " << checkpoint_count_ << "\n";
  } else {
    std::cout << "\n=== Performance Breakdown (Rank " << rank << ") ===\n"
              << "GPU Computation Time: " << computation_time_ms_ << " ms\n";
  }
}
