#include "GameOfLifeSimulation.h"

#include <mpi.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

GameOfLifeSimulation::GameOfLifeSimulation(
    int& argc, char**& argv, std::unique_ptr<Space<Cell>> space,
    std::shared_ptr<ModelCPU<Cell>> model,
    ParallelABM::Environment& environment, Renderer& renderer,
    std::chrono::milliseconds frame_delay, const std::string& checkpoint_dir)
    : SimulationCPU<Cell>(argc, argv, std::move(space), model, environment),
      renderer_(renderer),
      frame_delay_(frame_delay),
      checkpoint_dir_(checkpoint_dir) {
  // Create checkpoint directory if it doesn't exist
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::filesystem::create_directories(checkpoint_dir_);
  }
}

void GameOfLifeSimulation::OnTimeStepCompleted(unsigned int timestep) {
  // Only render and serialize on the coordinator process (rank 0)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto* game_space = dynamic_cast<GameOfLifeSpace*>(space_.get());
    if (game_space != nullptr) {
      renderer_.Render(*game_space, timestep);

      // Serialize every kCheckpointInterval steps
      if (timestep % kCheckpointInterval == 0) {
        std::ostringstream filename;
        filename << checkpoint_dir_ << "/state_" << std::setw(6)
                 << std::setfill('0') << timestep << ".dat";

        std::ofstream checkpoint_file(filename.str());
        if (checkpoint_file.is_open()) {
          game_space->Serialize(checkpoint_file, static_cast<int>(timestep));
          checkpoint_file.close();
        }
      }
    }

    // Add delay for visualization
    std::this_thread::sleep_for(frame_delay_);
  }
}
