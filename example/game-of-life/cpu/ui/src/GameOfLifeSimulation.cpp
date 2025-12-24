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
    std::chrono::milliseconds frame_delay,
    const std::string& checkpoint_file_path)
    : SimulationCPU<Cell>(argc, argv, std::move(space), model, environment),
      renderer_(renderer),
      frame_delay_(frame_delay),
      checkpoint_file_path_(checkpoint_file_path) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::filesystem::path file_path(checkpoint_file_path_);
    if (file_path.has_parent_path()) {
      std::filesystem::create_directories(file_path.parent_path());
    }
    checkpoint_file_.open(checkpoint_file_path_);
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

      if (timestep % kCheckpointInterval == 0 && checkpoint_file_.is_open()) {
        game_space->Serialize(checkpoint_file_, static_cast<int>(timestep));
        checkpoint_file_.flush();
      }
    }

    // Add delay for visualization
    std::this_thread::sleep_for(frame_delay_);
  }
}
