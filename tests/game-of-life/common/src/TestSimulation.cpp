#include "TestSimulation.h"

#include <mpi.h>

#include <filesystem>

TestSimulation::TestSimulation(int& argc, char**& argv,
                               std::unique_ptr<Space<TestAgent>> space,
                               std::shared_ptr<ModelCPU<TestAgent>> model,
                               ParallelABM::Environment& environment,
                               const std::string& checkpoint_file,
                               int checkpoint_interval)
    : SimulationCPU<TestAgent>(argc, argv, std::move(space), model,
                               environment),
      checkpoint_file_(checkpoint_file),
      checkpoint_interval_(checkpoint_interval) {
  // Create checkpoint directory and open file on rank 0
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::filesystem::path file_path(checkpoint_file_);
    std::filesystem::create_directories(file_path.parent_path());
    checkpoint_stream_.open(checkpoint_file_);
  }
}

TestSimulation::~TestSimulation() {
  // Close checkpoint file if open
  if (checkpoint_stream_.is_open()) {
    checkpoint_stream_.close();
  }
}

void TestSimulation::OnTimeStepCompleted(unsigned int timestep) {
  // Only serialize on the coordinator process (rank 0)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0 && checkpoint_stream_.is_open()) {
    // Save checkpoint at specified intervals
    if (timestep % checkpoint_interval_ == 0) {
      auto* test_space = dynamic_cast<TestSpace*>(space_.get());
      if (test_space != nullptr) {
        test_space->Serialize(checkpoint_stream_, static_cast<int>(timestep));
        checkpoint_stream_.flush();
      }
    }
  }
}
