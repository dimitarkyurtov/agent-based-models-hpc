#include "GameOfLifeSimulation.h"

#include <mpi.h>

#include <thread>

GameOfLifeSimulation::GameOfLifeSimulation(
    int& argc, char**& argv, std::unique_ptr<Space<Cell>> space,
    std::shared_ptr<ModelCPU<Cell>> model,
    ParallelABM::Environment& environment, Renderer& renderer,
    std::chrono::milliseconds frame_delay)
    : SimulationCPU<Cell>(argc, argv, std::move(space), model, environment),
      renderer_(renderer),
      frame_delay_(frame_delay) {}

void GameOfLifeSimulation::OnTimeStepCompleted(
    [[maybe_unused]] unsigned int timestep) {
  // Only render on the coordinator process (rank 0)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // Get the local region which has the complete agent state on coordinator
    const ParallelABM::LocalRegion<Cell>* local_region =
        mpi_worker_->GetLocalRegion();
    if (local_region != nullptr) {
      // renderer_.Render(local_region->GetAgents(), timestep);
    }

    // Add delay for visualization
    std::this_thread::sleep_for(frame_delay_);
  }
}
