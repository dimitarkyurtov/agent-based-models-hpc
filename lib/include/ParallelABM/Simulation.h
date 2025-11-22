#ifndef PARALLELABM_SIMULATION_H
#define PARALLELABM_SIMULATION_H

#include <mpi.h>

#include <memory>
#include <utility>

#include "Agent.h"
#include "Environment.h"
#include "LocalRegion.h"
#include "Logger.h"
#include "MPICoordinator.h"
#include "MPIWorker.h"
#include "Model.h"
#include "Space.h"

// Orchestrates simulation execution across timesteps with MPI distribution.
// The workload is divided among MPI processes statically.
template <typename ModelType>
class Simulation {
 public:
  ModelType model;                         // Agent interaction model
  std::unique_ptr<MPIWorker> mpi_worker_;  // MPI worker node
  ParallelABM::Environment& environment;   // Compute resource environment

  // Initialize MPI, space, model, and calculate this process's region
  Simulation(int& argc, char**& argv, std::unique_ptr<Space> space,
             const ModelType& model, ParallelABM::Environment& environment,
             ParallelABM::SplitFunction split_function);

  // Launch model computation on the local region.
  // Processes agents using the model's interaction rule.
  // NOLINTNEXTLINE(portability-template-virtual-member-function)
  virtual void LaunchModel(ParallelABM::LocalRegion* local_region) = 0;

  // Execute simulation for given number of timesteps
  void Start(unsigned int timesteps);

  // Copy constructor - deleted (simulation should not be copied)
  Simulation(const Simulation&) = delete;

  // Copy assignment - deleted
  Simulation& operator=(const Simulation&) = delete;

  // Move constructor - deleted (simulation should not be moved)
  Simulation(Simulation&&) = delete;

  // Move assignment - deleted
  Simulation& operator=(Simulation&&) = delete;

  virtual ~Simulation() = default;
};

template <typename ModelType>
Simulation<ModelType>::Simulation(int& argc, char**& argv,
                                  std::unique_ptr<Space> space,
                                  const ModelType& model,
                                  ParallelABM::Environment& environment,
                                  ParallelABM::SplitFunction split_function)
    : model(std::move(model)), mpi_worker_(nullptr), environment(environment) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int num_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  // Initialize Logger with MPI rank and world size
  ParallelABM::Logger::GetInstance().Initialize(rank, num_processes);
  ParallelABM::Logger::GetInstance().Info(
      "Simulation: Initialized with rank " + std::to_string(rank) + " of " +
      std::to_string(num_processes) + " processes");

  if (rank == 0) {
    // Coordinator inherits from MPIWorker
    mpi_worker_ = std::make_unique<MPICoordinator>(
        rank, num_processes, std::move(space), split_function);
    ParallelABM::Logger::GetInstance().Info(
        "Simulation: Created MPICoordinator");
  } else {
    mpi_worker_ = std::make_unique<MPIWorker>(rank, split_function);
    ParallelABM::Logger::GetInstance().Info("Simulation: Created MPIWorker");
  }
}

template <typename ModelType>
void Simulation<ModelType>::Start(unsigned int timesteps) {
  auto* coordinator = dynamic_cast<MPICoordinator*>(mpi_worker_.get());

  if (coordinator != nullptr) {
    coordinator->SendLocalRegionsToWorkers();
  } else {
    mpi_worker_->ReceiveLocalRegion();
  }

  for (unsigned int step = 0; step < timesteps; ++step) {
    if (coordinator != nullptr) {
      coordinator->SendNeighborsToWorkers();
    } else {
      mpi_worker_->ReceiveNeighbors();
    }

    ParallelABM::LocalRegion* local_region = mpi_worker_->GetLocalRegion();
    LaunchModel(local_region);

    if (coordinator != nullptr) {
      coordinator->ReceiveLocalRegionsFromWorkers();
    } else {
      mpi_worker_->SendLocalRegionToLeader();
    }
  }
}

#endif  // PARALLELABM_SIMULATION_H
