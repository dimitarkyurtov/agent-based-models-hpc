#ifndef PARALLELABM_SIMULATION_H
#define PARALLELABM_SIMULATION_H

#include <mpi.h>

#include <memory>
#include <utility>

#include "Environment.h"
#include "LocalRegion.h"
#include "Logger.h"
#include "MPICoordinator.h"
#include "MPIWorker.h"
#include "Model.h"
#include "Space.h"

// Orchestrates simulation execution across timesteps with MPI distribution.
// The workload is divided among MPI processes statically.
template <typename AgentT, typename ModelType>
class Simulation {
 public:
  std::shared_ptr<ModelType> model;  // Agent interaction model
  std::unique_ptr<MPIWorker<AgentT>> mpi_worker_ = nullptr;  // MPI worker node
  ParallelABM::Environment& environment;  // Compute resource environment

  // Initialize MPI, space, model, and calculate this process's region
  Simulation(int& argc, char**& argv, std::unique_ptr<Space<AgentT>> space,
             std::shared_ptr<ModelType> model,
             ParallelABM::Environment& environment,
             bool sync_regions_every_timestep = true);

  // Launch model computation on the local region.
  // Processes agents using the model's interaction rule.
  // NOLINTNEXTLINE(portability-template-virtual-member-function)
  // NOLINTNEXTLINE(clang-diagnostic-unused-parameter)
  virtual void LaunchModel(ParallelABM::LocalRegion<AgentT>* local_region) = 0;

  // Called at the end of each timestep. Override to perform custom actions
  // such as rendering, logging, or data collection.
  // NOLINTNEXTLINE(portability-template-virtual-member-function)
  virtual void OnTimeStepCompleted(unsigned int /*timestep*/) {}

  // Perform one-time setup before simulation starts.
  // Override to initialize data structures that persist across timesteps.
  // NOLINTNEXTLINE(portability-template-virtual-member-function)
  virtual void Setup() {}

  // Execute simulation for given number of timesteps
  void Start(unsigned int timesteps);

  // Explicitly synchronize regions between workers and coordinator
  // Call this when you need the coordinator to have updated region data
  void SyncRegions();

  // Copy constructor - deleted (simulation should not be copied)
  Simulation(const Simulation&) = delete;

  // Copy assignment - deleted
  Simulation& operator=(const Simulation&) = delete;

  // Move constructor - deleted (simulation should not be moved)
  Simulation(Simulation&&) = delete;

  // Move assignment - deleted
  Simulation& operator=(Simulation&&) = delete;

  virtual ~Simulation() { MPI_Finalize(); };

 protected:
  std::shared_ptr<Space<AgentT>>
      space_;                         // Shared ownership of simulation space
  bool sync_regions_every_timestep_;  // Whether to sync regions every timestep
};

template <typename AgentT, typename ModelType>
Simulation<AgentT, ModelType>::Simulation(int& argc, char**& argv,
                                          std::unique_ptr<Space<AgentT>> space,
                                          std::shared_ptr<ModelType> model,
                                          ParallelABM::Environment& environment,
                                          bool sync_regions_every_timestep)
    : model(model),
      mpi_worker_(nullptr),
      environment(environment),
      space_(std::move(space)),
      sync_regions_every_timestep_(sync_regions_every_timestep) {
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
    mpi_worker_ =
        std::make_unique<MPICoordinator<AgentT>>(rank, num_processes, space_);
    ParallelABM::Logger::GetInstance().Info(
        "Simulation: Created MPICoordinator");
  } else {
    mpi_worker_ = std::make_unique<MPIWorker<AgentT>>(rank);
    ParallelABM::Logger::GetInstance().Info("Simulation: Created MPIWorker");
  }
}

template <typename AgentT, typename ModelType>
void Simulation<AgentT, ModelType>::SyncRegions() {
  auto* coordinator = dynamic_cast<MPICoordinator<AgentT>*>(mpi_worker_.get());

  if (coordinator != nullptr) {
    coordinator->ReceiveLocalRegionsFromWorkers();
    coordinator->SendNeighborsToWorkers();
  } else {
    mpi_worker_->SendLocalRegionToLeader();
    mpi_worker_->ReceiveNeighbors();
  }
}

template <typename AgentT, typename ModelType>
void Simulation<AgentT, ModelType>::Start(unsigned int timesteps) {
  auto* coordinator = dynamic_cast<MPICoordinator<AgentT>*>(mpi_worker_.get());

  if (coordinator != nullptr) {
    coordinator->SendLocalRegionsToWorkers();
    coordinator->SendNeighborsToWorkers();
  } else {
    mpi_worker_->ReceiveLocalRegion();
    mpi_worker_->ReceiveNeighbors();
  }

  Setup();

  for (unsigned int step = 0; step < timesteps; ++step) {
    ParallelABM::LocalRegion<AgentT>*
        local_region =  // NOLINT(cppcoreguidelines-init-variables)
        mpi_worker_->GetLocalRegion();
    LaunchModel(local_region);

    // Only sync regions if configured to do so
    if (sync_regions_every_timestep_) {
      SyncRegions();
    }

    OnTimeStepCompleted(step);
  }
}

#endif  // PARALLELABM_SIMULATION_H
