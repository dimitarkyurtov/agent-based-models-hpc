#ifndef PARALLELABM_SIMULATION_H
#define PARALLELABM_SIMULATION_H

#include <mpi.h>

#include <memory>
#include <utility>

#include "Agent.h"
#include "MPICoordinator.h"
#include "MPINode.h"
#include "MPIWorker.h"
#include "Model.h"
#include "Space.h"

// Orchestrates simulation execution across timesteps with MPI distribution.
// The workload is divided among MPI processes statically.
template <typename ModelType>
class Simulation {
 public:
  ModelType model;                   // Agent interaction model
  std::unique_ptr<MPINode> mpiNode;  // MPI node (coordinator or worker)

  // Initialize MPI, space, model, and calculate this process's region
  Simulation(int& argc, char**& argv, std::unique_ptr<Space> space,
             const ModelType& model);

  // Copy computation results from device to host memory
  virtual void CopyBackToHost() = 0;

  // Launch model computation on agent subset with neighbors
  virtual void LaunchModel(Agent* agents, int size, Agent* neighbors,
                           int neighborSize) = 0;

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
                                  const ModelType& model)
    : model(std::move(model)), mpiNode(nullptr) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int num_processes = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  if (rank == 0) {
    // Move ownership of space to MPICoordinator
    mpiNode =
        std::make_unique<MPICoordinator>(rank, num_processes, std::move(space));
  } else {
    mpiNode = std::make_unique<MPIWorker>(rank);
  }
}

template <typename ModelType>
void Simulation<ModelType>::Start(unsigned int timesteps) {
  auto* coordinator = dynamic_cast<MPICoordinator*>(mpiNode.get());
  auto* worker = dynamic_cast<MPIWorker*>(mpiNode.get());

  if (coordinator != nullptr) {
    coordinator->SendLocalRegionsToWorkers();
  } else if (worker != nullptr) {
    worker->ReceiveLocalRegion();
  }

  for (unsigned int step = 0; step < timesteps; ++step) {
    if (coordinator != nullptr) {
      coordinator->SendNeighborsToWorkers();
    } else if (worker != nullptr) {
      worker->ReceiveNeighbors();
    }

    std::vector<Agent> local_region = worker->GetLocalRegion();
    std::vector<Agent> neighbors = worker->GetNeighbors();
    LaunchModel(local_region.data(), static_cast<int>(local_region.size()),
                neighbors.data(), static_cast<int>(neighbors.size()));
    CopyBackToHost();

    if (coordinator != nullptr) {
      coordinator->ReceiveLocalRegionsFromWorkers();
    } else if (worker != nullptr) {
      worker->SendLocalRegionToLeader();
    }
  }
}

#endif  // PARALLELABM_SIMULATION_H
