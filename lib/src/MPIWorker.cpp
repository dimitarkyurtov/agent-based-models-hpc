#include "ParallelABM/MPIWorker.h"

#include <mpi.h>

#include <string>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/Logger.h"
#include "ParallelABM/MPINode.h"

MPIWorker::MPIWorker(int rank) : MPINode(rank) {}

MPIWorker::MPIWorker(int rank, const std::vector<Agent>& region)
    : MPINode(rank), localRegion(region) {}

std::vector<Agent>& MPIWorker::GetLocalRegion() { return localRegion; }

std::vector<Agent>& MPIWorker::GetNeighbors() { return neighbors; }

void MPIWorker::SetLocalRegion(const std::vector<Agent>& region) {
  localRegion = region;
}

void MPIWorker::ReceiveLocalRegion() {
  int region_size = 0;
  MPI_Recv(&region_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Received region size (" +
                                          std::to_string(region_size) +
                                          ") from coordinator");

  localRegion.resize(region_size);

  MPI_Recv(localRegion.data(), static_cast<int>(region_size * sizeof(Agent)),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Received local region (" +
                                          std::to_string(region_size) +
                                          " agents) from coordinator");
}

void MPIWorker::SendLocalRegionToLeader() {
  const int kRegionSize = static_cast<int>(localRegion.size());
  MPI_Send(&kRegionSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent region size (" +
                                          std::to_string(kRegionSize) +
                                          ") to coordinator");

  MPI_Send(localRegion.data(), static_cast<int>(kRegionSize * sizeof(Agent)),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  ParallelABM::Logger::GetInstance().Info("MPIWorker: Sent local region (" +
                                          std::to_string(kRegionSize) +
                                          " agents) to coordinator");
}

void MPIWorker::ReceiveNeighbors() {
  int num_neighbors = 0;
  MPI_Recv(&num_neighbors, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ParallelABM::Logger::GetInstance().Info(
      "MPIWorker: Received neighbor count (" + std::to_string(num_neighbors) +
      ") from coordinator");

  neighbors.resize(num_neighbors);

  if (num_neighbors > 0) {
    MPI_Recv(neighbors.data(), static_cast<int>(num_neighbors * sizeof(Agent)),
             MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ParallelABM::Logger::GetInstance().Info("MPIWorker: Received neighbors (" +
                                            std::to_string(num_neighbors) +
                                            " agents) from coordinator");
  }
}
