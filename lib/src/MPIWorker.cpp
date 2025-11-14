#include "ParallelABM/MPIWorker.h"

#include <mpi.h>

#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/MPINode.h"

MPIWorker::MPIWorker(int rank) : MPINode(rank) {}

MPIWorker::MPIWorker(int rank, const std::vector<Agent>& region)
    : MPINode(rank), localRegion(region) {}

std::vector<Agent> MPIWorker::GetLocalRegion() { return localRegion; }

std::vector<Agent> MPIWorker::GetNeighbors() { return neighbors; }

void MPIWorker::SetLocalRegion(const std::vector<Agent>& region) {
  localRegion = region;
}

void MPIWorker::ReceiveLocalRegion() {
  int region_size = 0;
  MPI_Recv(&region_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  localRegion.resize(region_size);

  MPI_Recv(localRegion.data(), static_cast<int>(region_size * sizeof(Agent)),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void MPIWorker::SendLocalRegionToLeader() {
  const int kRegionSize = static_cast<int>(localRegion.size());
  MPI_Send(&kRegionSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

  MPI_Send(localRegion.data(), static_cast<int>(kRegionSize * sizeof(Agent)),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD);
}

void MPIWorker::ReceiveNeighbors() {
  int num_neighbors = 0;
  MPI_Recv(&num_neighbors, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  neighbors.resize(num_neighbors);

  MPI_Recv(neighbors.data(), static_cast<int>(num_neighbors * sizeof(Agent)),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
