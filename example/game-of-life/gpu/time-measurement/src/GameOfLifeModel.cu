#include <cuda_runtime.h>

#include "GameOfLifeModel.h"

// Global constants for grid dimensions (accessible from device)
__constant__ int d_grid_width;
__constant__ int d_grid_height;

/**
 * @brief CUDA kernel implementing Game of Life rules with O(1) neighbor lookup.
 *
 * Each thread processes one cell, counting alive neighbors using row-major
 * indexing for constant-time access. The agents array is organized in
 * row-major order (y * width + x), allowing direct calculation of neighbor
 * positions. Boundary cells are stored in the neighbors array (upper and
 * lower boundary rows).
 *
 * @param agents Device array of local cells in row-major order
 * @param num_agents Number of local cells
 * @param neighbors Device array of boundary cells (upper row + lower row)
 * @param num_neighbors Number of boundary cells (typically 2 * width)
 */
__global__ void GameOfLifeKernel(Cell* agents, int num_agents, Cell* neighbors,
                                 int num_neighbors) {
  const int kIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (kIdx >= num_agents) {
    return;
  }

  Cell& cell = agents[kIdx];
  int alive_neighbors = 0;

  const int kLocalRow = kIdx / d_grid_width;
  const int kCol = kIdx % d_grid_width;

  const int kNumRows = num_agents / d_grid_width;

  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;  // Skip self
      }

      const int kNeighborCol = (kCol + dx + d_grid_width) % d_grid_width;

      const int kNeighborLocalRow = kLocalRow + dy;

      bool neighbor_alive = false;

      if (kNeighborLocalRow >= 0 && kNeighborLocalRow < kNumRows) {
        // Use row-major indexing for O(1) access
        const int kNeighborIdx =
            kNeighborLocalRow * d_grid_width + kNeighborCol;
        neighbor_alive = agents[kNeighborIdx].alive;
      } else {
        if (kNeighborLocalRow < 0) {
          const int kNeighborIdx = kNeighborCol;
          if (kNeighborIdx < num_neighbors) {
            neighbor_alive = neighbors[kNeighborIdx].alive;
          }
        } else {
          const int kNeighborIdx = d_grid_width + kNeighborCol;
          if (kNeighborIdx < num_neighbors) {
            neighbor_alive = neighbors[kNeighborIdx].alive;
          }
        }
      }

      if (neighbor_alive) {
        ++alive_neighbors;
      }
    }
  }

  // Game of Life rules
  if (cell.alive) {
    cell.next_alive = (alive_neighbors == 2 || alive_neighbors == 3);
  } else {
    cell.next_alive = (alive_neighbors == 3);
  }
}

/**
 * @brief CUDA kernel to apply computed next states to current states.
 *
 * This separate kernel ensures synchronous update semantics by applying
 * all next_alive states to alive states after all cells have computed
 * their next states. This prevents race conditions where a cell might
 * read a neighbor's updated state during the same timestep.
 *
 * @param agents Device array of cells to update
 * @param num_agents Number of cells
 */
__global__ void ApplyNextStateKernel(Cell* agents, int num_agents) {
  const int kIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (kIdx >= num_agents) {
    return;
  }

  agents[kIdx].alive = agents[kIdx].next_alive;
}

GameOfLifeModel::GameOfLifeModel(int width, int height)
    : width_(width), height_(height) {
  // Copy grid dimensions to constant device memory
  cudaMemcpyToSymbol(d_grid_width, &width_, sizeof(int));
  cudaMemcpyToSymbol(d_grid_height, &height_, sizeof(int));
}

GameOfLifeModel::InteractionRuleCUDA GameOfLifeModel::GetInteractionKernel()
    const {
  return GameOfLifeKernel;
}

GameOfLifeModel::PostProcessKernelCUDA GameOfLifeModel::GetPostProcessKernel()
    const {
  return ApplyNextStateKernel;
}
