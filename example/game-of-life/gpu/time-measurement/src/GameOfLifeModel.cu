#include <cuda_runtime.h>

#include "GameOfLifeModel.h"

// Global constants for grid dimensions (accessible from device code)
__constant__ int d_grid_width;
__constant__ int d_grid_height;

/**
 * @brief CUDA kernel implementing Game of Life rules for cell updates.
 *
 * Each thread processes one cell, counting alive neighbors using a
 * coordinate-based lookup in a combined agents+neighbors array.
 * Applies Conway's rules and updates both alive and next_alive states.
 * Uses constant memory for grid dimensions.
 *
 * @param agents Device array of local cells to update
 * @param num_agents Number of local cells
 * @param neighbors Device array of neighbor cells (boundary/ghost cells)
 * @param num_neighbors Number of neighbor cells
 */
__global__ void GameOfLifeKernel(Cell* agents, int num_agents, Cell* neighbors,
                                 int num_neighbors) {
  const int kIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check
  if (kIdx >= num_agents) {
    return;
  }

  Cell& cell = agents[kIdx];
  int alive_neighbors = 0;

  // Count alive neighbors in 3x3 neighborhood (excluding self)
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;  // Skip self
      }

      // Compute neighbor coordinates with toroidal wrapping
      const int kNx = (cell.x + dx + d_grid_width) % d_grid_width;
      const int kNy = (cell.y + dy + d_grid_height) % d_grid_height;

      // Search in local agents first
      bool found = false;
      for (int i = 0; i < num_agents; ++i) {
        if (agents[i].x == kNx && agents[i].y == kNy) {
          if (agents[i].alive) {
            ++alive_neighbors;
          }
          found = true;
          break;
        }
      }

      // If not found in local agents, search in neighbors array
      if (!found) {
        for (int i = 0; i < num_neighbors; ++i) {
          if (neighbors[i].x == kNx && neighbors[i].y == kNy) {
            if (neighbors[i].alive) {
              ++alive_neighbors;
            }
            break;
          }
        }
      }
    }
  }

  // Apply Conway's Game of Life rules
  if (cell.alive) {
    // Alive cell survives with 2 or 3 neighbors
    cell.next_alive = (alive_neighbors == 2 || alive_neighbors == 3);
  } else {
    // Dead cell becomes alive with exactly 3 neighbors
    cell.next_alive = (alive_neighbors == 3);
  }

  // Apply next state immediately (synchronous update within timestep)
  cell.alive = cell.next_alive;
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
