#include "GameOfLifeModel.h"

GameOfLifeModel::GameOfLifeModel(int width, int height)
    : width_(width), height_(height) {}

void GameOfLifeModel::ComputeInteractions(
    std::vector<std::reference_wrapper<Cell>>& agents,
    [[maybe_unused]] const std::vector<std::reference_wrapper<const Cell>>&
        neighbors) {
  // First pass: compute next_alive for all cells
  for (Cell& cell : agents) {
    // Count alive neighbors (8-connected)
    int alive_neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;  // Skip self
        }

        // Calculate neighbor coordinates with wrapping (toroidal grid)
        const int kNx = (cell.x + dx + width_) % width_;
        const int kNy = (cell.y + dy + height_) % height_;

        // Search in local agents first
        bool found = false;
        for (const Cell& other_cell : agents) {
          if (other_cell.x == kNx && other_cell.y == kNy) {
            if (other_cell.alive) {
              ++alive_neighbors;
            }
            found = true;
            break;
          }
        }

        // If not found in local agents, search in neighbors
        if (!found) {
          for (const Cell& neighbor_cell : neighbors) {
            if (neighbor_cell.x == kNx && neighbor_cell.y == kNy) {
              if (neighbor_cell.alive) {
                ++alive_neighbors;
              }
              break;
            }
          }
        }
      }
    }

    // Apply Game of Life rules
    if (cell.alive) {
      // Alive cell survives with 2 or 3 neighbors
      cell.next_alive = (alive_neighbors == 2 || alive_neighbors == 3);
    } else {
      // Dead cell becomes alive with exactly 3 neighbors
      cell.next_alive = (alive_neighbors == 3);
    }
  }

  // Second pass: apply computed next states
  for (Cell& cell : agents) {
    cell.ApplyNextState();
  }
}
