#include "GameOfLifeModel.h"

#include <unordered_map>

GameOfLifeModel::GameOfLifeModel(int width, int height)
    : width_(width), height_(height) {}

void GameOfLifeModel::ComputeInteractions(
    std::vector<std::reference_wrapper<Cell>>& agents,
    [[maybe_unused]] const std::vector<std::reference_wrapper<const Cell>>&
        neighbors) {
  // Build coordinate-to-cell lookup map for O(1) neighbor access
  std::unordered_map<int, const Cell*> cell_map;
  cell_map.reserve(agents.size() + neighbors.size());

  for (const Cell& cell : agents) {
    const int key = cell.y * width_ + cell.x;
    cell_map[key] = &cell;
  }

  for (const Cell& neighbor_cell : neighbors) {
    const int key = neighbor_cell.y * width_ + neighbor_cell.x;
    cell_map[key] = &neighbor_cell;
  }

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
        const int key = kNy * width_ + kNx;

        // O(1) lookup in hash map
        auto it = cell_map.find(key);
        if (it != cell_map.end() && it->second->alive) {
          ++alive_neighbors;
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
