#include "GameOfLifeModel.h"

#include <ParallelABM/Logger.h>

#include <iostream>
#include <unordered_map>

GameOfLifeModel::GameOfLifeModel(int width, int height)
    : width_(width), height_(height) {}

void GameOfLifeModel::ComputeInteractions(
    std::vector<std::reference_wrapper<Cell>>& agents,
    [[maybe_unused]] const std::vector<Cell>& neighbors) {
  // Build coordinate-to-cell lookup map for O(1) neighbor access
  std::unordered_map<int, const Cell*> cell_map;
  cell_map.reserve(agents.size() + neighbors.size());

  for (const Cell& cell : agents) {
    const int kKey = cell.y * width_ + cell.x;
    cell_map[kKey] = &cell;
  }

  for (const Cell& neighbor_cell : neighbors) {
    const int kKey = neighbor_cell.y * width_ + neighbor_cell.x;
    cell_map[kKey] = &neighbor_cell;
  }

  for (Cell& cell : agents) {
    int alive_neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }

        const int kNx = (cell.x + dx + width_) % width_;
        const int kNy = (cell.y + dy + height_) % height_;
        const int kKey = kNy * width_ + kNx;

        // O(1) lookup in hash map
        auto it = cell_map.find(kKey);
        if (it != cell_map.end() && it->second->alive) {
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

  // Second pass: apply computed next states
  for (Cell& cell : agents) {
    cell.ApplyNextState();
  }
}
