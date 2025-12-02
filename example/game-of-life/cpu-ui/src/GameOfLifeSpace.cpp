#include "GameOfLifeSpace.h"

#include <ParallelABM/LocalRegion.h>

#include <algorithm>
#include <iostream>

GameOfLifeSpace::GameOfLifeSpace(int width, int height, double density)
    : width_(width),
      height_(height),
      density_(density),
      rng_(std::random_device{}()) {}

void GameOfLifeSpace::Initialize() {
  agents.clear();
  agents.reserve(static_cast<size_t>(width_ * height_));

  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Create cells in row-major order (direct value storage)
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const bool kAlive = dist(rng_) < density_;
      std::cout << "Creating cell at (" << x << ", " << y
                << ") with state: " << (kAlive ? "Alive" : "Dead") << "\n";
      agents.push_back(Cell(x, y, kAlive));
    }
  }
}

std::vector<Space<Cell>::Region> GameOfLifeSpace::SplitIntoRegions(
    int num_regions) const {
  std::vector<Region> regions;
  regions.reserve(static_cast<size_t>(num_regions));

  const int kTotalRows = height_;
  const int kBaseRowsPerRegion = kTotalRows / num_regions;
  const int kExtraRows = kTotalRows % num_regions;

  int current_row = 0;

  for (int region_id = 0; region_id < num_regions; ++region_id) {
    // Distribute extra rows among first few regions
    const int kRowsInRegion =
        kBaseRowsPerRegion + (region_id < kExtraRows ? 1 : 0);

    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(kRowsInRegion * width_));

    // Add all cell indices for rows in this region
    for (int row = current_row; row < current_row + kRowsInRegion; ++row) {
      for (int col = 0; col < width_; ++col) {
        indices.push_back(CoordToIndex(col, row));
      }
    }

    // Collect neighbor cells from adjacent regions (boundary rows)
    std::vector<Cell> neighbors;

    // Add cells from the row above this region (if exists)
    if (current_row > 0) {
      const int kAboveRow = current_row - 1;
      for (int col = 0; col < width_; ++col) {
        const int kIdx = CoordToIndex(col, kAboveRow);
        neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
      }
    }

    // Add cells from the row below this region (if exists)
    const int kLastRow = current_row + kRowsInRegion - 1;
    if (kLastRow < height_ - 1) {
      const int kBelowRow = kLastRow + 1;
      for (int col = 0; col < width_; ++col) {
        const int kIdx = CoordToIndex(col, kBelowRow);
        neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
      }
    }

    regions.emplace_back(region_id, std::move(indices), std::move(neighbors));
    current_row += kRowsInRegion;
  }

  return regions;
}

std::vector<ParallelABM::LocalSubRegion<Cell>>
GameOfLifeSpace::SplitLocalRegion(ParallelABM::LocalRegion<Cell>& region,
                                  int num_subregions) const {
  std::vector<ParallelABM::LocalSubRegion<Cell>> subregions;
  subregions.reserve(static_cast<size_t>(num_subregions));

  const std::vector<Cell>& agents = region.GetAgents();
  const int kTotalAgents = static_cast<int>(agents.size());
  const int kBaseAgentsPerSubregion = kTotalAgents / num_subregions;
  const int kExtraAgents = kTotalAgents % num_subregions;

  int current_index = 0;

  for (int i = 0; i < num_subregions; ++i) {
    // Distribute extra agents among first few subregions
    const int kAgentsInSubregion =
        kBaseAgentsPerSubregion + (i < kExtraAgents ? 1 : 0);

    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(kAgentsInSubregion));

    for (int j = 0; j < kAgentsInSubregion; ++j) {
      indices.push_back(current_index + j);
    }

    subregions.emplace_back(std::move(indices), &region, region.GetNeighbors());
    current_index += kAgentsInSubregion;
  }

  return subregions;
}

Cell& GameOfLifeSpace::GetCellAt(int x, int y) {
  return agents[static_cast<size_t>(CoordToIndex(x, y))];
}

const Cell& GameOfLifeSpace::GetCellAt(int x, int y) const {
  return agents[static_cast<size_t>(CoordToIndex(x, y))];
}

int GameOfLifeSpace::CoordToIndex(int x, int y) const noexcept {
  return y * width_ + x;
}
