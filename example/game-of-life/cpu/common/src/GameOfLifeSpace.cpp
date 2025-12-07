#include "GameOfLifeSpace.h"

#include <ParallelABM/LocalRegion.h>
#include <ParallelABM/Logger.h>
#include <openssl/sha.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

GameOfLifeSpace::GameOfLifeSpace(int width, int height, double density,
                                 InitializationMode mode)
    : width_(width),
      height_(height),
      density_(density),
      init_mode_(mode),
      rng_(std::random_device{}()) {}

void GameOfLifeSpace::Initialize() {
  agents.clear();
  agents.reserve(static_cast<size_t>(width_ * height_));

  if (init_mode_ == InitializationMode::kRandom) {
    // Random initialization based on density
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const bool kAlive = dist(rng_) < density_;
        agents.push_back(Cell(x, y, kAlive));
      }
    }
  } else {
    // Deterministic patterns for reproducible testing
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        bool alive = false;

        // Glider pattern (top-left corner)
        if ((x == 1 && y == 0) || (x == 2 && y == 1) || (x == 0 && y == 2) ||
            (x == 1 && y == 2) || (x == 2 && y == 2)) {
          alive = true;
        }

        // Blinker pattern (center)
        const int kCenterX = width_ / 2;
        const int kCenterY = height_ / 2;
        if (y == kCenterY &&
            (x == kCenterX - 1 || x == kCenterX || x == kCenterX + 1)) {
          alive = true;
        }

        // Toad pattern (upper-right area)
        const int kToadX = width_ - 10;
        const int kToadY = 5;
        if ((y == kToadY &&
             (x == kToadX + 1 || x == kToadX + 2 || x == kToadX + 3)) ||
            (y == kToadY + 1 &&
             (x == kToadX || x == kToadX + 1 || x == kToadX + 2))) {
          alive = true;
        }

        // Beacon pattern (lower-left area)
        const int kBeaconX = 5;
        const int kBeaconY = height_ - 10;
        if ((x == kBeaconX && y == kBeaconY) ||
            (x == kBeaconX + 1 && y == kBeaconY) ||
            (x == kBeaconX && y == kBeaconY + 1) ||
            (x == kBeaconX + 3 && y == kBeaconY + 2) ||
            (x == kBeaconX + 2 && y == kBeaconY + 3) ||
            (x == kBeaconX + 3 && y == kBeaconY + 3)) {
          alive = true;
        }

        // Second glider (lower-right area, moving opposite direction)
        const int kGlider2X = width_ - 8;
        const int kGlider2Y = height_ - 8;
        if ((x == kGlider2X && y == kGlider2Y) ||
            (x == kGlider2X + 1 && y == kGlider2Y + 1) ||
            (x == kGlider2X + 2 && y == kGlider2Y - 1) ||
            (x == kGlider2X + 2 && y == kGlider2Y) ||
            (x == kGlider2X + 2 && y == kGlider2Y + 1)) {
          alive = true;
        }

        // Pulsar pattern (offset from center-left)
        const int kPulsarX = width_ / 4;
        const int kPulsarY = height_ / 2;
        if ((x == kPulsarX && (y == kPulsarY - 2 || y == kPulsarY + 2)) ||
            (x == kPulsarX + 2 &&
             (y == kPulsarY || y == kPulsarY - 2 || y == kPulsarY + 2)) ||
            (x == kPulsarX - 2 &&
             (y == kPulsarY || y == kPulsarY - 2 || y == kPulsarY + 2))) {
          alive = true;
        }

        agents.push_back(Cell(x, y, alive));
      }
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

    // Neighbors are now calculated dynamically via GetRegionNeighbours
    // on each timestep, not during region creation
    regions.emplace_back(region_id, std::move(indices));
    current_row += kRowsInRegion;
  }

  return regions;
}

std::vector<Cell> GameOfLifeSpace::GetRegionNeighbours(
    const Region& region) const {
  const std::vector<int>& indices = region.GetIndices();

  // Early return if region has no agents
  if (indices.empty()) {
    return {};
  }

  std::vector<Cell> neighbors;

  // Determine the first and last row of this region
  // Indices are stored in row-major order (y * width_ + x)
  const int kFirstIndex = indices.front();
  const int kLastIndex = indices.back();
  const int kFirstRow = kFirstIndex / width_;
  const int kLastRow = kLastIndex / width_;

  // Add cells from the row above this region (with toroidal wrap-around)
  int k_above_row = 0;
  if (kFirstRow > 0) {
    k_above_row = kFirstRow - 1;
  } else {
    // First row of grid: wrap to last row
    k_above_row = height_ - 1;
  }
  for (int col = 0; col < width_; ++col) {
    const int kIdx = CoordToIndex(col, k_above_row);
    neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
  }

  // Add cells from the row below this region (with toroidal wrap-around)
  int k_below_row = 0;
  if (kLastRow < height_ - 1) {
    k_below_row = kLastRow + 1;
  } else {
    // Last row of grid: wrap to first row
    k_below_row = 0;
  }
  for (int col = 0; col < width_; ++col) {
    const int kIdx = CoordToIndex(col, k_below_row);
    neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
  }

  return neighbors;
}

std::vector<ParallelABM::LocalSubRegion<Cell>>
GameOfLifeSpace::SplitLocalRegion(ParallelABM::LocalRegion<Cell>& region,
                                  int num_subregions) const {
  std::vector<ParallelABM::LocalSubRegion<Cell>> subregions;
  subregions.reserve(static_cast<size_t>(num_subregions));

  const std::vector<Cell>& agents = region.GetAgents();
  const int kTotalAgents = static_cast<int>(agents.size());

  const int kTotalRows = kTotalAgents / width_;
  const int kBaseRowsPerSubregion = kTotalRows / num_subregions;
  const int kExtraRows = kTotalRows % num_subregions;

  int current_row = 0;

  for (int i = 0; i < num_subregions; ++i) {
    // Distribute extra rows among first few subregions
    const int kRowsInSubregion =
        kBaseRowsPerSubregion + (i < kExtraRows ? 1 : 0);
    const int kAgentsInSubregion = kRowsInSubregion * width_;
    const int kStartIndex = current_row * width_;

    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(kAgentsInSubregion));

    for (int j = 0; j < kAgentsInSubregion; ++j) {
      indices.push_back(kStartIndex + j);
    }

    // Calculate neighbor indices for this subregion
    // Positive indices (0, 1, 2, ...) refer to agents in the parent region
    // Negative indices (-1, -2, -3, ...) refer to parent's neighbors where
    // -1 = parent_neighbors[0], -2 = parent_neighbors[1], etc.
    std::vector<int> neighbor_indices;

    const auto& parent_neighbors = region.GetNeighbors();
    const bool kRegionSpansFullGrid = (kTotalRows == height_);

    if (i > 0) {
      const int kPrevSubregionLastRowStart = (current_row - 1) * width_;
      for (int col = 0; col < width_; ++col) {
        neighbor_indices.push_back(kPrevSubregionLastRowStart + col);
      }
    } else {
      if (kRegionSpansFullGrid && num_subregions > 1) {
        const int kLastSubregionLastRowStart = (kTotalRows - 1) * width_;
        for (int col = 0; col < width_; ++col) {
          neighbor_indices.push_back(kLastSubregionLastRowStart + col);
        }
      } else {
        const size_t kUpperBoundarySize =
            std::min(static_cast<size_t>(width_), parent_neighbors.size());

        for (size_t n = 0; n < kUpperBoundarySize; ++n) {
          neighbor_indices.push_back(-static_cast<int>(n + 1));
        }
      }
    }

    if (i < num_subregions - 1) {
      const int kNextSubregionFirstRowStart =
          (current_row + kRowsInSubregion) * width_;
      for (int col = 0; col < width_; ++col) {
        neighbor_indices.push_back(kNextSubregionFirstRowStart + col);
      }
    } else {
      if (kRegionSpansFullGrid && num_subregions > 1) {
        for (int col = 0; col < width_; ++col) {
          neighbor_indices.push_back(col);
        }
      } else {
        if (parent_neighbors.size() > static_cast<size_t>(width_)) {
          const size_t kStartIdx = parent_neighbors.size() - width_;

          for (size_t n = kStartIdx; n < parent_neighbors.size(); ++n) {
            neighbor_indices.push_back(-static_cast<int>(n + 1));
          }
        }
      }
    }

    subregions.emplace_back(std::move(indices), &region,
                            std::move(neighbor_indices));
    current_row += kRowsInSubregion;
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

void GameOfLifeSpace::Serialize(std::ostream& os, int step) const {
  // Build binary grid state (0 or 1 for each cell)
  std::vector<unsigned char> grid_state;
  grid_state.reserve(agents.size());

  int alive_count = 0;
  for (const auto& cell : agents) {
    const unsigned char kCellState = cell.alive ? 1 : 0;
    grid_state.push_back(kCellState);
    if (kCellState == 1) {
      ++alive_count;
    }
  }

  // Compute SHA-256 hash
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(grid_state.data(), grid_state.size(), hash);

  // Convert hash to hex string
  std::ostringstream hash_hex;
  hash_hex << std::hex << std::setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    hash_hex << std::setw(2) << static_cast<int>(hash[i]);
  }

  // Write checkpoint data
  os << "STEP: " << step << "\n";
  os << "CHECKSUM: " << hash_hex.str() << "\n";
  os << "ALIVE_CELLS: " << alive_count << "\n";
}
