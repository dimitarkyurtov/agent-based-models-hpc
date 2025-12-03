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

  // Deterministic patterns for reproducible testing across multiple threads
  // Creates various oscillators and gliders in different locations
  // for (int x = 0; x < width_; ++x) {
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
      // Simplified pulsar (just a few key cells for visual interest)
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

  // Random initialization (commented out for deterministic testing)
  // std::uniform_real_distribution<double> dist(0.0, 1.0);
  // for (int y = 0; y < height_; ++y) {
  //   for (int x = 0; x < width_; ++x) {
  //     const bool kAlive = dist(rng_) < density_;
  //     agents.push_back(Cell(x, y, kAlive));
  //   }
  // }
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
    // For toroidal topology: first region wraps to last row, last region wraps
    // to first row
    std::vector<Cell> neighbors;

    // Add cells from the row above this region (with wrap-around)
    int k_above_row = 0;
    if (current_row > 0) {
      k_above_row = current_row - 1;
    } else {
      // First region: wrap to last row of the grid
      k_above_row = height_ - 1;
    }
    for (int col = 0; col < width_; ++col) {
      const int kIdx = CoordToIndex(col, k_above_row);
      neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
    }

    // Add cells from the row below this region (with wrap-around)
    const int kLastRow = current_row + kRowsInRegion - 1;
    int k_below_row = 0;
    if (kLastRow < height_ - 1) {
      k_below_row = kLastRow + 1;
    } else {
      // Last region: wrap to first row of the grid
      k_below_row = 0;
    }
    for (int col = 0; col < width_; ++col) {
      const int kIdx = CoordToIndex(col, k_below_row);
      neighbors.push_back(agents[static_cast<size_t>(kIdx)]);
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

  // Calculate rows in this region (agents are in row-major order)
  const int kTotalRows = kTotalAgents / width_;
  const int kBaseRowsPerSubregion = kTotalRows / num_subregions;
  const int kExtraRows = kTotalRows % num_subregions;

  std::cout << "Splitting local region into " << num_subregions
            << " subregions\n";
  std::cout << "Total agents: " << kTotalAgents
            << ", Total rows: " << kTotalRows
            << ", Base rows per subregion: " << kBaseRowsPerSubregion
            << ", Extra rows: " << kExtraRows << "\n";

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

    // Parent neighbors format (from SplitIntoRegions):
    // - First width_ cells: row above the region (always present)
    // - Last width_ cells: row below the region (always present)
    const auto& parent_neighbors = region.GetNeighbors();
    const bool kRegionSpansFullGrid = (kTotalRows == height_);

    // Add neighbor indices from row above this subregion
    if (i > 0) {
      // Not first subregion: get bottommost row of previous subregion
      const int kPrevSubregionLastRowStart = (current_row - 1) * width_;
      for (int col = 0; col < width_; ++col) {
        neighbor_indices.push_back(kPrevSubregionLastRowStart + col);
      }
    } else {
      // First subregion: needs upper boundary from parent
      if (kRegionSpansFullGrid && num_subregions > 1) {
        // Region spans full grid: wrap to last row of last subregion
        const int kLastSubregionLastRowStart = (kTotalRows - 1) * width_;
        for (int col = 0; col < width_; ++col) {
          neighbor_indices.push_back(kLastSubregionLastRowStart + col);
        }
      } else {
        // Region doesn't span full grid: get parent's upper boundary
        // Use negative indices to reference parent neighbors
        const size_t kUpperBoundarySize =
            std::min(static_cast<size_t>(width_), parent_neighbors.size());
        for (size_t n = 0; n < kUpperBoundarySize; ++n) {
          neighbor_indices.push_back(-static_cast<int>(n + 1));
        }
      }
    }

    // Add neighbor indices from row below this subregion
    if (i < num_subregions - 1) {
      // Not last subregion: get topmost row of next subregion
      const int kNextSubregionFirstRowStart =
          (current_row + kRowsInSubregion) * width_;
      for (int col = 0; col < width_; ++col) {
        neighbor_indices.push_back(kNextSubregionFirstRowStart + col);
      }
    } else {
      // Last subregion: needs lower boundary from parent
      if (kRegionSpansFullGrid && num_subregions > 1) {
        // Region spans full grid: wrap to first row of first subregion
        for (int col = 0; col < width_; ++col) {
          neighbor_indices.push_back(col);
        }
      } else {
        // Region doesn't span full grid: get parent's lower boundary
        // Use negative indices to reference parent neighbors
        if (parent_neighbors.size() > static_cast<size_t>(width_)) {
          const size_t kStartIdx = parent_neighbors.size() - width_;
          for (size_t n = kStartIdx; n < parent_neighbors.size(); ++n) {
            neighbor_indices.push_back(-static_cast<int>(n + 1));
          }
        }
      }
    }

    std::cout << "  Subregion " << i << ": rows " << current_row << " to "
              << (current_row + kRowsInSubregion - 1) << ", agents "
              << kStartIndex << " to " << (kStartIndex + kAgentsInSubregion - 1)
              << " (count: " << kAgentsInSubregion
              << "), neighbor_indices: " << neighbor_indices.size() << "\n";

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
  // Row-major order: y * width_ + x
  return y * width_ + x;
}
