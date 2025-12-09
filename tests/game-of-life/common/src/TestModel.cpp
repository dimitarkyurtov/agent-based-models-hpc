#include "TestModel.h"

#include <ParallelABM/Logger.h>

#include <iostream>
#include <unordered_map>

TestModel::TestModel(int width, int height) : width_(width), height_(height) {}

void TestModel::ComputeInteractions(
    std::vector<std::reference_wrapper<TestAgent>>& agents,
    [[maybe_unused]] const std::vector<TestAgent>& neighbors) {
  // Build coordinate-to-agent lookup map for O(1) neighbor access
  std::unordered_map<int, const TestAgent*> agent_map;
  agent_map.reserve(agents.size() + neighbors.size());

  for (const TestAgent& agent : agents) {
    const int kKey = agent.y * width_ + agent.x;
    agent_map[kKey] = &agent;
  }

  for (const TestAgent& neighbor_agent : neighbors) {
    const int kKey = neighbor_agent.y * width_ + neighbor_agent.x;
    agent_map[kKey] = &neighbor_agent;
  }

  for (TestAgent& agent : agents) {
    int alive_neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;  // Skip self
        }

        const int kNx = (agent.x + dx + width_) % width_;
        const int kNy = (agent.y + dy + height_) % height_;
        const int kKey = kNy * width_ + kNx;

        // O(1) lookup in hash map
        auto it = agent_map.find(kKey);
        if (it != agent_map.end() && it->second->alive) {
          ++alive_neighbors;
        }
      }
    }

    // Apply Game of Life rules
    if (agent.alive) {
      // Alive agent survives with 2 or 3 neighbors
      agent.next_alive = (alive_neighbors == 2 || alive_neighbors == 3);
    } else {
      // Dead agent becomes alive with exactly 3 neighbors
      agent.next_alive = (alive_neighbors == 3);
    }
  }

  // Second pass: apply computed next states
  for (TestAgent& agent : agents) {
    agent.ApplyNextState();
  }
}
