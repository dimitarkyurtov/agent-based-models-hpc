#ifndef TEST_TESTMODEL_H
#define TEST_TESTMODEL_H

#include <ParallelABM/ModelCPU.h>

#include <functional>
#include <vector>

#include "TestAgent.h"

/**
 * @class TestModel
 * @brief CPU implementation of Game of Life rules for test simulations.
 *
 * Derives from ModelCPU<TestAgent> and implements the ComputeInteractions
 * virtual method to define the Game of Life logic with type-safe TestAgent
 * references.
 */
class TestModel : public ModelCPU<TestAgent> {
 public:
  /**
   * @brief Constructs a TestModel with specified grid dimensions.
   * @param width Grid width for neighbor calculation
   * @param height Grid height for neighbor calculation
   */
  TestModel(int width, int height);

  /**
   * @brief Copy constructor.
   */
  TestModel(const TestModel&) = default;

  /**
   * @brief Copy assignment operator.
   */
  TestModel& operator=(const TestModel&) = default;

  /**
   * @brief Move constructor.
   */
  TestModel(TestModel&&) = default;

  /**
   * @brief Move assignment operator.
   */
  TestModel& operator=(TestModel&&) = default;

  /**
   * @brief Virtual destructor.
   */
  ~TestModel() override = default;

  /**
   * @brief Compute agent interactions implementing Game of Life rules.
   *
   * Implements Conway's Game of Life rules:
   * - Any live agent with 2 or 3 live neighbors survives
   * - Any dead agent with exactly 3 live neighbors becomes alive
   * - All other agents die or stay dead
   *
   * @param agents Vector of references to agents to process (mutable)
   * @param neighbors Vector of references to neighbor agents from adjacent
   * regions (read-only)
   */
  void ComputeInteractions(
      std::vector<std::reference_wrapper<TestAgent>>& agents,
      const std::vector<TestAgent>& neighbors) override;

 private:
  int width_;   ///< Grid width for neighbor calculation
  int height_;  ///< Grid height for neighbor calculation
};

#endif  // TEST_TESTMODEL_H
