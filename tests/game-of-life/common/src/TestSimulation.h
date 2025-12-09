#ifndef TEST_TESTSIMULATION_H
#define TEST_TESTSIMULATION_H

#include <ParallelABM/SimulationCPU.h>

#include <fstream>
#include <string>

#include "TestAgent.h"
#include "TestSpace.h"

/**
 * @class TestSimulation
 * @brief Simulation for test checkpoints with all steps in a single file.
 *
 * Extends SimulationCPU<TestAgent> to save checkpoints at regular intervals,
 * appending each checkpoint to a single file.
 */
class TestSimulation : public ParallelABM::SimulationCPU<TestAgent> {
 public:
  /**
   * @brief Construct a test simulation with checkpoint configuration.
   * @param argc Command line argument count (passed to MPI)
   * @param argv Command line arguments (passed to MPI)
   * @param space The game space containing agents
   * @param model Shared pointer to the CPU model with interaction rules
   * @param environment Compute environment configuration
   * @param checkpoint_file Path to the single checkpoint file
   * @param checkpoint_interval Number of steps between checkpoints
   */
  TestSimulation(int& argc, char**& argv,
                 std::unique_ptr<Space<TestAgent>> space,
                 std::shared_ptr<ModelCPU<TestAgent>> model,
                 ParallelABM::Environment& environment,
                 const std::string& checkpoint_file, int checkpoint_interval);

  /**
   * @brief Called after each timestep to save checkpoints.
   * @param timestep The completed timestep number
   */
  void OnTimeStepCompleted(unsigned int timestep) override;

  /**
   * @brief Virtual destructor.
   */
  ~TestSimulation() override;

 private:
  std::string checkpoint_file_;      ///< Path to checkpoint file
  int checkpoint_interval_;          ///< Steps between checkpoints
  std::ofstream checkpoint_stream_;  ///< File stream for checkpoints
};

#endif  // TEST_TESTSIMULATION_H
