#ifndef TEST_TESTAGENT_H
#define TEST_TESTAGENT_H

/**
 * @struct TestAgent
 * @brief Represents a single agent in test simulations.
 *
 * A simple POD-like structure used with the template-based ParallelABM library.
 * Each agent maintains its current state, next computed state, and grid
 * position. Trivially copyable for efficient MPI communication.
 */
struct TestAgent {
  bool alive;       ///< Current state: true if agent is alive
  bool next_alive;  ///< Next state computed during timestep update
  int x;            ///< X coordinate in the grid
  int y;            ///< Y coordinate in the grid

  /**
   * @brief Default constructor creating a dead agent at origin.
   */
  TestAgent();

  /**
   * @brief Construct an agent with specified position and state.
   * @param x X coordinate in the grid
   * @param y Y coordinate in the grid
   * @param alive Initial alive state
   */
  TestAgent(int x, int y, bool alive);

  /**
   * @brief Copy constructor.
   */
  TestAgent(const TestAgent&) = default;

  /**
   * @brief Copy assignment operator.
   */
  TestAgent& operator=(const TestAgent&) = default;

  /**
   * @brief Move constructor.
   */
  TestAgent(TestAgent&&) = default;

  /**
   * @brief Move assignment operator.
   */
  TestAgent& operator=(TestAgent&&) = default;

  /**
   * @brief Destructor.
   */
  ~TestAgent() = default;

  /**
   * @brief Apply the computed next state to the current state.
   *
   * Called after all agents have computed their next_alive values
   * to synchronously update the grid.
   */
  void ApplyNextState();
};

#endif  // TEST_TESTAGENT_H
