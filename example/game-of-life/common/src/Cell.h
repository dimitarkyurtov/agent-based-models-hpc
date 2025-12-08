#ifndef GAMEOFLIFE_CELL_H
#define GAMEOFLIFE_CELL_H

/**
 * @struct Cell
 * @brief Represents a single cell in Conway's Game of Life.
 *
 * A simple POD-like structure used with the template-based ParallelABM library.
 * Each cell maintains its current state, next computed state, and grid
 * position. Trivially copyable for efficient MPI communication.
 */
struct Cell {
  bool alive;       ///< Current state: true if cell is alive
  bool next_alive;  ///< Next state computed during timestep update
  int x;            ///< X coordinate in the grid
  int y;            ///< Y coordinate in the grid

  /**
   * @brief Default constructor creating a dead cell at origin.
   */
  Cell();

  /**
   * @brief Construct a cell with specified position and state.
   * @param x X coordinate in the grid
   * @param y Y coordinate in the grid
   * @param alive Initial alive state
   */
  Cell(int x, int y, bool alive);

  /**
   * @brief Copy constructor.
   */
  Cell(const Cell&) = default;

  /**
   * @brief Copy assignment operator.
   */
  Cell& operator=(const Cell&) = default;

  /**
   * @brief Move constructor.
   */
  Cell(Cell&&) = default;

  /**
   * @brief Move assignment operator.
   */
  Cell& operator=(Cell&&) = default;

  /**
   * @brief Destructor.
   */
  ~Cell() = default;

  /**
   * @brief Apply the computed next state to the current state.
   *
   * Called after all cells have computed their next_alive values
   * to synchronously update the grid.
   */
  void ApplyNextState();
};

#endif  // GAMEOFLIFE_CELL_H
