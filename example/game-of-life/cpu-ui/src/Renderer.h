#ifndef GAMEOFLIFE_RENDERER_H
#define GAMEOFLIFE_RENDERER_H

#include "Cell.h"
#include "GameOfLifeSpace.h"

/**
 * @class Renderer
 * @brief ImGui-based renderer for Game of Life visualization.
 *
 * Uses ImGui to render the Game of Life grid in a graphical window.
 * Alive cells are displayed with a distinct color from dead cells,
 * along with a legend and title.
 */
class Renderer {
 public:
  /**
   * @brief Construct a renderer for a grid of specified dimensions.
   * @param width Grid width (number of columns)
   * @param height Grid height (number of rows)
   */
  Renderer(int width, int height);

  /**
   * @brief Default destructor.
   */
  ~Renderer() = default;

  /**
   * @brief Copy constructor.
   */
  Renderer(const Renderer&) = default;

  /**
   * @brief Copy assignment operator.
   */
  Renderer& operator=(const Renderer&) = default;

  /**
   * @brief Move constructor.
   */
  Renderer(Renderer&&) = default;

  /**
   * @brief Move assignment operator.
   */
  Renderer& operator=(Renderer&&) = default;

  /**
   * @brief Render the current state of the space using ImGui.
   *
   * Creates an ImGui window displaying the Game of Life grid with
   * alive and dead cells in different colors, along with a legend
   * and title showing the current timestep.
   *
   * @param space The spatial structure containing all cells
   * @param timestep Current simulation timestep for display
   */
  void Render(const GameOfLifeSpace& space, unsigned int timestep) const;

 private:
  int width_;   ///< Grid width
  int height_;  ///< Grid height

  /// Cell size in pixels for rendering
  static constexpr float kCellSize = 10.0f;

  /// Color for alive cells (green)
  static constexpr float kAliveColor[4] = {0.0f, 0.8f, 0.0f, 1.0f};

  /// Color for dead cells (dark gray)
  static constexpr float kDeadColor[4] = {0.2f, 0.2f, 0.2f, 1.0f};
};

#endif  // GAMEOFLIFE_RENDERER_H
