#ifndef GAMEOFLIFE_RENDERER_H
#define GAMEOFLIFE_RENDERER_H

#include <GLFW/glfw3.h>

#include "Cell.h"
#include "GameOfLifeSpace.h"

/**
 * @class Renderer
 * @brief ImGui-based renderer for Game of Life visualization.
 *
 * Manages the complete rendering pipeline including window creation,
 * OpenGL/ImGui initialization, and frame-by-frame rendering of the
 * Game of Life grid.
 */
class Renderer {
 public:
  /**
   * @brief Construct a renderer for a grid of specified dimensions.
   * @param width Grid width (number of columns)
   * @param height Grid height (number of rows)
   * @param window_width Window width in pixels (default: 1280)
   * @param window_height Window height in pixels (default: 720)
   */
  Renderer(int width, int height, int window_width = 1280,
           int window_height = 720);

  /**
   * @brief Destructor - cleans up GLFW and ImGui resources.
   */
  ~Renderer();

  // Disable copying and moving to prevent resource management issues
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;

  /**
   * @brief Initialize GLFW window and ImGui context.
   *
   * Sets up the rendering environment including:
   * - GLFW initialization and window creation
   * - OpenGL context configuration
   * - ImGui context and backends initialization
   *
   * @return true if setup succeeded, false otherwise
   */
  bool Setup();

  /**
   * @brief Render a single frame with the current space state.
   *
   * Performs full frame rendering including:
   * - Event polling and input handling
   * - ImGui frame preparation
   * - Grid visualization
   * - Buffer swapping
   *
   * @param space The spatial structure containing all cells
   * @param timestep Current simulation timestep for display
   * @return true if rendering succeeded and window is still open
   */
  bool Render(const GameOfLifeSpace& space, unsigned int timestep);

  /**
   * @brief Check if the window should close.
   * @return true if window close was requested
   */
  [[nodiscard]] bool ShouldClose() const noexcept;

 private:
  /**
   * @brief Render the ImGui window content with the grid.
   * @param space The spatial structure containing all cells
   * @param timestep Current simulation timestep for display
   */
  void RenderContent(const GameOfLifeSpace& space, unsigned int timestep) const;
  int width_;                    ///< Grid width
  int height_;                   ///< Grid height
  int window_width_;             ///< Window width in pixels
  int window_height_;            ///< Window height in pixels
  GLFWwindow* window_{nullptr};  ///< GLFW window handle

  /// Cell size in pixels for rendering
  static constexpr float kCellSize = 10.0f;

  /// Color for alive cells (green)
  static constexpr float kAliveColor[4] = {0.0f, 0.8f, 0.0f, 1.0f};

  /// Color for dead cells (dark gray)
  static constexpr float kDeadColor[4] = {0.2f, 0.2f, 0.2f, 1.0f};
};

#endif  // GAMEOFLIFE_RENDERER_H
