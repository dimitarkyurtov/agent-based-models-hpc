#include "Renderer.h"

#include <imgui.h>

#include <iostream>

#include "Cell.h"
#include "GameOfLifeSpace.h"

Renderer::Renderer(int width, int height) : width_(width), height_(height) {}

void Renderer::Render(const GameOfLifeSpace& space,
                      unsigned int timestep) const {
  // Create ImGui window with title
  ImGui::Begin("Conway's Game of Life");

  // Display timestep and grid information
  ImGui::Text("Timestep: %u | Grid: %dx%d", timestep, width_, height_);
  ImGui::Separator();

  // Calculate window size based on grid dimensions
  const float kGridWidth = static_cast<float>(width_) * kCellSize;
  const float kGridHeight = static_cast<float>(height_) * kCellSize;

  // Get the draw list for rendering cells
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

  // Render each cell in the grid
  const auto& game_of_life_space = static_cast<const GameOfLifeSpace&>(space);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const Cell& cell = game_of_life_space.GetCellAt(x, y);

      // Calculate cell position on screen
      ImVec2 cell_min(canvas_pos.x + static_cast<float>(x) * kCellSize,
                      canvas_pos.y + static_cast<float>(y) * kCellSize);
      ImVec2 cell_max(cell_min.x + kCellSize, cell_min.y + kCellSize);

      // Choose color based on cell state
      const float* color = cell.alive ? kAliveColor : kDeadColor;
      // const float* color = kAliveColor;
      ImU32 col = ImGui::ColorConvertFloat4ToU32(
          ImVec4(color[0], color[1], color[2], color[3]));

      // Draw filled rectangle for the cell
      draw_list->AddRectFilled(cell_min, cell_max, col);
    }
  }

  // Reserve space for the grid
  ImGui::Dummy(ImVec2(kGridWidth, kGridHeight));

  // Add spacing before legend
  ImGui::Spacing();
  ImGui::Separator();

  // Display legend
  ImGui::Text("Legend:");

  // Alive cell indicator
  ImGui::SameLine();
  ImGui::ColorButton(
      "##alive",
      ImVec4(kAliveColor[0], kAliveColor[1], kAliveColor[2], kAliveColor[3]), 0,
      ImVec2(20, 20));
  ImGui::SameLine();
  ImGui::Text("Alive Cell");

  // Dead cell indicator
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  ImGui::ColorButton(
      "##dead",
      ImVec4(kDeadColor[0], kDeadColor[1], kDeadColor[2], kDeadColor[3]), 0,
      ImVec2(20, 20));
  ImGui::SameLine();
  ImGui::Text("Dead Cell");

  ImGui::End();
}
