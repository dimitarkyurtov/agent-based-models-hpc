#include "Renderer.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>

#include "Cell.h"
#include "GameOfLifeSpace.h"

Renderer::Renderer(int width, int height, int window_width, int window_height)
    : width_(width),
      height_(height),
      window_width_(window_width),
      window_height_(window_height) {}

Renderer::~Renderer() {
  if (window_ != nullptr) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}

bool Renderer::Setup() {
  if (!glfwInit()) {
    std::cerr << "Error: Failed to initialize GLFW.\n";
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  window_ = glfwCreateWindow(window_width_, window_height_,
                             "Game of Life - Simulation", nullptr, nullptr);
  if (window_ == nullptr) {
    std::cerr << "Error: Failed to create GLFW window.\n";
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // Enable vsync

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  return true;
}

bool Renderer::Render(const GameOfLifeSpace& space, unsigned int timestep) {
  if (window_ == nullptr || glfwWindowShouldClose(window_)) {
    return false;
  }

  glfwPollEvents();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  RenderContent(space, timestep);

  ImGui::Render();
  int display_w = 0;
  int display_h = 0;
  glfwGetFramebufferSize(window_, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glfwSwapBuffers(window_);

  return true;
}

bool Renderer::ShouldClose() const noexcept {
  return window_ != nullptr && glfwWindowShouldClose(window_);
}

void Renderer::RenderContent(const GameOfLifeSpace& space,
                             unsigned int timestep) const {
  ImGui::Begin("Conway's Game of Life");

  ImGui::Text("Timestep: %u | Grid: %dx%d", timestep, width_, height_);
  ImGui::Separator();

  const float kGridWidth = static_cast<float>(width_) * kCellSize;
  const float kGridHeight = static_cast<float>(height_) * kCellSize;

  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

  const auto& game_of_life_space = static_cast<const GameOfLifeSpace&>(space);

  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      const Cell& cell = game_of_life_space.GetCellAt(x, y);

      ImVec2 cell_min(canvas_pos.x + static_cast<float>(x) * kCellSize,
                      canvas_pos.y + static_cast<float>(y) * kCellSize);
      ImVec2 cell_max(cell_min.x + kCellSize, cell_min.y + kCellSize);

      const float* color = cell.alive ? kAliveColor : kDeadColor;

      ImU32 col = ImGui::ColorConvertFloat4ToU32(
          ImVec4(color[0], color[1], color[2], color[3]));

      draw_list->AddRectFilled(cell_min, cell_max, col);
    }
  }

  ImGui::Dummy(ImVec2(kGridWidth, kGridHeight));

  ImGui::Spacing();
  ImGui::Separator();

  ImGui::Text("Legend:");

  ImGui::SameLine();
  ImGui::ColorButton(
      "##alive",
      ImVec4(kAliveColor[0], kAliveColor[1], kAliveColor[2], kAliveColor[3]), 0,
      ImVec2(20, 20));
  ImGui::SameLine();
  ImGui::Text("Alive Cell");

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
