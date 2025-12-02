/**
 * @file main.cpp
 * @brief Game of Life example using the ParallelABM library.
 *
 * Demonstrates Conway's Game of Life simulation with ImGui-based
 * visualization using the ParallelABM library's CPU simulation capabilities.
 */

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "GameOfLifeSpace.h"
#include "Renderer.h"

namespace {

/// Default grid width
constexpr int kDefaultWidth = 50;

/// Default grid height
constexpr int kDefaultHeight = 25;

/// Default initial alive cell density
constexpr double kDefaultDensity = 0.3;

/// Window width
constexpr int kWindowWidth = 1280;

/// Window height
constexpr int kWindowHeight = 720;

/**
 * @brief Print usage information to stderr.
 * @param program_name Name of the executable
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [width] [height] [density]\n"
            << "\n"
            << "Arguments:\n"
            << "  width   - Grid width (default: " << kDefaultWidth << ")\n"
            << "  height  - Grid height (default: " << kDefaultHeight << ")\n"
            << "  density - Initial alive cell density 0.0-1.0 (default: "
            << kDefaultDensity << ")\n";
}

}  // namespace

/**
 * @brief Main entry point for the Game of Life visualization.
 *
 * Parses command line arguments, initializes ImGui and GLFW,
 * creates the initial Game of Life state, and displays it.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code (0 on success)
 */
int main(int argc, char* argv[]) {
  // Parse command line arguments
  int width = kDefaultWidth;
  int height = kDefaultHeight;
  double density = kDefaultDensity;

  if (argc > 1) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      PrintUsage(argv[0]);
      return 0;
    }
    width = std::atoi(argv[1]);
  }
  if (argc > 2) {
    height = std::atoi(argv[2]);
  }
  if (argc > 3) {
    density = std::atof(argv[3]);
  }

  // Validate parameters
  if (width <= 0 || height <= 0) {
    std::cerr << "Error: Width and height must be positive.\n";
    PrintUsage(argv[0]);
    return 1;
  }

  if (density < 0.0 || density > 1.0) {
    std::cerr << "Error: Density must be between 0.0 and 1.0.\n";
    return 1;
  }

  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Error: Failed to initialize GLFW.\n";
    return 1;
  }

  // Set OpenGL version (3.3 Core)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // Create window
  GLFWwindow* window =
      glfwCreateWindow(kWindowWidth, kWindowHeight,
                       "Game of Life - Initial State", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Error: Failed to create GLFW window.\n";
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Initialize ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Setup ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Create space and initialize
  auto space = std::make_unique<GameOfLifeSpace>(width, height, density);
  space->Initialize();

  // Create renderer
  Renderer renderer(width, height);

  // Main render loop - displays initial state
  while (!glfwWindowShouldClose(window)) {
    // Poll events
    glfwPollEvents();

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Render the initial state (timestep 0)
    renderer.Render(*space, 0);

    // Rendering
    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
