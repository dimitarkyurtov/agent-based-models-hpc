#ifndef PARALLELABM_MODELCPU_H
#define PARALLELABM_MODELCPU_H

#include <functional>
#include <vector>

#include "Model.h"

/**
 * @class ModelCPU
 * @brief Template-based CPU implementation for multi-threaded agent
 * interactions.
 *
 * @tparam AgentT The concrete agent type for this model. Must be:
 *   - Copy constructible and copy assignable
 *   - Default constructible (for container operations)
 *
 * ModelCPU provides a CPU-based implementation of the computational model where
 * agent interactions are defined through a virtual method. This implementation
 * is designed for execution on CPU cores using threading for parallelism.
 *
 * ARCHITECTURE:
 * =============
 * This is an abstract base class that defines the interface for CPU-based
 * models. Derived classes must implement the ComputeInteractions virtual method
 * to define their specific interaction rules.
 *
 * INTERACTION RULE METHOD:
 * ========================
 * Derived classes must override ComputeInteractions with the signature:
 *     void ComputeInteractions(std::vector<std::reference_wrapper<AgentT>>&
 * agents, const std::vector<std::reference_wrapper<const AgentT>>& neighbors)
 *
 * Parameters:
 * - agents: Vector of references to agents to process (mutable)
 * - neighbors: Vector of references to neighbor agents (read-only)
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * struct MyCustomAgent {
 *   double x, y;
 *   double vx, vy;
 * };
 *
 * class MyPhysicsModel : public ModelCPU<MyCustomAgent> {
 *  public:
 *   MyPhysicsModel() = default;
 *
 *   void ComputeInteractions(
 *       std::vector<std::reference_wrapper<MyCustomAgent>>& agents,
 *       const std::vector<std::reference_wrapper<const MyCustomAgent>>&
 * neighbors) override { const double dt = 0.01; const double damping = 0.99;
 *
 *     for (MyCustomAgent& agent : agents) {
 *         agent.x += agent.vx * dt;
 *         agent.y += agent.vy * dt;
 *         agent.vx *= damping;
 *         agent.vy *= damping;
 *
 *         for (const MyCustomAgent& neighbor : neighbors) {
 *             // Calculate forces, collisions with neighbors, etc.
 *         }
 *     }
 *   }
 * };
 *
 * @see Model Base class defining the interface
 * @see ModelCUDA GPU implementation for CUDA devices
 */
template <typename AgentT>
class ModelCPU : public Model {
 public:
  /**
   * @brief Function type for CPU interaction rules (deprecated).
   *
   * NOTE: This typedef is kept for reference but should not be used for new
   * code. Use virtual method override instead.
   *
   * @deprecated Use ComputeInteractions virtual method override instead
   */
  using InteractionRuleCPU =
      std::function<void(std::vector<std::reference_wrapper<AgentT>>&,
                         const std::vector<AgentT>&)>;

  /**
   * @brief Default constructor.
   */
  ModelCPU() = default;

  /**
   * @brief Copy constructor.
   */
  ModelCPU(const ModelCPU&) = default;

  /**
   * @brief Copy assignment operator.
   */
  ModelCPU& operator=(const ModelCPU&) = default;

  /**
   * @brief Move constructor.
   */
  ModelCPU(ModelCPU&&) = default;

  /**
   * @brief Move assignment operator.
   */
  ModelCPU& operator=(ModelCPU&&) = default;

  /**
   * @brief Virtual destructor for proper cleanup.
   */
  ~ModelCPU() override = default;

  /**
   * @brief Compute agent interactions for a subregion.
   *
   * This pure virtual method must be implemented by derived classes to define
   * the interaction logic for the simulation. Called by SimulationCPU for each
   * subregion being processed in parallel.
   *
   * @param agents Vector of references to agents to process (mutable)
   * @param neighbors Vector of references to neighbor agents from adjacent
   * regions (read-only)
   */
  virtual void ComputeInteractions(
      std::vector<std::reference_wrapper<AgentT>>& agents,
      const std::vector<AgentT>& neighbors) = 0;
};

#endif  // PARALLELABM_MODELCPU_H
