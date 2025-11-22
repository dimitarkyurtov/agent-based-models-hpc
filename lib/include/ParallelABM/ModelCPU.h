#ifndef PARALLELABM_MODELCPU_H
#define PARALLELABM_MODELCPU_H

#include <functional>
#include <vector>

#include "Agent.h"
#include "Model.h"

/**
 * @class ModelCPU
 * @brief CPU implementation of the Model interface for multi-threaded agent
 * interactions.
 *
 * ModelCPU provides a CPU-based implementation of the computational model where
 * agent interactions are defined through a std::function. This implementation
 * is designed for execution on CPU cores using threading for parallelism.
 *
 * ARCHITECTURE:
 * =============
 * This class stores a single member variable - a function that defines the
 * interaction rule. When the library needs to process a region of agents on
 * the CPU, it invokes this interaction rule function.
 *
 * INTERACTION RULE FUNCTION:
 * ==========================
 * The interaction rule is a user-provided function with the signature:
 *     void interactionRule(std::vector<std::reference_wrapper<Agent>>& agents,
 *                          const std::vector<Agent>& neighbors)
 *
 * Parameters:
 * - agents: Vector of references to agents to process (mutable)
 * - neighbors: Vector of neighbor agents (read-only)
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * // Define an interaction rule for a simple physics simulation
 * auto myPhysicsRule = [](std::vector<std::reference_wrapper<Agent>>& agents,
 *                         const std::vector<Agent>& neighbors) {
 *     const double dt = 0.01;
 *     const double damping = 0.99;
 *
 *     for (Agent& agent : agents) {
 *         // Cast to your specific agent type if needed
 *         auto& myAgent = static_cast<MyCustomAgent&>(agent);
 *
 *         // Update position based on velocity
 *         myAgent.x += myAgent.vx * dt;
 *         myAgent.y += myAgent.vy * dt;
 *
 *         // Apply damping
 *         myAgent.vx *= damping;
 *         myAgent.vy *= damping;
 *
 *         // Interact with other agents in the subregion
 *         for (Agent& other : agents) {
 *             if (&agent != &other.get()) {
 *                 // Calculate forces, collisions, etc.
 *             }
 *         }
 *
 *         // Interact with neighbor agents from other regions
 *         for (const Agent& neighbor : neighbors) {
 *             // Calculate forces, collisions with neighbors, etc.
 *         }
 *     }
 * };
 *
 * @see Model Base class defining the interface
 * @see ModelCUDA GPU implementation for CUDA devices
 */
class ModelCPU : public Model {
 public:
  /**
   * @brief Function type for CPU interaction rules.
   *
   * This defines the signature of the interaction rule function:
   * - Parameter 1: Vector of references to agents to process
   * - Parameter 2: Const reference to vector of neighbor agents
   * - Return: void (modifies agents in place via references)
   *
   * The function does not need to be thread-safe as each subregion will be
   * processed independently by a single thread.
   */
  using InteractionRuleCPU = std::function<void(
      std::vector<std::reference_wrapper<Agent>>&, const std::vector<Agent>&)>;

  /**
   * @brief The CPU interaction rule function.
   *
   * This member variable stores the user-provided function that defines
   * how agents interact with each other in the simulation.
   */
  InteractionRuleCPU interaction_rule_;

  /**
   * @brief Constructs a ModelCPU with the specified interaction rule.
   *
   * @param rule Function defining agent interactions. Must be callable.
   *
   * Example:
   *     auto myRule = [](std::vector<std::reference_wrapper<Agent>>& agents,
   *                      const std::vector<Agent>& neighbors) { ... };
   *     ModelCPU model(myRule);
   */
  explicit ModelCPU(InteractionRuleCPU rule)
      : interaction_rule_(std::move(rule)) {}

  /**
   * @brief Default constructor - deleted.
   *
   * A ModelCPU must be constructed with an interaction rule.
   */
  ModelCPU() = delete;

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
};

#endif  // PARALLELABM_MODELCPU_H
