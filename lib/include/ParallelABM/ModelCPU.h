#ifndef PARALLELABM_MODELCPU_H
#define PARALLELABM_MODELCPU_H

#include "Agent.h"
#include "Model.h"

/**
 * @class ModelCPU
 * @brief CPU implementation of the Model interface for multi-threaded agent
 * interactions.
 *
 * ModelCPU provides a CPU-based implementation of the computational model where
 * agent interactions are defined through a standard C++ function pointer. This
 * implementation is designed for execution on CPU cores using threading for
 * parallelism.
 *
 * ARCHITECTURE:
 * =============
 * This class stores a single member variable - a function pointer that defines
 * the interaction rule. When the library needs to process a region of agents on
 * the CPU, it calls the Execute() method, which invokes this interaction rule
 * function.
 *
 * INTERACTION RULE FUNCTION:
 * ==========================
 * The interaction rule is a user-provided function with the signature:
 *     void interactionRule(Agent* agents, int size, Agent* neighbors, int
 * neighborSize)
 *
 * Parameters:
 * - agents: Pointer to an array of agents to process
 * - size: Number of agents in the array
 * - neighbors: Pointer to an array of neighbor agents
 * - neighborSize: Number of neighbors in the array
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * // Define an interaction rule for a simple physics simulation
 * void myPhysicsRule(Agent* agents, int size, Agent* neighbors, int
 * neighborSize) { const double dt = 0.01;  // Time step const double damping =
 * 0.99;
 *
 *     // Update each agent based on simple physics
 *     for (int i = 0; i < size; ++i) {
 *         // Cast to your specific agent type if needed
 *         MyCustomAgent* agent = static_cast<MyCustomAgent*>(&agents[i]);
 *
 *         // Update position based on velocity
 *         agent->x += agent->vx * dt;
 *         agent->y += agent->vy * dt;
 *
 *         // Apply damping
 *         agent->vx *= damping;
 *         agent->vy *= damping;
 *
 *         // Interact with other agents in the local region
 *         for (int j = 0; j < size; ++j) {
 *             if (i != j) {
 *                 MyCustomAgent* other =
 * static_cast<MyCustomAgent*>(&agents[j]);
 *                 // Calculate forces, collisions, etc.
 *             }
 *         }
 *
 *         // Interact with neighbor agents from other regions
 *         for (int j = 0; j < neighborSize; ++j) {
 *             MyCustomAgent* neighbor =
 * static_cast<MyCustomAgent*>(&neighbors[j]);
 *             // Calculate forces, collisions with neighbors, etc.
 *         }
 *     }
 * }
 *
 * @see Model Base class defining the interface
 * @see ModelCUDA GPU implementation for CUDA devices
 */
class ModelCPU : public Model {
 public:
  /**
   * @brief Function pointer type for CPU interaction rules.
   *
   * This defines the signature of the interaction rule function:
   * - Parameter 1: Pointer to array of local agents
   * - Parameter 2: Number of local agents in the array
   * - Parameter 3: Pointer to array of neighbor agents
   * - Parameter 4: Number of neighbor agents in the array
   * - Return: void (modifies agents in place)
   *
   * The function does not need to be thread-safe as the region will be
   * processed independently.
   *
   */
  using InteractionRuleCPU = void (*)(Agent*, int, Agent*, int);

  /**
   * @brief The CPU interaction rule function pointer.
   *
   * This member variable stores the user-provided function that defines
   * how agents interact with each other in the simulation.
   *
   */
  InteractionRuleCPU interactionRule;

  /**
   * @brief Constructs a ModelCPU with the specified interaction rule.
   *
   * @param rule Function pointer to the CPU interaction rule that defines
   *             agent interactions. Must not be null.
   *
   * Example:
   *     void myRule(Agent* agents, int size, Agent* neighbors, int
   * neighborSize) { ... } ModelCPU model(myRule);
   */
  explicit ModelCPU(InteractionRuleCPU rule) : interactionRule(rule) {}

  /**
   * @brief Default constructor creating a model with no interaction rule.
   *
   * When using this constructor, the interactionRule member must be set
   * manually before calling Execute().
   *
   * Example:
   *     ModelCPU model;
   *     model.interactionRule = myRule;
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
