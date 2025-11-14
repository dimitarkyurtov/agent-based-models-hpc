#ifndef PARALLELABM_MODELCUDA_H
#define PARALLELABM_MODELCUDA_H

#include "Agent.h"
#include "Model.h"

/**
 * @class ModelCUDA
 * @brief CUDA/GPU implementation of the Model interface for parallel agent
 * interactions.
 *
 * ModelCUDA provides a GPU-based implementation of the computational model
 * where agent interactions are defined through a CUDA device function pointer.
 * This implementation is designed for execution on NVIDIA GPUs using CUDA.
 *
 * ARCHITECTURE:
 * =============
 * This class stores a single member variable - a function pointer to a CUDA
 * device function that defines the interaction rule.
 *
 * INTERACTION RULE FUNCTION:
 * ==========================
 * The interaction rule is a user-provided CUDA device function with the
 * signature:
 *     __device__ void interactionRule(Agent* agents, int size, Agent*
 * neighbors, int neighborSize)
 *
 * Or more commonly, a global kernel:
 *     __global__ void interactionRule(Agent* agents, int size, Agent*
 * neighbors, int neighborSize)
 *
 * Parameters:
 * - agents: Pointer to device memory containing the local agent array
 * - size: Number of agents in the local array
 * - neighbors: Pointer to device memory containing the neighbor agent array
 * - neighborSize: Number of neighbors in the array
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * // Define a CUDA kernel for a simple physics simulation
 * __global__ void myPhysicsKernel(Agent* agents, int size, Agent* neighbors,
 * int neighborSize) {
 *     // Calculate global thread index
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     // Ensure we don't access beyond the array
 *     if (idx >= size) return;
 *
 *     // Cast to specific agent type if needed
 *     MyGPUAgent* agent = &((MyGPUAgent*)agents)[idx];
 *
 *     const float dt = 0.01f;  // Time step
 *     const float damping = 0.99f;
 *
 *     // Update this agent's position based on velocity
 *     agent->x += agent->vx * dt;
 *     agent->y += agent->vy * dt;
 *
 *     // Apply damping
 *     agent->vx *= damping;
 *     agent->vy *= damping;
 *
 *     // Interact with agents in the local region
 *     for (int j = 0; j < size; ++j) {
 *         if (idx != j) {
 *             MyGPUAgent* other = &((MyGPUAgent*)agents)[j];
 *
 *             // Calculate distance
 *             float dx = other->x - agent->x;
 *             float dy = other->y - agent->y;
 *             float distSq = dx*dx + dy*dy;
 *
 *             // Apply forces based on distance
 *             if (distSq < 1.0f && distSq > 0.0001f) {
 *                 float force = 1.0f / distSq;
 *                 agent->vx -= dx * force;
 *                 agent->vy -= dy * force;
 *             }
 *         }
 *     }
 *
 *     // Interact with neighbor agents from other regions
 *     for (int j = 0; j < neighborSize; ++j) {
 *         MyGPUAgent* neighbor = &((MyGPUAgent*)neighbors)[j];
 *
 *         // Calculate distance to neighbor
 *         float dx = neighbor->x - agent->x;
 *         float dy = neighbor->y - agent->y;
 *         float distSq = dx*dx + dy*dy;
 *
 *         // Apply forces based on distance
 *         if (distSq < 1.0f && distSq > 0.0001f) {
 *             float force = 1.0f / distSq;
 *             agent->vx -= dx * force;
 *             agent->vy -= dy * force;
 *         }
 *     }
 * }
 *
 * @see Model Base class defining the interface
 * @see ModelCPU CPU implementation for multi-core processors
 */
class ModelCUDA : public Model {
 public:
  /**
   * @brief Function pointer type for CUDA interaction rules.
   *
   * This defines the signature of the interaction rule function for GPU
   * execution. The function should be a CUDA kernel (__global__) or device
   * function (__device__) that operates on agents stored in device memory.
   *
   * Signature:
   * - Parameter 1: Pointer to device memory containing local agent array
   * - Parameter 2: Number of local agents in the array
   * - Parameter 3: Pointer to device memory containing neighbor agent array
   * - Parameter 4: Number of neighbor agents in the array
   * - Return: void (modifies agents in place in device memory)
   *
   * The function is executed on the GPU with many parallel threads,
   * each processing one or more agents simultaneously.
   *
   */
  using InteractionRuleCUDA = void (*)(Agent*, int, Agent*, int);

  /**
   * @brief The CUDA interaction rule function pointer.
   */
  InteractionRuleCUDA interactionRule;

  /**
   * @brief Constructs a ModelCUDA with the specified interaction rule.
   */
  explicit ModelCUDA(InteractionRuleCUDA rule) : interactionRule(rule) {}

  ModelCUDA() = delete;

  /**
   * @brief Copy constructor.
   */
  ModelCUDA(const ModelCUDA&) = default;

  /**
   * @brief Copy assignment operator.
   */
  ModelCUDA& operator=(const ModelCUDA&) = default;

  /**
   * @brief Move constructor.
   */
  ModelCUDA(ModelCUDA&&) = default;

  /**
   * @brief Move assignment operator.
   */
  ModelCUDA& operator=(ModelCUDA&&) = default;

  /**
   * @brief Virtual destructor for proper cleanup.
   */
  ~ModelCUDA() override = default;
};

#endif  // PARALLELABM_MODELCUDA_H
