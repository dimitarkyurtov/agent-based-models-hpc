#ifndef PARALLELABM_MODELCUDA_H
#define PARALLELABM_MODELCUDA_H

#include "Model.h"

/**
 * @class ModelCUDA
 * @brief CUDA/GPU implementation of the Model interface for parallel agent
 * interactions.
 *
 * ModelCUDA provides a GPU-based implementation of the computational model
 * where agent interactions are defined through a virtual method that returns
 * a CUDA kernel function pointer. This implementation is designed for execution
 * on NVIDIA GPUs using CUDA.
 *
 * @tparam AgentType The user-defined agent type used in the simulation.
 *                   Must be a POD-like type suitable for GPU memory transfer.
 *
 * ARCHITECTURE:
 * =============
 * This is an abstract base class that defines the interface for CUDA-based
 * models. Derived classes must implement the GetInteractionKernel virtual
 * method to return their specific CUDA kernel function pointer.
 *
 * INTERACTION RULE KERNEL:
 * ========================
 * Derived classes must override GetInteractionKernel to return a CUDA kernel
 * function pointer with the signature:
 *     __global__ void interactionRule(AgentType* agents, int size,
 *                                      AgentType* neighbors, int neighborSize)
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
 * struct MyAgent {
 *     float x, y;
 *     float vx, vy;
 * };
 *
 * __global__ void myPhysicsKernel(MyAgent* agents, int size,
 *                                  MyAgent* neighbors, int neighborSize) {
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (idx >= size) return;
 *
 *     MyAgent& agent = agents[idx];
 *     const float dt = 0.01f;
 *     agent.x += agent.vx * dt;
 *     agent.y += agent.vy * dt;
 * }
 *
 * class MyPhysicsModelCUDA : public ModelCUDA<MyAgent> {
 *  public:
 *   MyPhysicsModelCUDA() = default;
 *
 *   InteractionRuleCUDA GetInteractionKernel() const override {
 *     return myPhysicsKernel;
 *   }
 * };
 *
 * @see Model Base class defining the interface
 * @see ModelCPU CPU implementation for multi-core processors
 */
template <typename AgentType>
class ModelCUDA : public Model {
 public:
  /**
   * @brief Function pointer type for CUDA interaction rule kernels.
   *
   * This defines the signature of the interaction rule kernel for GPU
   * execution. The function should be a CUDA __global__ kernel that operates
   * on agents stored in device memory.
   *
   * Signature:
   * - Parameter 1: Pointer to device memory containing local agent array
   * - Parameter 2: Number of local agents in the array
   * - Parameter 3: Pointer to device memory containing neighbor agent array
   * - Parameter 4: Number of neighbor agents in the array
   * - Return: void (modifies agents in place in device memory)
   *
   * The kernel is executed on the GPU with many parallel threads,
   * each processing one or more agents simultaneously.
   */
  using InteractionRuleCUDA = void (*)(AgentType*, int, AgentType*, int);

  /**
   * @brief Default constructor.
   */
  ModelCUDA() = default;

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

  /**
   * @brief Get the CUDA kernel function pointer for agent interactions.
   *
   * This pure virtual method must be implemented by derived classes to return
   * the CUDA kernel that implements the interaction logic. Called by
   * SimulationCUDA when launching the kernel for each GPU subregion.
   *
   * @return Function pointer to the CUDA __global__ kernel
   */
  virtual InteractionRuleCUDA GetInteractionKernel() const = 0;
};

#endif  // PARALLELABM_MODELCUDA_H
