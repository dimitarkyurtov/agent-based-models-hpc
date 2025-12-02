#ifndef PARALLELABM_SIMULATIONCUDA_H
#define PARALLELABM_SIMULATIONCUDA_H

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "LocalRegion.h"
#include "ModelCUDA.h"
#include "Simulation.h"

namespace ParallelABM {

/**
 * @brief Check CUDA API call result and throw on error.
 * @param result CUDA API return code
 * @param operation Description of the operation for error message
 */
inline void CheckCudaError(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + operation + ": " +
                             cudaGetErrorString(result));
  }
}

/**
 * @class SimulationCUDA
 * @brief CUDA/GPU-specific simulation implementation using NVIDIA GPUs.
 *
 * SimulationCUDA executes agent-based simulations on NVIDIA GPUs by
 * distributing work across multiple GPU devices. Each GPU processes a subregion
 * of agents using the ModelCUDA's interaction rule kernel.
 *
 * @tparam AgentType The user-defined agent type used in the simulation.
 *                   Must be a POD-like type suitable for GPU memory transfer
 *                   (no virtual functions, no dynamic memory, no STL
 * containers).
 *
 * GPU EXECUTION MODEL:
 * ====================
 * - Reads GPU count from Environment::GetNumberOfGPUs()
 * - Splits the local region into subregions matching GPU count
 * - For each GPU:
 *   1. Sets the active CUDA device
 *   2. Allocates device memory for agents and neighbors
 *   3. Copies data from host to device
 *   4. Launches the interaction rule kernel with maximum thread blocks/threads
 *   5. Copies updated agents back to host memory
 *   6. Frees device memory
 * - Synchronizes all GPU operations before returning
 *
 * KERNEL CONFIGURATION:
 * =====================
 * The kernel is launched with optimal block and thread counts based on
 * device properties (queried via cudaGetDeviceProperties). The configuration
 * maximizes occupancy by using the maximum threads per block supported by
 * the device and calculating the required number of blocks to cover all agents.
 *
 * USAGE EXAMPLE:
 * ==============
 *
 * struct MyAgent {
 *     float x, y;
 *     float vx, vy;
 * };
 *
 * __global__ void myKernel(MyAgent* agents, int size,
 *                          MyAgent* neighbors, int neighborSize) {
 *     // ... kernel implementation
 * }
 *
 * ModelCUDA<MyAgent> model(myKernel);
 * SimulationCUDA<MyAgent> sim(argc, argv, space, model, env, splitFn);
 * sim.Start(100);
 *
 * @see Simulation Base class providing MPI coordination
 * @see ModelCUDA CUDA interaction rule implementation
 */
template <typename AgentType>
class SimulationCUDA : public Simulation<ModelCUDA<AgentType>> {
 public:
  // Inherit base class constructors
  using Simulation<ModelCUDA<AgentType>>::Simulation;

  // Copy constructor - deleted (inherited from base class semantics)
  SimulationCUDA(const SimulationCUDA&) = delete;

  // Copy assignment - deleted
  SimulationCUDA& operator=(const SimulationCUDA&) = delete;

  // Move constructor - deleted (inherited from base class semantics)
  SimulationCUDA(SimulationCUDA&&) = delete;

  // Move assignment - deleted
  SimulationCUDA& operator=(SimulationCUDA&&) = delete;

  /**
   * @brief Execute model computation across available NVIDIA GPUs.
   *
   * Splits the local region into subregions based on available GPUs from the
   * environment. For each subregion, launches the model's interaction rule
   * kernel on the corresponding GPU with maximum available thread blocks
   * and threads.
   *
   * @param local_region Pointer to the local region containing agents
   */
  void LaunchModel(LocalRegion* local_region) override;

  ~SimulationCUDA() override = default;

 private:
  /**
   * @brief GPU execution context holding device memory and stream for one
   * subregion.
   */
  struct GPUContext {
    int device_id = 0;
    AgentType* d_agents = nullptr;
    AgentType* d_neighbors = nullptr;
    int num_agents = 0;
    int num_neighbors = 0;
    std::vector<int> indices{};
    cudaStream_t stream{};
  };
};

template <typename AgentType>
void SimulationCUDA<AgentType>::LaunchModel(LocalRegion* local_region) {
  const auto kNumGpus = this->environment.GetNumberOfGPUs();

  // Split local region into subregions (one per GPU)
  std::vector<LocalSubRegion> subregions =
      this->space_->SplitLocalRegion(*local_region, static_cast<int>(kNumGpus));

  std::vector<Agent>& all_agents = local_region->GetAgents();
  const auto& neighbors = local_region->GetNeighbors();
  const int kNumNeighbors = static_cast<int>(neighbors.size());

  // Prepare GPU contexts and allocate memory for each device
  std::vector<GPUContext> contexts;
  contexts.reserve(subregions.size());
  std::vector<std::vector<AgentType>> host_agent_buffers(subregions.size());
  std::vector<AgentType> host_neighbors;

  // Convert neighbors to AgentType
  host_neighbors.reserve(neighbors.size());
  for (const auto& neighbor : neighbors) {
    host_neighbors.push_back(static_cast<const AgentType&>(neighbor));
  }

  for (std::size_t i = 0; i < subregions.size(); ++i) {
    const int kDeviceId = static_cast<int>(i);
    CheckCudaError(cudaSetDevice(kDeviceId), "cudaSetDevice");

    const auto& indices = subregions[i].GetIndices();
    const int kNumAgents = static_cast<int>(indices.size());

    if (kNumAgents == 0) {
      continue;
    }

    // Build contiguous agent array for this subregion
    host_agent_buffers[i].reserve(indices.size());
    for (const int kIdx : indices) {
      host_agent_buffers[i].push_back(
          static_cast<const AgentType&>(all_agents[kIdx]));
    }

    GPUContext ctx{};
    ctx.device_id = kDeviceId;
    ctx.num_agents = kNumAgents;
    ctx.num_neighbors = kNumNeighbors;
    ctx.indices = indices;
    ctx.d_agents = nullptr;
    ctx.d_neighbors = nullptr;

    // Create stream for this device
    CheckCudaError(cudaStreamCreate(&ctx.stream), "cudaStreamCreate");

    // Allocate device memory for agents
    const std::size_t kAgentsSize =
        static_cast<std::size_t>(kNumAgents) * sizeof(AgentType);
    CheckCudaError(cudaMalloc(&ctx.d_agents, kAgentsSize), "cudaMalloc agents");

    // Allocate device memory for neighbors
    if (kNumNeighbors > 0) {
      const std::size_t kNeighborsSize =
          static_cast<std::size_t>(kNumNeighbors) * sizeof(AgentType);
      CheckCudaError(cudaMalloc(&ctx.d_neighbors, kNeighborsSize),
                     "cudaMalloc neighbors");
    }

    contexts.push_back(ctx);
  }

  // Copy data to all devices asynchronously
  for (std::size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");

    const std::size_t kAgentsSize =
        static_cast<std::size_t>(ctx.num_agents) * sizeof(AgentType);
    CheckCudaError(
        cudaMemcpyAsync(ctx.d_agents, host_agent_buffers[i].data(), kAgentsSize,
                        cudaMemcpyHostToDevice, ctx.stream),
        "cudaMemcpyAsync agents");

    if (ctx.num_neighbors > 0) {
      const std::size_t kNeighborsSize =
          static_cast<std::size_t>(ctx.num_neighbors) * sizeof(AgentType);
      CheckCudaError(
          cudaMemcpyAsync(ctx.d_neighbors, host_neighbors.data(),
                          kNeighborsSize, cudaMemcpyHostToDevice, ctx.stream),
          "cudaMemcpyAsync neighbors");
    }
  }

  // Launch kernels on all devices
  for (auto& ctx : contexts) {
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");

    // Get device properties for optimal kernel configuration
    cudaDeviceProp device_props;
    CheckCudaError(cudaGetDeviceProperties(&device_props, ctx.device_id),
                   "cudaGetDeviceProperties");

    // Configure kernel launch parameters for maximum occupancy
    const int kMaxThreadsPerBlock = device_props.maxThreadsPerBlock;
    const int kThreadsPerBlock = (ctx.num_agents < kMaxThreadsPerBlock)
                                     ? ctx.num_agents
                                     : kMaxThreadsPerBlock;
    const int kNumBlocks =
        (ctx.num_agents + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Get the interaction kernel and launch it
    auto kernel = this->model.GetInteractionKernel();
    kernel<<<kNumBlocks, kThreadsPerBlock, 0, ctx.stream>>>(
        ctx.d_agents, ctx.num_agents, ctx.d_neighbors, ctx.num_neighbors);

    CheckCudaError(cudaGetLastError(), "kernel launch");
  }

  // Copy results back from all devices asynchronously
  for (std::size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");

    const std::size_t kAgentsSize =
        static_cast<std::size_t>(ctx.num_agents) * sizeof(AgentType);
    CheckCudaError(
        cudaMemcpyAsync(host_agent_buffers[i].data(), ctx.d_agents, kAgentsSize,
                        cudaMemcpyDeviceToHost, ctx.stream),
        "cudaMemcpyAsync agents back");
  }

  // Synchronize all streams and cleanup
  for (std::size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");
    CheckCudaError(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize");

    // Write updated agents back to the parent region
    for (std::size_t j = 0; j < ctx.indices.size(); ++j) {
      static_cast<AgentType&>(all_agents[ctx.indices[j]]) =
          host_agent_buffers[i][j];
    }

    // Cleanup device resources
    CheckCudaError(cudaFree(ctx.d_agents), "cudaFree agents");
    if (ctx.d_neighbors != nullptr) {
      CheckCudaError(cudaFree(ctx.d_neighbors), "cudaFree neighbors");
    }
    CheckCudaError(cudaStreamDestroy(ctx.stream), "cudaStreamDestroy");
  }
}

}  // namespace ParallelABM

#endif  // PARALLELABM_SIMULATIONCUDA_H
