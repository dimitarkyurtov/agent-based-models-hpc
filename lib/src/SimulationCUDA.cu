#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "ParallelABM/Agent.h"
#include "ParallelABM/LocalRegion.h"
#include "ParallelABM/SimulationCUDA.h"

namespace ParallelABM {

/**
 * @brief Check CUDA API call result and throw on error.
 * @param result CUDA API return code
 * @param operation Description of the operation for error message
 */
static void CheckCudaError(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + operation + ": " +
                             cudaGetErrorString(result));
  }
}

/**
 * @brief GPU execution context holding device memory and stream for one
 * subregion.
 */
struct GPUContext {
  int device_id = 0;
  Agent* d_agents = nullptr;
  Agent* d_neighbors = nullptr;
  int num_agents = 0;
  int num_neighbors = 0;
  std::vector<int> indices{};
  cudaStream_t stream{};
};

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void SimulationCUDA::LaunchModel(LocalRegion* local_region) {
  const auto kNumGpus = environment.GetNumberOfGPUs();

  // Split local region into subregions (one per GPU)
  std::vector<LocalSubRegion> subregions =
      local_region->SplitIntoSubRegions(static_cast<int>(kNumGpus));

  std::vector<Agent>& all_agents = local_region->GetAgents();
  const auto& neighbors = local_region->GetNeighbors();
  const int kNumNeighbors = static_cast<int>(neighbors.size());

  // Prepare GPU contexts and allocate memory for each device
  std::vector<GPUContext> contexts;
  contexts.reserve(subregions.size());
  std::vector<std::vector<Agent>> host_agent_buffers(subregions.size());

  for (size_t i = 0; i < subregions.size(); ++i) {
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
      host_agent_buffers[i].push_back(all_agents[kIdx]);
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
    const size_t kAgentsSize = static_cast<size_t>(kNumAgents) * sizeof(Agent);
    CheckCudaError(cudaMalloc(&ctx.d_agents, kAgentsSize), "cudaMalloc agents");

    // Allocate device memory for neighbors
    if (kNumNeighbors > 0) {
      const size_t kNeighborsSize =
          static_cast<size_t>(kNumNeighbors) * sizeof(Agent);
      CheckCudaError(cudaMalloc(&ctx.d_neighbors, kNeighborsSize),
                     "cudaMalloc neighbors");
    }

    contexts.push_back(ctx);
  }

  // Copy data to all devices asynchronously
  for (size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");

    const size_t kAgentsSize =
        static_cast<size_t>(ctx.num_agents) * sizeof(Agent);
    CheckCudaError(
        cudaMemcpyAsync(ctx.d_agents, host_agent_buffers[i].data(), kAgentsSize,
                        cudaMemcpyHostToDevice, ctx.stream),
        "cudaMemcpyAsync agents");

    if (ctx.num_neighbors > 0) {
      const size_t kNeighborsSize =
          static_cast<size_t>(ctx.num_neighbors) * sizeof(Agent);
      CheckCudaError(
          cudaMemcpyAsync(ctx.d_neighbors, neighbors.data(), kNeighborsSize,
                          cudaMemcpyHostToDevice, ctx.stream),
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

    // Launch the interaction rule kernel
    model.interactionRule<<<kNumBlocks, kThreadsPerBlock, 0, ctx.stream>>>(
        ctx.d_agents, ctx.num_agents, ctx.d_neighbors, ctx.num_neighbors);

    CheckCudaError(cudaGetLastError(), "kernel launch");
  }

  // Copy results back from all devices asynchronously
  for (size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");

    const size_t kAgentsSize =
        static_cast<size_t>(ctx.num_agents) * sizeof(Agent);
    CheckCudaError(
        cudaMemcpyAsync(host_agent_buffers[i].data(), ctx.d_agents, kAgentsSize,
                        cudaMemcpyDeviceToHost, ctx.stream),
        "cudaMemcpyAsync agents back");
  }

  // Synchronize all streams and cleanup
  for (size_t i = 0; i < contexts.size(); ++i) {
    auto& ctx = contexts[i];
    CheckCudaError(cudaSetDevice(ctx.device_id), "cudaSetDevice");
    CheckCudaError(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize");

    // Write updated agents back to the parent region
    for (size_t j = 0; j < ctx.indices.size(); ++j) {
      all_agents[ctx.indices[j]] = host_agent_buffers[i][j];
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
