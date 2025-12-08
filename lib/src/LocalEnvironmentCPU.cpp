#include <cstdint>
#include <thread>

#include "ParallelABM/LocalEnvironment.h"

namespace ParallelABM {

LocalEnvironment::LocalEnvironment() {
  // Query hardware concurrency at construction time
  const unsigned int kConcurrency = std::thread::hardware_concurrency();

  // hardware_concurrency() returns 0 if the value is not computable
  // In this case, default to 1 core as a safe fallback
  cpuCores_ = (kConcurrency > 0) ? static_cast<std::uint32_t>(kConcurrency) : 1;

  // Default to 0 GPUs
  gpuCount_ = 0;
}

LocalEnvironment::LocalEnvironment(std::uint32_t cpu_cores)
    : cpuCores_(cpu_cores > 0 ? cpu_cores : 1) {
  // Ensure at least 1 core is configured, no GPUs by default
}

LocalEnvironment::LocalEnvironment(std::uint32_t cpu_cores,
                                   std::uint32_t gpu_count)
    : cpuCores_(cpu_cores > 0 ? cpu_cores : 1), gpuCount_(gpu_count) {
  // Use provided values, ensuring at least 1 CPU core
}

std::uint32_t LocalEnvironment::GetNumberOfCPUCores() const {
  return cpuCores_;
}

std::uint32_t LocalEnvironment::GetNumberOfGPUs() const { return gpuCount_; }

}  // namespace ParallelABM
