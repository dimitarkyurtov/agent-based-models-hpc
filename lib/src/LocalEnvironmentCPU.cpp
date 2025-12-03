#include "ParallelABM/LocalEnvironmentCPU.h"

#include <cstdint>
#include <thread>

namespace ParallelABM {

LocalEnvironmentCPU::LocalEnvironmentCPU() {
  // Query hardware concurrency at construction time
  const unsigned int kConcurrency = std::thread::hardware_concurrency();

  // hardware_concurrency() returns 0 if the value is not computable
  // In this case, default to 1 core as a safe fallback
  cpuCores_ = (kConcurrency > 0) ? static_cast<std::uint32_t>(kConcurrency) : 1;
}

LocalEnvironmentCPU::LocalEnvironmentCPU(std::uint32_t cpu_cores)
    : cpuCores_(cpu_cores > 0 ? cpu_cores : 1) {
  // Ensure at least 1 core is configured
}

std::uint32_t LocalEnvironmentCPU::GetNumberOfCPUCores() const {
  return cpuCores_;
}

std::uint32_t LocalEnvironmentCPU::GetNumberOfGPUs() const {
  // Local CPU environment does not support GPUs
  return 0;
}

}  // namespace ParallelABM
