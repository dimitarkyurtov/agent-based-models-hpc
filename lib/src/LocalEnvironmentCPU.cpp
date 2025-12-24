#include <cstdint>
#include <thread>

#include "ParallelABM/LocalEnvironment.h"

namespace ParallelABM {

LocalEnvironment::LocalEnvironment() {
  const unsigned int kConcurrency = std::thread::hardware_concurrency();

  cpuCores_ = (kConcurrency > 0) ? static_cast<std::uint32_t>(kConcurrency) : 1;

  gpuCount_ = 0;
}

LocalEnvironment::LocalEnvironment(std::uint32_t cpu_cores)
    : cpuCores_(cpu_cores > 0 ? cpu_cores : 1) {}

LocalEnvironment::LocalEnvironment(std::uint32_t cpu_cores,
                                   std::uint32_t gpu_count)
    : cpuCores_(cpu_cores > 0 ? cpu_cores : 1), gpuCount_(gpu_count) {}

std::uint32_t LocalEnvironment::GetNumberOfCPUCores() const {
  return cpuCores_;
}

std::uint32_t LocalEnvironment::GetNumberOfGPUs() const { return gpuCount_; }

}  // namespace ParallelABM
