#include "ParallelABM/EnvironmentSlurm.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

namespace ParallelABM {

EnvironmentSlurm::EnvironmentSlurm() {
  // Query Slurm environment variables at construction time
  auto cpu_cores = GetEnvAsUInt("SLURM_CPUS_ON_NODE");
  cpuCores_ = cpu_cores.value_or(0);

  auto gpu_count = GetEnvAsUInt("SLURM_GPUS_ON_NODE");
  gpuCount_ = gpu_count.value_or(0);
}

std::uint32_t EnvironmentSlurm::GetNumberOfCPUCores() const {
  return cpuCores_;
}

std::uint32_t EnvironmentSlurm::GetNumberOfGPUs() const { return gpuCount_; }

std::optional<std::uint32_t> EnvironmentSlurm::GetEnvAsUInt(
    const std::string& var_name) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  const char* env_value = std::getenv(var_name.c_str());

  if (env_value == nullptr) {
    return std::nullopt;
  }

  try {
    char* end_ptr = nullptr;  // NOLINT(misc-const-correctness)
    std::uint64_t parsed_value = std::strtoul(env_value, &end_ptr, 10);

    if (end_ptr == env_value || *end_ptr != '\0') {
      return std::nullopt;
    }

    if (parsed_value > UINT32_MAX) {
      return std::nullopt;
    }

    return static_cast<std::uint32_t>(parsed_value);
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace ParallelABM
