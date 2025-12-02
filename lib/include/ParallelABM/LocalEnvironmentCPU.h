#ifndef PARALLELABM_LOCAL_ENVIRONMENT_CPU_H
#define PARALLELABM_LOCAL_ENVIRONMENT_CPU_H

#include <cstdint>

#include "Environment.h"

namespace ParallelABM {

/**
 * @brief Local CPU-only implementation of Environment interface
 *
 * Provides environment information for local execution outside of a
 * SLURM cluster. Uses std::thread::hardware_concurrency() to determine
 * the number of available CPU cores. Reports zero GPUs as this class
 * is intended for CPU-only execution.
 */
class LocalEnvironmentCPU : public Environment {
 public:
  LocalEnvironmentCPU();

  /**
   * @brief Get CPU cores using std::thread::hardware_concurrency()
   * @return Number of CPU cores available on the local machine
   */
  [[nodiscard]] std::uint32_t GetNumberOfCPUCores() const override;

  /**
   * @brief Returns 0 as this environment does not support GPUs
   * @return Always returns 0
   */
  [[nodiscard]] std::uint32_t GetNumberOfGPUs() const override;

  ~LocalEnvironmentCPU() override = default;

  // Delete copy and move operations
  LocalEnvironmentCPU(const LocalEnvironmentCPU&) = delete;
  LocalEnvironmentCPU& operator=(const LocalEnvironmentCPU&) = delete;
  LocalEnvironmentCPU(LocalEnvironmentCPU&&) = delete;
  LocalEnvironmentCPU& operator=(LocalEnvironmentCPU&&) = delete;

 private:
  std::uint32_t cpuCores_{0};  ///< Cached CPU core count
};

}  // namespace ParallelABM

#endif  // PARALLELABM_LOCAL_ENVIRONMENT_CPU_H
