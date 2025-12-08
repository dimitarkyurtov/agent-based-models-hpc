#ifndef PARALLELABM_LOCAL_ENVIRONMENT_CPU_H
#define PARALLELABM_LOCAL_ENVIRONMENT_CPU_H

#include <cstdint>

#include "Environment.h"

namespace ParallelABM {

/**
 * @brief Local implementation of Environment interface supporting CPU and GPU
 *
 * Provides environment information for local execution outside of a
 * SLURM cluster. Supports automatic detection of CPU cores via
 * std::thread::hardware_concurrency(). GPU count must be explicitly
 * configured via constructor parameters.
 */
class LocalEnvironment : public Environment {
 public:
  /**
   * @brief Construct with automatic CPU detection and no GPUs
   *
   * Detects available CPU cores using std::thread::hardware_concurrency()
   * and sets GPU count to 0
   */
  LocalEnvironment();

  /**
   * @brief Construct with explicit CPU core count only
   * @param cpu_cores Number of CPU cores to use (0 for auto-detection)
   */
  explicit LocalEnvironment(std::uint32_t cpu_cores);

  /**
   * @brief Construct with explicit CPU and GPU counts
   * @param cpu_cores Number of CPU cores to use (0 for auto-detection)
   * @param gpu_count Number of GPUs to use (0 for auto-detection)
   */
  LocalEnvironment(std::uint32_t cpu_cores, std::uint32_t gpu_count);

  /**
   * @brief Get CPU cores from configuration or auto-detection
   * @return Number of CPU cores available
   */
  [[nodiscard]] std::uint32_t GetNumberOfCPUCores() const override;

  /**
   * @brief Get GPU count from configuration or auto-detection
   * @return Number of CUDA-capable GPUs available
   */
  [[nodiscard]] std::uint32_t GetNumberOfGPUs() const override;

  ~LocalEnvironment() override = default;

  // Delete copy and move operations
  LocalEnvironment(const LocalEnvironment&) = delete;
  LocalEnvironment& operator=(const LocalEnvironment&) = delete;
  LocalEnvironment(LocalEnvironment&&) = delete;
  LocalEnvironment& operator=(LocalEnvironment&&) = delete;

 private:
  std::uint32_t cpuCores_{0};  ///< Cached CPU core count
  std::uint32_t gpuCount_{0};  ///< Cached GPU count
};

}  // namespace ParallelABM

#endif  // PARALLELABM_LOCAL_ENVIRONMENT_CPU_H
