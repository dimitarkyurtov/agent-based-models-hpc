#ifndef PARALLELABM_ENVIRONMENT_SLURM_H
#define PARALLELABM_ENVIRONMENT_SLURM_H

#include <cstdint>
#include <optional>
#include <string>

#include "Environment.h"

namespace ParallelABM {

/**
 * @brief Slurm-specific implementation of Environment interface
 *
 * Queries Slurm environment variables to determine per-process resource
 * allocation. Uses SLURM_CPUS_ON_NODE and SLURM_GPUS_ON_NODE which reflect
 * resources available to the current process on its assigned node.
 */
class EnvironmentSlurm : public Environment {
 public:
  EnvironmentSlurm();

  /**
   * @brief Get CPU cores from SLURM_CPUS_ON_NODE environment variable
   * @return Number of CPU cores allocated to this process (0 if not set)
   */
  [[nodiscard]] std::uint32_t GetNumberOfCPUCores() const override;

  /**
   * @brief Get GPUs from SLURM_GPUS_ON_NODE environment variable
   * @return Number of GPUs allocated to this process (0 if not set)
   */
  [[nodiscard]] std::uint32_t GetNumberOfGPUs() const override;

  ~EnvironmentSlurm() override = default;

  // Delete copy and move operations
  EnvironmentSlurm(const EnvironmentSlurm&) = delete;
  EnvironmentSlurm& operator=(const EnvironmentSlurm&) = delete;
  EnvironmentSlurm(EnvironmentSlurm&&) = delete;
  EnvironmentSlurm& operator=(EnvironmentSlurm&&) = delete;

 private:
  /**
   * @brief Read and parse an environment variable as unsigned integer
   * @param var_name Name of the environment variable
   * @return Parsed value if variable exists and is valid, std::nullopt
   * otherwise
   */
  static std::optional<std::uint32_t> GetEnvAsUInt(const std::string& var_name);

  std::uint32_t cpuCores_{0};  ///< Cached CPU core count
  std::uint32_t gpuCount_{0};  ///< Cached GPU count
};

}  // namespace ParallelABM

#endif  // PARALLELABM_ENVIRONMENT_SLURM_H
