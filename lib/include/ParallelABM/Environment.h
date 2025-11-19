#ifndef PARALLELABM_ENVIRONMENT_H
#define PARALLELABM_ENVIRONMENT_H

#include <cstdint>

namespace ParallelABM {

/**
 * @brief Abstract base class for querying compute resource allocation
 *
 * Provides interface for determining CPU and GPU resources available
 * to the current MPI process in a parallel computing environment.
 */
class Environment {
 public:
  /**
   * @brief Get the number of CPU cores allocated to this process
   * @return Number of CPU cores available
   */
  [[nodiscard]] virtual std::uint32_t GetNumberOfCPUCores() const = 0;

  /**
   * @brief Get the number of GPUs allocated to this process
   * @return Number of GPUs available
   */
  [[nodiscard]] virtual std::uint32_t GetNumberOfGPUs() const = 0;

  virtual ~Environment() = default;

  // Delete copy and move operations
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(Environment&&) = delete;
  Environment& operator=(Environment&&) = delete;

 protected:
  Environment() = default;
};

}  // namespace ParallelABM

#endif  // PARALLELABM_ENVIRONMENT_H
