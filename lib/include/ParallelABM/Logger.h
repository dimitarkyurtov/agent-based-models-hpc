#ifndef PARALLELABM_LOGGER_H
#define PARALLELABM_LOGGER_H

#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace ParallelABM {

/**
 * @brief Log severity levels
 */
enum class LogLevel : std::uint8_t {
  kDebug,    ///< Detailed debug information
  kInfo,     ///< General informational messages
  kWarning,  ///< Warning messages
  kError,    ///< Error messages
  kFatal     ///< Fatal error messages
};

/**
 * @brief Thread-safe singleton Logger for MPI-based parallel applications
 *
 * Provides formatted logging with timestamp, MPI rank, thread ID, and log
 * level. Must be initialized once at program start with the MPI rank.
 */
class Logger {
 public:
  /**
   * @brief Get the singleton instance
   * @return Reference to the Logger instance
   */
  static Logger& GetInstance();

  /**
   * @brief Initialize the logger with MPI rank
   * @param rank MPI process rank
   * @param worldSize Total number of MPI processes
   */
  void Initialize(int rank, int worldSize);

  /**
   * @brief Log a message at the specified level
   * @param level Log severity level
   * @param message Message to log
   */
  void Log(LogLevel level, const std::string& message);

  /**
   * @brief Log a debug message
   * @param message Message to log
   */
  void Debug(const std::string& message);

  /**
   * @brief Log an info message
   * @param message Message to log
   */
  void Info(const std::string& message);

  /**
   * @brief Log a warning message
   * @param message Message to log
   */
  void Warning(const std::string& message);

  /**
   * @brief Log an error message
   * @param message Message to log
   */
  void Error(const std::string& message);

  /**
   * @brief Log a fatal error message
   * @param message Message to log
   */
  void Fatal(const std::string& message);

  /**
   * @brief Set the minimum log level to display
   * @param level Minimum log level
   */
  void SetLogLevel(LogLevel level);

  /**
   * @brief Get the current MPI rank
   * @return MPI rank (-1 if not initialized)
   */
  int GetRank() const;

  // Delete copy and move constructors and assignment operators
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;
  Logger(Logger&&) = delete;
  Logger& operator=(Logger&&) = delete;

 private:
  Logger();
  ~Logger() = default;

  /**
   * @brief Convert log level to string representation
   * @param level Log level
   * @return String representation
   */
  static std::string LogLevelToString(LogLevel level);

  /**
   * @brief Get current timestamp as formatted string
   * @return Formatted timestamp
   */
  static std::string GetCurrentTimestamp();

  /**
   * @brief Get current thread ID as string
   * @return Thread ID
   */
  static std::string GetThreadId();

  /**
   * @brief Format and output the log message
   * @param level Log level
   * @param message Message content
   */
  void WriteLog(LogLevel level, const std::string& message);

  int mpiRank_{-1};                     ///< MPI process rank
  int mpiWorldSize_{0};                 ///< Total number of MPI processes
  LogLevel minLevel_{LogLevel::kInfo};  ///< Minimum log level to display
  bool initialized_{false};             ///< Initialization flag
  mutable std::mutex mutex_;            ///< Mutex for thread-safe logging
};

}  // namespace ParallelABM

#endif  // PARALLELABM_LOGGER_H
