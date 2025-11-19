#include "ParallelABM/Logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace ParallelABM {

Logger::Logger() = default;

Logger& Logger::GetInstance() {
  static Logger instance;
  return instance;
}

void Logger::Initialize(int rank, int worldSize) {
  const std::scoped_lock kLock(mutex_);
  mpiRank_ = rank;
  mpiWorldSize_ = worldSize;
  initialized_ = true;
}

void Logger::Log(LogLevel level, const std::string& message) {
  if (level < minLevel_) {
    return;
  }
  WriteLog(level, message);
}

void Logger::Debug(const std::string& message) {
  Log(LogLevel::kDebug, message);
}

void Logger::Info(const std::string& message) { Log(LogLevel::kInfo, message); }

void Logger::Warning(const std::string& message) {
  Log(LogLevel::kWarning, message);
}

void Logger::Error(const std::string& message) {
  Log(LogLevel::kError, message);
}

void Logger::Fatal(const std::string& message) {
  Log(LogLevel::kFatal, message);
}

void Logger::SetLogLevel(LogLevel level) {
  const std::scoped_lock kLock(mutex_);
  minLevel_ = level;
}

int Logger::GetRank() const { return mpiRank_; }

std::string Logger::LogLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG  ";
    case LogLevel::kInfo:
      return "INFO   ";
    case LogLevel::kWarning:
      return "WARNING";
    case LogLevel::kError:
      return "ERROR  ";
    case LogLevel::kFatal:
      return "FATAL  ";
    default:
      return "UNKNOWN";
  }
}

std::string Logger::GetCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;

  std::tm tm_buf{};
  localtime_r(&now_time, &tm_buf);

  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
      << std::setw(3) << now_ms.count();
  return oss.str();
}

std::string Logger::GetThreadId() {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  return oss.str();
}

void Logger::WriteLog(LogLevel level, const std::string& message) {
  const std::scoped_lock kLock(mutex_);

  std::ostringstream log_stream;

  // Format: [timestamp] [rank/worldSize] [thread_id] [level] message
  log_stream << "[" << GetCurrentTimestamp() << "] ";

  if (initialized_) {
    log_stream << "[Rank " << std::setw(3) << std::setfill('0') << mpiRank_
               << "/" << std::setw(3) << std::setfill('0') << mpiWorldSize_
               << "] ";
  } else {
    log_stream << R"([Rank ???/???] )";
  }

  log_stream << "[Thread " << GetThreadId() << "] ";
  log_stream << "[" << LogLevelToString(level) << "] ";
  log_stream << message;

  // Output to stdout or stderr based on level
  if (level >= LogLevel::kError) {
    std::cerr << log_stream.str() << '\n';
  } else {
    std::cout << log_stream.str() << '\n';
  }
}

}  // namespace ParallelABM
