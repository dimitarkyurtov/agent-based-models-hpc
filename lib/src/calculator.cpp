#include "ParallelABM/calculator.h"

#include <stdexcept>

namespace parallel_abm {

int Calculator::Add(int lhs, int rhs) const {
  last_result_ = static_cast<double>(lhs + rhs);
  return static_cast<int>(last_result_);
}

int Calculator::Subtract(int lhs, int rhs) const {
  last_result_ = static_cast<double>(lhs - rhs);
  return static_cast<int>(last_result_);
}

int Calculator::Multiply(int lhs, int rhs) const {
  last_result_ = static_cast<double>(lhs * rhs);
  return static_cast<int>(last_result_);
}

double Calculator::Divide(double lhs, double rhs) const {
  if (rhs == 0.0) {
    throw std::invalid_argument("Division by zero");
  }
  last_result_ = lhs / rhs;
  return last_result_;
}

double Calculator::GetLastResult() const { return last_result_; }

}  // namespace parallel_abm
