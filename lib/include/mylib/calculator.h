#ifndef MYLIB_CALCULATOR_H
#define MYLIB_CALCULATOR_H

namespace mylib {

class Calculator {
 public:
  Calculator() = default;
  ~Calculator() = default;
  Calculator(const Calculator&) = default;
  Calculator& operator=(const Calculator&) = default;
  Calculator(Calculator&&) = default;
  Calculator& operator=(Calculator&&) = default;

  // Basic arithmetic operations
  int Add(int lhs, int rhs) const;
  int Subtract(int lhs, int rhs) const;
  int Multiply(int lhs, int rhs) const;
  double Divide(double lhs, double rhs) const;

  // Returns the last result
  double GetLastResult() const;

 private:
  mutable double last_result_ = 0.0;
};

}  // namespace mylib

#endif  // MYLIB_CALCULATOR_H
