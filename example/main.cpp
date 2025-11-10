#include <iostream>
#include <stdexcept>

#include "mylib/calculator.h"

int main() {
  const mylib::Calculator kCalc;

  std::cout << "Calculator Example Application\n";
  std::cout << "==============================\n\n";

  // Demonstrate addition
  const int kSum = kCalc.Add(10, 5);
  std::cout << "10 + 5 = " << kSum << "\n";
  std::cout << "Last result: " << kCalc.GetLastResult() << "\n\n";

  // Demonstrate subtraction
  const int kDifference = kCalc.Subtract(20, 8);
  std::cout << "20 - 8 = " << kDifference << "\n";
  std::cout << "Last result: " << kCalc.GetLastResult() << "\n\n";

  // Demonstrate multiplication
  const int kProduct = kCalc.Multiply(7, 6);
  std::cout << "7 * 6 = " << kProduct << "\n";
  std::cout << "Last result: " << kCalc.GetLastResult() << "\n\n";

  // Demonstrate division
  try {
    const double kQuotient = kCalc.Divide(15.0, 3.0);
    std::cout << "15.0 / 3.0 = " << kQuotient << "\n";
    std::cout << "Last result: " << kCalc.GetLastResult() << "\n\n";

    // Test division by zero handling
    std::cout << "Attempting division by zero...\n";
    kCalc.Divide(10.0, 0.0);
  } catch (const std::invalid_argument& e) {
    std::cout << "Caught exception: " << e.what() << "\n";
  }

  return 0;
}
