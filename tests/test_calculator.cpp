#include <gtest/gtest.h>

#include <stdexcept>

#include "ParallelABM/calculator.h"

namespace parallel_abm {
namespace {

class CalculatorTest : public ::testing::Test {
 protected:
  Calculator calc;
};

TEST_F(CalculatorTest, AddPositiveNumbers) {
  EXPECT_EQ(calc.Add(5, 3), 8);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 8.0);
}

TEST_F(CalculatorTest, AddNegativeNumbers) {
  EXPECT_EQ(calc.Add(-5, -3), -8);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), -8.0);
}

TEST_F(CalculatorTest, AddMixedNumbers) {
  EXPECT_EQ(calc.Add(10, -5), 5);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 5.0);
}

TEST_F(CalculatorTest, SubtractPositiveNumbers) {
  EXPECT_EQ(calc.Subtract(10, 3), 7);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 7.0);
}

TEST_F(CalculatorTest, SubtractNegativeNumbers) {
  EXPECT_EQ(calc.Subtract(-5, -3), -2);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), -2.0);
}

TEST_F(CalculatorTest, MultiplyPositiveNumbers) {
  EXPECT_EQ(calc.Multiply(4, 5), 20);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 20.0);
}

TEST_F(CalculatorTest, MultiplyByZero) {
  EXPECT_EQ(calc.Multiply(100, 0), 0);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 0.0);
}

TEST_F(CalculatorTest, MultiplyNegativeNumbers) {
  EXPECT_EQ(calc.Multiply(-3, -4), 12);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 12.0);
}

TEST_F(CalculatorTest, DividePositiveNumbers) {
  EXPECT_DOUBLE_EQ(calc.Divide(15.0, 3.0), 5.0);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 5.0);
}

TEST_F(CalculatorTest, DivideResultsInFraction) {
  EXPECT_DOUBLE_EQ(calc.Divide(10.0, 4.0), 2.5);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 2.5);
}

TEST_F(CalculatorTest, DivideByZeroThrowsException) {
  EXPECT_THROW(calc.Divide(10.0, 0.0), std::invalid_argument);
}

TEST_F(CalculatorTest, LastResultIsUpdatedAfterEachOperation) {
  calc.Add(5, 3);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 8.0);

  calc.Multiply(2, 4);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 8.0);

  calc.Divide(10.0, 2.0);
  EXPECT_DOUBLE_EQ(calc.GetLastResult(), 5.0);
}

}  // namespace
}  // namespace parallel_abm
