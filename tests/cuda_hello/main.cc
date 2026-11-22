// MIT License
//
// Copyright (c) 2026 CUI Xin (崔 欣)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "vector_add.h"

#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr int kMaxCount = 1025;
constexpr float kSentinel = -12345.0f;

class VectorAddTest : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override {
    const std::size_t bytes = kMaxCount * sizeof(float);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&a_, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&b_, bytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&result_, bytes));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_)
      EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_));
    if (a_)
      EXPECT_EQ(cudaSuccess, cudaFree(a_));
    if (b_)
      EXPECT_EQ(cudaSuccess, cudaFree(b_));
    if (result_)
      EXPECT_EQ(cudaSuccess, cudaFree(result_));
  }

  float *a_ = nullptr;
  float *b_ = nullptr;
  float *result_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

TEST_P(VectorAddTest, AddsEveryElementWithoutWritingPastTheEnd) {
  const int count = GetParam();
  std::vector<float> a(count + 1), b(count + 1);
  std::vector<float> result(count + 1, kSentinel);
  for (int i = 0; i < count; ++i) {
    a[i] = static_cast<float>(i - 500);
    b[i] = static_cast<float>(2 * i + 1);
  }
  const std::size_t bytes = result.size() * sizeof(float);
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(a_, a.data(), bytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(b_, b.data(), bytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(result_, result.data(), bytes, cudaMemcpyHostToDevice));
  const cudaStream_t stream = count == 1 ? nullptr : stream_;
  ASSERT_EQ(cudaSuccess, launchVectorAdd(a_, b_, result_, count, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(result.data(), result_, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(a[i] + b[i], result[i]) << "index " << i;
  }
  EXPECT_FLOAT_EQ(kSentinel, result[count]);
}

INSTANTIATE_TEST_SUITE_P(Sizes, VectorAddTest,
                         ::testing::Values(1, 255, 256, 257, 1024));

TEST(VectorAddArgumentsTest, HandlesEmptyAndInvalidInputs) {
  EXPECT_EQ(cudaSuccess, launchVectorAdd(nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(cudaErrorInvalidValue,
            launchVectorAdd(nullptr, nullptr, nullptr, -1));
  EXPECT_EQ(cudaErrorInvalidValue,
            launchVectorAdd(nullptr, nullptr, nullptr, 1));
}

} // namespace