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

#include "calculator/mhc/mhc_post.h"
#include "mhc_test_utils.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace {

using mhc_test::ManagedBuffer;
using mhc_test::sample;

using PostLaunch = decltype(&cuinfer::mhc::post);

void checkPost(PostLaunch launch) {
  for (const int hidden : {64, 127, 1536, 2048, 4096, 7168, 8192, 16392}) {
    SCOPED_TRACE(hidden);
    const int tokens = hidden == 127 ? 6 : 1;
    ManagedBuffer<__nv_bfloat16> x(tokens * hidden);
    ManagedBuffer<__nv_bfloat16> residual(tokens * 4 * hidden);
    ManagedBuffer<__nv_bfloat16> output(tokens * 4 * hidden);
    ManagedBuffer<float> post(tokens * 4);
    ManagedBuffer<float> comb(tokens * 16);
    for (int i = 0; i < tokens * hidden; ++i)
      x[i] = __float2bfloat16(sample(i));
    for (int i = 0; i < tokens * 4 * hidden; ++i)
      residual[i] = __float2bfloat16(sample(i, 0.5f));
    for (int i = 0; i < tokens * 4; ++i)
      post[i] = sample(i);
    for (int i = 0; i < tokens * 16; ++i)
      comb[i] = sample(i, 0.7f);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    const auto status =
        launch(x.data(), residual.data(), post.data(), comb.data(),
               output.data(), tokens, hidden, stream);
    const auto sync_status = cudaStreamSynchronize(stream);
    const auto destroy_status = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, status);
    ASSERT_EQ(cudaSuccess, sync_status);
    ASSERT_EQ(cudaSuccess, destroy_status);
    double error = 0.0, norm = 0.0;
    for (int t = 0; t < tokens; ++t)
      for (int j = 0; j < 4; ++j)
        for (int h = 0; h < hidden; ++h) {
          float expected =
              __bfloat162float(x[t * hidden + h]) * post[t * 4 + j];
          for (int i = 0; i < 4; ++i)
            expected += __bfloat162float(residual[(t * 4 + i) * hidden + h]) *
                        comb[t * 16 + i * 4 + j];
          expected = __bfloat162float(__float2bfloat16(expected));
          const float got = __bfloat162float(output[(t * 4 + j) * hidden + h]);
          error += (got - expected) * (got - expected);
          norm += expected * expected;
          EXPECT_NEAR(got, expected, 0.01f + 0.01f * std::abs(expected));
        }
    EXPECT_LT(std::sqrt(error / std::max(norm, 1e-12)), 0.005);
  }
}

TEST(MhcPostTest, MatchesScalarReferenceAcrossShapes) {
  checkPost(cuinfer::mhc::post);
}

TEST(MhcPostPtxTest, MatchesScalarReferenceAcrossShapes) {
  checkPost(cuinfer::mhc::postPtx);
}

TEST(MhcPostArgumentsTest, RejectsInvalidLaunchParameters) {
  EXPECT_EQ(cudaErrorInvalidValue, cuinfer::mhc::post(nullptr, nullptr, nullptr,
                                                      nullptr, nullptr, 1, 64));
}

TEST(MhcPostPtxTest, RejectsInvalidLaunchParameters) {
  EXPECT_EQ(cudaErrorInvalidValue,
            cuinfer::mhc::postPtx(nullptr, nullptr, nullptr, nullptr, nullptr,
                                  1, 64));
  EXPECT_EQ(cudaSuccess, cuinfer::mhc::postPtx(nullptr, nullptr, nullptr,
                                               nullptr, nullptr, 0, 64));
}

} // namespace