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

#include "calculator/mhc/mhc_pre.h"
#include "mhc_test_utils.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <utility>

namespace {

using mhc_test::ManagedBuffer;
using mhc_test::sample;

using PreLaunch = decltype(&cuinfer::mhc::pre);

void checkPre(PreLaunch launch, int tokens, int hidden, int splits,
              bool prenorm, int block_size = 0, int repeat = 20,
              float rms_eps = 1e-6f, float pre_eps = 1e-6f,
              float sinkhorn_eps = 1e-6f, float post_multiplier = 1.0f) {
  ManagedBuffer<__nv_bfloat16> residual(tokens * 4 * hidden);
  ManagedBuffer<__nv_bfloat16> layer(tokens * hidden);
  ManagedBuffer<float> dot(splits * tokens * 24);
  ManagedBuffer<float> square(splits * tokens);
  ManagedBuffer<float> scale(3), base(24), post(tokens * 4), comb(tokens * 16);
  for (int i = 0; i < tokens * 4 * hidden; ++i)
    residual[i] = __float2bfloat16(sample(i, 0.01f));
  for (int i = 0; i < splits * tokens * 24; ++i)
    dot[i] = sample(i, 0.01f);
  for (int i = 0; i < splits * tokens; ++i)
    square[i] = 4.0f * hidden * (0.25f + 0.02f * (i % 7)) / splits;
  for (int i = 0; i < 3; ++i)
    scale[i] = sample(i, 0.1f);
  for (int i = 0; i < 24; ++i)
    base[i] = sample(i, 0.1f);

  const cuinfer::mhc::PreOptions options{4 * hidden,   splits,         repeat,
                                         block_size,   rms_eps,        pre_eps,
                                         sinkhorn_eps, post_multiplier};
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  const auto status =
      launch(dot.data(), prenorm ? nullptr : square.data(), residual.data(),
             scale.data(), base.data(), post.data(), comb.data(), layer.data(),
             tokens, hidden, options, prenorm, stream);
  const auto sync_status = cudaStreamSynchronize(stream);
  const auto destroy_status = cudaStreamDestroy(stream);
  ASSERT_EQ(cudaSuccess, status);
  ASSERT_EQ(cudaSuccess, sync_status);
  ASSERT_EQ(cudaSuccess, destroy_status);

  for (int t = 0; t < tokens; ++t) {
    float square_total = 0.0f;
    if (prenorm) {
      for (int i = 0; i < 4 * hidden; ++i) {
        const float value = __bfloat162float(residual[t * 4 * hidden + i]);
        square_total += value * value;
      }
    } else {
      for (int s = 0; s < splits; ++s)
        square_total += square[s * tokens + t];
    }
    const float rstd = 1.0f / std::sqrt(square_total / (4 * hidden) + rms_eps);
    float logits[24] = {};
    for (int s = 0; s < splits; ++s)
      for (int i = 0; i < 24; ++i)
        logits[i] += dot[(s * tokens + t) * 24 + i];
    float pre[4], matrix[4][4];
    double post_error = 0.0, post_norm = 0.0;
    for (int i = 0; i < 4; ++i) {
      pre[i] =
          1.0f / (1.0f + std::exp(-(logits[i] * rstd * scale[0] + base[i]))) +
          pre_eps;
      const float expected =
          post_multiplier /
          (1.0f + std::exp(-(logits[i + 4] * rstd * scale[1] + base[i + 4])));
      post_error += (post[t * 4 + i] - expected) * (post[t * 4 + i] - expected);
      post_norm += expected * expected;
      EXPECT_NEAR(post[t * 4 + i], expected,
                  2e-3f + 2e-3f * std::abs(expected));
    }
    EXPECT_LT(std::sqrt(post_error / std::max(post_norm, 1e-12)), 0.006);
    for (int i = 0; i < 4; ++i) {
      float denominator = 0.0f;
      for (int j = 0; j < 4; ++j) {
        const int index = 8 + i * 4 + j;
        matrix[i][j] = std::exp(logits[index] * rstd * scale[2] + base[index]);
        denominator += matrix[i][j];
      }
      for (int j = 0; j < 4; ++j)
        matrix[i][j] = matrix[i][j] / denominator + sinkhorn_eps;
    }
    for (int iteration = 0; iteration < repeat; ++iteration) {
      if (iteration) {
        for (int i = 0; i < 4; ++i) {
          float sum = sinkhorn_eps;
          for (int j = 0; j < 4; ++j)
            sum += matrix[i][j];
          for (int j = 0; j < 4; ++j)
            matrix[i][j] /= sum;
        }
      }
      for (int j = 0; j < 4; ++j) {
        float sum = sinkhorn_eps;
        for (int i = 0; i < 4; ++i)
          sum += matrix[i][j];
        for (int i = 0; i < 4; ++i)
          matrix[i][j] /= sum;
      }
    }
    double comb_error = 0.0, comb_norm = 0.0;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        const float diff = comb[t * 16 + i * 4 + j] - matrix[i][j];
        comb_error += diff * diff;
        comb_norm += matrix[i][j] * matrix[i][j];
        EXPECT_NEAR(comb[t * 16 + i * 4 + j], matrix[i][j], 0.0025f);
      }
    EXPECT_LT(std::sqrt(comb_error / std::max(comb_norm, 1e-12)), 0.006);
    double error = 0, norm = 0;
    for (int h = 0; h < hidden; ++h) {
      float expected = 0;
      for (int i = 0; i < 4; ++i)
        expected +=
            pre[i] * __bfloat162float(residual[(t * 4 + i) * hidden + h]);
      expected = __bfloat162float(__float2bfloat16(expected));
      const float got = __bfloat162float(layer[t * hidden + h]);
      error += (got - expected) * (got - expected);
      norm += expected * expected;
      EXPECT_NEAR(got, expected, 0.01f + 0.01f * std::abs(expected));
    }
    EXPECT_LT(std::sqrt(error / std::max(norm, 1e-12)), 0.006);
  }
}

TEST(MhcPreTest, MatchesReferenceForAllSplitCounts) {
  for (const auto shape :
       {std::pair{2, 64}, std::pair{1, 4096}, std::pair{6, 128}})
    for (int splits : {1, 2, 4, 8, 16}) {
      SCOPED_TRACE(::testing::Message()
                   << "tokens=" << shape.first << " H=" << shape.second
                   << " splits=" << splits);
      checkPre(cuinfer::mhc::pre, shape.first, shape.second, splits, false);
    }
}

TEST(MhcPreTest, ComputesSquareSumsFromResidual) {
  for (const auto shape :
       {std::pair{2, 64}, std::pair{1, 4096}, std::pair{4, 7168}}) {
    SCOPED_TRACE(shape.second);
    checkPre(cuinfer::mhc::pre, shape.first, shape.second, 1, true);
  }
}

TEST(MhcPrePtxTest, MatchesReferenceForAllSplitCounts) {
  for (const auto shape :
       {std::pair{2, 64}, std::pair{1, 4096}, std::pair{6, 128}})
    for (int splits : {1, 2, 4, 8, 16}) {
      SCOPED_TRACE(::testing::Message()
                   << "tokens=" << shape.first << " H=" << shape.second
                   << " splits=" << splits);
      checkPre(cuinfer::mhc::prePtx, shape.first, shape.second, splits, false);
    }
}

TEST(MhcPrePtxTest, ComputesSquareSumsFromResidual) {
  for (const auto shape : {std::pair{2, 64}, std::pair{1, 4096},
                           std::pair{4, 7168}, std::pair{2, 65}})
    for (int block_size : {128, 256, 512}) {
      SCOPED_TRACE(::testing::Message()
                   << "H=" << shape.second << " block_size=" << block_size);
      checkPre(cuinfer::mhc::prePtx, shape.first, shape.second, 1, true,
               block_size);
    }
}

TEST(MhcPrePtxTest, SupportsNondefaultParameters) {
  for (int repeat : {1, 3}) {
    SCOPED_TRACE(repeat);
    checkPre(cuinfer::mhc::prePtx, 2, 65, 4, false, 256, repeat, 0.02f, 0.03f,
             0.04f, 1.7f);
    checkPre(cuinfer::mhc::prePtx, 2, 65, 1, true, 512, repeat, 0.02f, 0.03f,
             0.04f, 1.7f);
  }
}

TEST(MhcPreArgumentsTest, RejectsInvalidLaunchParameters) {
  cuinfer::mhc::PreOptions options{256, 3, 20, 0, 1e-6f, 1e-6f, 1e-6f, 1.0f};
  EXPECT_EQ(cudaErrorInvalidValue,
            cuinfer::mhc::pre(nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, 1, 64, options,
                              false));
}

TEST(MhcPrePtxTest, RejectsInvalidLaunchParameters) {
  cuinfer::mhc::PreOptions options{256, 3, 20, 0, 1e-6f, 1e-6f, 1e-6f, 1.0f};
  EXPECT_EQ(cudaErrorInvalidValue,
            cuinfer::mhc::prePtx(nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, 1, 64, options,
                                 false));
  options.splits = 1;
  EXPECT_EQ(cudaSuccess, cuinfer::mhc::prePtx(
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, 0, 64, options, false));
}

} // namespace
