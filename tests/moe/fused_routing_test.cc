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

#include "calculator/moe/fused_routing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using cuinfer::moe::DataType;
using cuinfer::moe::RoutingOptions;
using Launch = cudaError_t (*)(const void *, const void *, void *, int32_t *,
                               int16_t *, RoutingOptions, cudaStream_t);

template <typename T> class Buffer {
public:
  explicit Buffer(std::size_t count) {
    const cudaError_t status = cudaMallocManaged(&data_, count * sizeof(T));
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
  ~Buffer() {
    const cudaError_t status = cudaFree(data_);
    if (status != cudaSuccess)
      ADD_FAILURE() << cudaGetErrorString(status);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  T *data() { return data_; }
  T &operator[](std::size_t i) { return data_[i]; }

private:
  T *data_ = nullptr;
};

void checkRouting(Launch launch, RoutingOptions o) {
  const int per_group = o.experts / o.groups;
  Buffer<float> scores(static_cast<std::size_t>(o.tokens) * o.experts);
  Buffer<float> bias(o.experts);
  Buffer<float> weights(static_cast<std::size_t>(o.tokens) * o.top_k + 1);
  Buffer<int32_t> indices(static_cast<std::size_t>(o.tokens) * o.top_k + 1);
  Buffer<int16_t> replay(static_cast<std::size_t>(o.tokens) * o.top_k + 1);
  for (int e = 0; e < o.experts; ++e)
    bias[e] = static_cast<float>((e * 37) % o.experts) * 0.02f - 2.0f;
  for (int t = 0; t < o.tokens; ++t)
    for (int e = 0; e < o.experts; ++e)
      scores[static_cast<std::size_t>(t) * o.experts + e] =
          std::sin((t * o.experts + e) * 0.17f);
  const std::size_t end = static_cast<std::size_t>(o.tokens) * o.top_k;
  weights[end] = -9.0f;
  indices[end] = -9;
  replay[end] = -9;

  ASSERT_EQ(cudaSuccess, launch(scores.data(), bias.data(), weights.data(),
                                indices.data(), replay.data(), o, nullptr));
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  for (int t = 0; t < o.tokens; ++t) {
    std::vector<float> sigmoid(o.experts), ranked(o.experts),
        group_scores(o.groups);
    for (int e = 0; e < o.experts; ++e) {
      sigmoid[e] =
          1.0f /
          (1.0f +
           std::exp(-scores[static_cast<std::size_t>(t) * o.experts + e]));
      ranked[e] = sigmoid[e] + bias[e];
    }
    for (int g = 0; g < o.groups; ++g) {
      std::vector<float> group(ranked.begin() + g * per_group,
                               ranked.begin() + (g + 1) * per_group);
      std::partial_sort(group.begin(), group.begin() + 2, group.end(),
                        std::greater<float>());
      group_scores[g] = group[0] + group[1];
    }
    std::vector<int> groups(o.groups);
    std::iota(groups.begin(), groups.end(), 0);
    std::stable_sort(groups.begin(), groups.end(), [&](int a, int b) {
      return group_scores[a] > group_scores[b];
    });
    std::vector<int> experts;
    for (int g = 0; g < o.selected_groups; ++g)
      for (int e = groups[g] * per_group; e < (groups[g] + 1) * per_group; ++e)
        experts.push_back(e);
    std::sort(experts.begin(), experts.end(), [&](int a, int b) {
      return ranked[a] != ranked[b] ? ranked[a] > ranked[b] : a < b;
    });
    float sum = 0.0f;
    for (int k = 0; k < o.top_k; ++k)
      sum += sigmoid[experts[k]];
    for (int k = 0; k < o.top_k; ++k) {
      const std::size_t i = static_cast<std::size_t>(t) * o.top_k + k;
      const float expected =
          static_cast<float>(static_cast<double>(sigmoid[experts[k]]) *
                             o.scale / (static_cast<double>(sum) + 1e-20));
      EXPECT_EQ(indices[i], experts[k]) << "token " << t << ", slot " << k;
      EXPECT_EQ(replay[i], indices[i]);
      EXPECT_NEAR(weights[i], expected, 2e-6f);
    }
  }
  EXPECT_EQ(weights[end], -9.0f);
  EXPECT_EQ(indices[end], -9);
  EXPECT_EQ(replay[end], -9);
}

void checkAllRouting(Launch launch) {
  for (const RoutingOptions o :
       {RoutingOptions{7, 256, 8, 4, 8, 2.5, DataType::Float32,
                       DataType::Float32},
        RoutingOptions{3, 128, 4, 2, 4, 1.0, DataType::Float32,
                       DataType::Float32},
        RoutingOptions{4, 384, 1, 1, 1, 1.0, DataType::Float32,
                       DataType::Float32},
        RoutingOptions{2, 64, 32, 3, 5, 0.5, DataType::Float32,
                       DataType::Float32}}) {
    SCOPED_TRACE(::testing::Message()
                 << "experts=" << o.experts << " groups=" << o.groups);
    checkRouting(launch, o);
  }
}

void checkValidation(Launch launch) {
  RoutingOptions o{0, 256, 8, 4, 8, 2.5, DataType::Float32, DataType::Float32};
  EXPECT_EQ(cudaSuccess,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
  o.tokens = 1;
  EXPECT_EQ(cudaErrorInvalidValue,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
  o.tokens = 0;
  o.groups = 0;
  EXPECT_EQ(cudaErrorInvalidValue,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
  o.groups = 8;
  o.scale = std::numeric_limits<double>::infinity();
  EXPECT_EQ(cudaErrorInvalidValue,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
  o.scale = 1.0;
  o.score_type = static_cast<DataType>(99);
  EXPECT_EQ(cudaErrorInvalidValue,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
  o.score_type = DataType::Float32;
  o.selected_groups = 1;
  o.top_k = 9;
  EXPECT_EQ(cudaErrorInvalidValue,
            launch(nullptr, nullptr, nullptr, nullptr, nullptr, o, nullptr));
}

TEST(FusedRoutingTest, MatchesCpuReference) {
  checkAllRouting(cuinfer::moe::fusedRouting);
}

TEST(FusedRoutingTest, RejectsInvalidAndAllowsEmptyInputs) {
  checkValidation(cuinfer::moe::fusedRouting);
}

TEST(FusedRoutingPtxTest, MatchesCpuReference) {
  checkAllRouting(cuinfer::moe::fusedRoutingPtx);
}

TEST(FusedRoutingPtxTest, RejectsInvalidAndAllowsEmptyInputs) {
  checkValidation(cuinfer::moe::fusedRoutingPtx);
}

} // namespace