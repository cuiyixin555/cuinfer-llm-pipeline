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

#pragma once

#include "fused_routing.h"

#include <cmath>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace cuinfer::moe::detail {

constexpr int kMaxExperts = 384;
constexpr int kMaxGroups = 32;
constexpr int kMaxTopK = 8;
constexpr int kThreads = 256;

inline bool validType(DataType type) {
  return type == DataType::Float32 || type == DataType::Float16 ||
         type == DataType::BFloat16;
}

inline cudaError_t validate(const void *scores, const void *bias, void *weights,
                            int32_t *indices, RoutingOptions o) {
  if (o.tokens < 0 || o.experts < 2 || o.experts > kMaxExperts ||
      o.groups < 1 || o.groups > kMaxGroups || o.experts % o.groups != 0 ||
      o.experts / o.groups < 2 || o.selected_groups < 1 ||
      o.selected_groups > o.groups || o.top_k < 1 || o.top_k > kMaxTopK ||
      o.top_k > (o.experts / o.groups) * o.selected_groups ||
      !std::isfinite(o.scale) || !validType(o.score_type) ||
      !validType(o.bias_type))
    return cudaErrorInvalidValue;
  if (o.tokens > 0 && (!scores || !bias || !weights || !indices))
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

__device__ inline float load(const void *data, DataType type, std::size_t i) {
  switch (type) {
  case DataType::Float32:
    return static_cast<const float *>(data)[i];
  case DataType::Float16:
    return __half2float(static_cast<const __half *>(data)[i]);
  case DataType::BFloat16:
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(data)[i]);
  }
  return 0.0f;
}

__device__ inline void store(void *data, DataType type, std::size_t i,
                             float value) {
  switch (type) {
  case DataType::Float32:
    static_cast<float *>(data)[i] = value;
    break;
  case DataType::Float16:
    static_cast<__half *>(data)[i] = __float2half_rn(value);
    break;
  case DataType::BFloat16:
    static_cast<__nv_bfloat16 *>(data)[i] = __float2bfloat16_rn(value);
    break;
  }
}

} // namespace cuinfer::moe::detail
