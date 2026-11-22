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

#include <cstdint>
#include <cuda_runtime.h>

namespace cuinfer::moe {

enum class DataType : int { Float32 = 0, Float16 = 1, BFloat16 = 2 };

struct RoutingOptions {
  int tokens;
  int experts;
  int groups;
  int selected_groups;
  int top_k;
  double scale;
  DataType score_type;
  DataType bias_type;
};

// All buffers are contiguous device arrays and outputs must not alias inputs.
// Scores are [tokens, experts], bias is [experts], and outputs are
// [tokens, top_k]. Replay may be null; otherwise it holds int16 expert IDs.
// The launch is asynchronous on stream.
cudaError_t fusedRouting(const void *scores, const void *bias, void *weights,
                         int32_t *indices, int16_t *replay,
                         RoutingOptions options, cudaStream_t stream = nullptr);

cudaError_t fusedRoutingPtx(const void *scores, const void *bias, void *weights,
                            int32_t *indices, int16_t *replay,
                            RoutingOptions options,
                            cudaStream_t stream = nullptr);

} // namespace cuinfer::moe