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

#include "fused_routing.h"

extern "C" {

int cuinfer_moe_fused_routing(const void *scores, const void *bias,
                              void *weights, int32_t *indices, int16_t *replay,
                              int tokens, int experts, int groups,
                              int selected_groups, int top_k, double scale,
                              int score_type, int bias_type, int use_ptx,
                              void *stream) {
  const cuinfer::moe::RoutingOptions options{
      tokens,
      experts,
      groups,
      selected_groups,
      top_k,
      scale,
      static_cast<cuinfer::moe::DataType>(score_type),
      static_cast<cuinfer::moe::DataType>(bias_type)};
  const auto launch =
      use_ptx ? cuinfer::moe::fusedRoutingPtx : cuinfer::moe::fusedRouting;
  return launch(scores, bias, weights, indices, replay, options,
                static_cast<cudaStream_t>(stream));
}

const char *cuinfer_moe_cuda_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}

} // extern "C"