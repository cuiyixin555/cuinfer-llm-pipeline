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

#include "mhc_common.h"

#include <cmath>

namespace cuinfer::mhc::detail {

inline cudaError_t validatePost(const __nv_bfloat16 *x,
                                const __nv_bfloat16 *residual,
                                const float *post_mix, const float *comb_mix,
                                __nv_bfloat16 *output, int tokens, int hidden) {
  if (tokens < 0 || hidden <= 0)
    return cudaErrorInvalidValue;
  if (tokens == 0)
    return cudaSuccess;
  if (!x || !residual || !post_mix || !comb_mix || !output)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

inline cudaError_t validatePre(const float *dot_mix, const float *square_sum,
                               const __nv_bfloat16 *residual,
                               const float *scale, const float *base,
                               float *post_mix, float *comb_mix,
                               __nv_bfloat16 *layer_input, int tokens,
                               int hidden, PreOptions options,
                               bool compute_square_sum) {
  if (tokens < 0 || hidden <= 0 || options.k <= 0 ||
      (options.splits != 1 && options.splits != 2 && options.splits != 4 &&
       options.splits != 8 && options.splits != 16) ||
      (compute_square_sum && options.splits != 1) ||
      options.sinkhorn_repeat < 1 || options.block_size < 0 ||
      !(options.rms_eps > 0.0f) || !std::isfinite(options.rms_eps) ||
      !(options.pre_eps > 0.0f) || !std::isfinite(options.pre_eps) ||
      !(options.sinkhorn_eps > 0.0f) || !std::isfinite(options.sinkhorn_eps) ||
      !std::isfinite(options.post_multiplier))
    return cudaErrorInvalidValue;
  if (tokens == 0)
    return cudaSuccess;
  if (!dot_mix || (!compute_square_sum && !square_sum) || !residual || !scale ||
      !base || !post_mix || !comb_mix || !layer_input)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

} // namespace cuinfer::mhc::detail
