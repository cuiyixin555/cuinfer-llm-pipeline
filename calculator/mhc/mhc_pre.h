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

namespace cuinfer::mhc {

// Inputs and outputs are contiguous device arrays. dot_mix and square_sum use
// [splits, tokens, ...] layout; square_sum is unused when computed internally.
// Output buffers must not alias inputs. The launch is asynchronous on stream.
cudaError_t pre(const float *dot_mix, const float *square_sum,
                const __nv_bfloat16 *residual, const float *scale,
                const float *base, float *post_mix, float *comb_mix,
                __nv_bfloat16 *layer_input, int tokens, int hidden,
                PreOptions options, bool compute_square_sum,
                cudaStream_t stream = nullptr);

// Equivalent operation with inline PTX arithmetic.
cudaError_t prePtx(const float *dot_mix, const float *square_sum,
                   const __nv_bfloat16 *residual, const float *scale,
                   const float *base, float *post_mix, float *comb_mix,
                   __nv_bfloat16 *layer_input, int tokens, int hidden,
                   PreOptions options, bool compute_square_sum,
                   cudaStream_t stream = nullptr);

} // namespace cuinfer::mhc