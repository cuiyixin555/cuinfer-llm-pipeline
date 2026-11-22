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

// Buffers are contiguous device arrays; output must not alias an input.
// The launch is asynchronous on stream.
cudaError_t post(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                 const float *post_mix, const float *comb_mix,
                 __nv_bfloat16 *output, int tokens, int hidden,
                 cudaStream_t stream = nullptr);

// Equivalent operation with inline PTX arithmetic.
cudaError_t postPtx(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                    const float *post_mix, const float *comb_mix,
                    __nv_bfloat16 *output, int tokens, int hidden,
                    cudaStream_t stream = nullptr);

} // namespace cuinfer::mhc