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

#include "mhc_launch_utils.h"
#include "mhc_post.h"

#include <algorithm>
#include <cstddef>

namespace cuinfer::mhc {
namespace {

__global__ void postKernel(const __nv_bfloat16 *x,
                           const __nv_bfloat16 *residual, const float *post_mix,
                           const float *comb_mix, __nv_bfloat16 *output,
                           int tokens, int hidden) {
  const std::size_t total = static_cast<std::size_t>(tokens) * hidden;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const std::size_t token = i / hidden;
    const std::size_t h = i % hidden;
    const float value = detail::Arithmetic<false>::fromBf16(x[i]);
    for (int out_head = 0; out_head < kHeads; ++out_head) {
      float combined = 0.0f;
      for (int in_head = 0; in_head < kHeads; ++in_head) {
        combined = detail::Arithmetic<false>::multiplyAdd(
            detail::Arithmetic<false>::fromBf16(
                residual[(token * kHeads + in_head) * hidden + h]),
            comb_mix[token * kHeads * kHeads + in_head * kHeads + out_head],
            combined);
      }
      output[(token * kHeads + out_head) * hidden + h] =
          detail::Arithmetic<false>::toBf16(
              detail::Arithmetic<false>::multiplyAdd(
                  value, post_mix[token * kHeads + out_head], combined));
    }
  }
}

} // namespace

cudaError_t post(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                 const float *post_mix, const float *comb_mix,
                 __nv_bfloat16 *output, int tokens, int hidden,
                 cudaStream_t stream) {
  const cudaError_t status = detail::validatePost(
      x, residual, post_mix, comb_mix, output, tokens, hidden);
  if (status != cudaSuccess || tokens == 0)
    return status;
  const std::size_t total = static_cast<std::size_t>(tokens) * hidden;
  const int blocks =
      static_cast<int>(std::min<std::size_t>((total + 255) / 256, 65535));
  postKernel<<<blocks, 256, 0, stream>>>(x, residual, post_mix, comb_mix,
                                         output, tokens, hidden);
  return cudaGetLastError();
}

} // namespace cuinfer::mhc