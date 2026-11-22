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

#include "mhc_post.h"
#include "mhc_pre.h"

extern "C" {

int cuinfer_mhc_post(const void *x, const void *residual, const void *post_mix,
                     const void *comb_mix, void *output, int tokens, int hidden,
                     void *stream) {
  return cuinfer::mhc::post(static_cast<const __nv_bfloat16 *>(x),
                            static_cast<const __nv_bfloat16 *>(residual),
                            static_cast<const float *>(post_mix),
                            static_cast<const float *>(comb_mix),
                            static_cast<__nv_bfloat16 *>(output), tokens,
                            hidden, static_cast<cudaStream_t>(stream));
}

int cuinfer_mhc_pre(const void *dot_mix, const void *square_sum,
                    const void *residual, const void *scale, const void *base,
                    void *post_mix, void *comb_mix, void *layer_input,
                    int tokens, int hidden, int k, int splits, int repeat,
                    int block_size, float rms_eps, float pre_eps,
                    float sinkhorn_eps, float post_multiplier,
                    int compute_square_sum, void *stream) {
  const cuinfer::mhc::PreOptions options{
      k,       splits,  repeat,       block_size,
      rms_eps, pre_eps, sinkhorn_eps, post_multiplier};
  return cuinfer::mhc::pre(
      static_cast<const float *>(dot_mix),
      static_cast<const float *>(square_sum),
      static_cast<const __nv_bfloat16 *>(residual),
      static_cast<const float *>(scale), static_cast<const float *>(base),
      static_cast<float *>(post_mix), static_cast<float *>(comb_mix),
      static_cast<__nv_bfloat16 *>(layer_input), tokens, hidden, options,
      compute_square_sum != 0, static_cast<cudaStream_t>(stream));
}

const char *cuinfer_cuda_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}

} // extern "C"