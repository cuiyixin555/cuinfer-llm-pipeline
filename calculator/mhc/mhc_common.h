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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cuinfer::mhc {

constexpr int kHeads = 4;
constexpr int kMixValues = 24;

struct PreOptions {
  int k;
  int splits;
  int sinkhorn_repeat;
  int block_size;
  float rms_eps;
  float pre_eps;
  float sinkhorn_eps;
  float post_multiplier;
};

#ifdef __CUDACC__
namespace detail {

template <bool UsePtx> struct Arithmetic {
  __device__ static float fromBf16(__nv_bfloat16 value) {
    if constexpr (UsePtx) {
      float result;
      const unsigned short bits = __bfloat16_as_ushort(value);
      asm("mov.b32 %0, {0, %1};" : "=f"(result) : "h"(bits));
      return result;
    }
    return __bfloat162float(value);
  }

  __device__ static __nv_bfloat16 toBf16(float value) {
    if constexpr (UsePtx) {
      unsigned short bits;
      asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
      return __ushort_as_bfloat16(bits);
    }
    return __float2bfloat16(value);
  }

  __device__ static float add(float a, float b) {
    if constexpr (UsePtx) {
      float result;
      asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
      return result;
    }
    return a + b;
  }

  __device__ static float multiplyAdd(float a, float b, float c) {
    if constexpr (UsePtx) {
      float result;
      asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
      return result;
    }
    return a * b + c;
  }

  __device__ static float divide(float a, float b) {
    if constexpr (UsePtx) {
      float result;
      asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
      return result;
    }
    return a / b;
  }

  __device__ static float reciprocalSqrt(float value) {
    if constexpr (UsePtx) {
      float result;
      asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
      return result;
    }
    return rsqrtf(value);
  }
};

} // namespace detail
#endif

} // namespace cuinfer::mhc