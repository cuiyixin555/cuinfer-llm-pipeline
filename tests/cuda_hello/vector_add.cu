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

#include "vector_add.h"

#include <cstddef>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void vectorAddKernel(const float *a, const float *b, float *result,
                                int count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < static_cast<std::size_t>(count)) {
    result[index] = a[index] + b[index];
  }
}

} // namespace

cudaError_t launchVectorAdd(const float *a, const float *b, float *result,
                            int count, cudaStream_t stream) {
  if (count < 0 || (count > 0 && (!a || !b || !result))) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }

  const int blocks = 1 + (count - 1) / kThreadsPerBlock;
  vectorAddKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(a, b, result, count);
  return cudaGetLastError();
}