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

#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace mhc_test {

template <typename T> class ManagedBuffer {
public:
  explicit ManagedBuffer(std::size_t count) {
    const cudaError_t status = cudaMallocManaged(&data_, count * sizeof(T));
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
  ~ManagedBuffer() {
    const cudaError_t status = cudaFree(data_);
    if (status != cudaSuccess)
      ADD_FAILURE() << cudaGetErrorString(status);
  }
  ManagedBuffer(const ManagedBuffer &) = delete;
  ManagedBuffer &operator=(const ManagedBuffer &) = delete;
  T *data() { return data_; }
  T &operator[](std::size_t i) { return data_[i]; }

private:
  T *data_ = nullptr;
};

inline float sample(int i, float factor = 1.0f) {
  return std::sin(i * 0.173f + 0.3f) * factor;
}

} // namespace mhc_test
