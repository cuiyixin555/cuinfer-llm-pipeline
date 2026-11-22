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

#include "calculator/mhc/mhc_post.h"
#include "calculator/mhc/mhc_pre.h"

#include <cuda_profiler_api.h>

#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() {
    const cudaError_t status = cudaFree(data_);
    if (status != cudaSuccess)
      std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(status));
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *data() { return data_; }
  void fill(T value) {
    const std::vector<T> host(count_, value);
    check(cudaMemcpy(data_, host.data(), count_ * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
  }

private:
  T *data_ = nullptr;
  std::size_t count_;
};

int parsePositive(const char *text, const char *name) {
  const std::string input(text);
  int value = 0;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), value);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      value <= 0)
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  return value;
}

void run(const std::string &operation, bool ptx, int tokens, int hidden,
         int warmup, int iterations) {
  if (tokens > 65535 || hidden > 1048576)
    throw std::invalid_argument("tokens or hidden exceeds benchmark limits");
  const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
  DeviceBuffer<__nv_bfloat16> x(n), residual(4 * n), output(4 * n), layer(n);
  DeviceBuffer<float> post(4 * tokens), comb(16 * tokens), dot(24 * tokens),
      square(tokens), scale(3), base(24);
  x.fill(__float2bfloat16(0.25f));
  residual.fill(__float2bfloat16(0.01f));
  post.fill(0.5f);
  comb.fill(0.25f);
  dot.fill(0.01f);
  square.fill(4.0f * hidden * 0.0001f);
  scale.fill(0.1f);
  base.fill(0.0f);

  const cuinfer::mhc::PreOptions options{4 * hidden, 1,     20,    0,
                                         1e-6f,      1e-6f, 1e-6f, 1.0f};
  const auto launch = [&]() {
    if (operation == "post") {
      const auto fn = ptx ? cuinfer::mhc::postPtx : cuinfer::mhc::post;
      return fn(x.data(), residual.data(), post.data(), comb.data(),
                output.data(), tokens, hidden, nullptr);
    }
    const auto fn = ptx ? cuinfer::mhc::prePtx : cuinfer::mhc::pre;
    const bool prenorm = operation == "prenorm";
    return fn(dot.data(), prenorm ? nullptr : square.data(), residual.data(),
              scale.data(), base.data(), post.data(), comb.data(), layer.data(),
              tokens, hidden, options, prenorm, nullptr);
  };

  for (int i = 0; i < warmup; ++i)
    check(launch(), "warmup launch");
  check(cudaDeviceSynchronize(), "warmup synchronize");
  check(cudaProfilerStart(), "cudaProfilerStart");
  for (int i = 0; i < iterations; ++i)
    check(launch(), "measured launch");
  check(cudaDeviceSynchronize(), "measured synchronize");
  check(cudaProfilerStop(), "cudaProfilerStop");

  const std::size_t logical_bytes =
      operation == "post"
          ? static_cast<std::size_t>(tokens) * (18ULL * hidden + 80)
          : static_cast<std::size_t>(tokens) *
                    (10ULL * hidden + (operation == "prenorm" ? 176 : 180)) +
                108;
  std::printf("operation=%s variant=%s tokens=%d hidden=%d warmup=%d "
              "iterations=%d logical_bytes=%zu\n",
              operation.c_str(), ptx ? "ptx" : "cpp", tokens, hidden, warmup,
              iterations, logical_bytes);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "Usage: %s <post|pre|prenorm> <cpp|ptx> <tokens> <hidden> "
                 "<warmup> <iterations>\n",
                 argv[0]);
    return 1;
  }
  try {
    const std::string operation(argv[1]);
    const std::string variant(argv[2]);
    if ((operation != "post" && operation != "pre" && operation != "prenorm") ||
        (variant != "cpp" && variant != "ptx"))
      throw std::invalid_argument("invalid operation or variant");
    run(operation, variant == "ptx", parsePositive(argv[3], "tokens"),
        parsePositive(argv[4], "hidden"), parsePositive(argv[5], "warmup"),
        parsePositive(argv[6], "iterations"));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "mHC benchmark: %s\n", error.what());
    return 1;
  }
  return 0;
}
