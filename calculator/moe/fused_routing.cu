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

#include "fused_routing_common.cuh"

#include <algorithm>

namespace cuinfer::moe {
namespace {

__global__ void routingKernel(const void *scores, const void *bias,
                              void *weights, int32_t *indices, int16_t *replay,
                              RoutingOptions o) {
  __shared__ float activated[detail::kMaxExperts];
  __shared__ float ranked[detail::kMaxExperts];

  for (std::size_t token = blockIdx.x;
       token < static_cast<std::size_t>(o.tokens); token += gridDim.x) {
    for (int expert = threadIdx.x; expert < o.experts; expert += blockDim.x) {
      const float logit =
          detail::load(scores, o.score_type, token * o.experts + expert);
      const float value = 1.0f / (1.0f + expf(-logit));
      activated[expert] = value;
      ranked[expert] = value + detail::load(bias, o.bias_type, expert);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int per_group = o.experts / o.groups;
      float group_scores[detail::kMaxGroups];
      for (int group = 0; group < o.groups; ++group) {
        float first = -INFINITY;
        float second = -INFINITY;
        for (int e = group * per_group; e < (group + 1) * per_group; ++e) {
          const float score = ranked[e];
          if (score > first) {
            second = first;
            first = score;
          } else if (score > second) {
            second = score;
          }
        }
        group_scores[group] = first + second;
      }

      uint32_t group_mask = 0;
      for (int i = 0; i < o.selected_groups; ++i) {
        int best = -1;
        for (int group = 0; group < o.groups; ++group) {
          if (!(group_mask & (uint32_t{1} << group)) &&
              (best < 0 || group_scores[group] > group_scores[best]))
            best = group;
        }
        group_mask |= uint32_t{1} << best;
      }

      int chosen[detail::kMaxTopK];
      float terms[detail::kMaxTopK] = {};
      for (int i = 0; i < o.top_k; ++i) {
        int best = -1;
        for (int e = 0; e < o.experts; ++e) {
          if (!(group_mask & (uint32_t{1} << (e / per_group))))
            continue;
          bool already_chosen = false;
          for (int j = 0; j < i; ++j)
            already_chosen |= chosen[j] == e;
          if (!already_chosen && (best < 0 || ranked[e] > ranked[best]))
            best = e;
        }
        chosen[i] = best;
        terms[i] = activated[best];
      }

      for (int step = detail::kMaxTopK / 2; step > 0; step /= 2)
        for (int i = 0; i < step; ++i)
          terms[i] += terms[i + step];
      const float sum = terms[0];

      for (int i = 0; i < o.top_k; ++i) {
        const std::size_t offset = token * o.top_k + i;
        const int expert = chosen[i];
        const float weight =
            static_cast<float>(static_cast<double>(activated[expert]) *
                               o.scale / (static_cast<double>(sum) + 1e-20));
        detail::store(weights, o.score_type, offset, weight);
        indices[offset] = expert;
        if (replay)
          replay[offset] = static_cast<int16_t>(expert);
      }
    }
    __syncthreads();
  }
}

} // namespace

cudaError_t fusedRouting(const void *scores, const void *bias, void *weights,
                         int32_t *indices, int16_t *replay, RoutingOptions o,
                         cudaStream_t stream) {
  const cudaError_t status =
      detail::validate(scores, bias, weights, indices, o);
  if (status != cudaSuccess || o.tokens == 0)
    return status;
  const int blocks = std::min(o.tokens, 65535);
  routingKernel<<<blocks, detail::kThreads, 0, stream>>>(scores, bias, weights,
                                                         indices, replay, o);
  return cudaGetLastError();
}

} // namespace cuinfer::moe