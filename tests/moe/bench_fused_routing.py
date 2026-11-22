# MIT License
#
# Copyright (c) 2026 CUI Xin (崔 欣)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Profile a fixed DeepSeek-V3-style routing workload with Nsight Systems."""

import argparse

import torch

from cuinfer_llm.moe import fused_topk_deepseek


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0 or number > 2**31 - 1:
        raise argparse.ArgumentTypeError("must be a positive 32-bit integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("cpp", "ptx"))
    parser.add_argument("tokens", type=positive_int)
    parser.add_argument("--warmup", type=positive_int, default=200)
    parser.add_argument("--iterations", type=positive_int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("an NVIDIA GPU with compute capability 8.0 or later is required")

    experts, groups, selected_groups, top_k, scale = 256, 8, 4, 8, 2.5
    generator = torch.Generator(device="cuda").manual_seed(2026)
    scores = torch.randn(
        (args.tokens, experts), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    bias = torch.randn(experts, dtype=torch.float32, device="cuda", generator=generator)
    weights = torch.empty((args.tokens, top_k), dtype=scores.dtype, device="cuda")
    indices = torch.empty((args.tokens, top_k), dtype=torch.int32, device="cuda")
    replay = torch.empty((args.tokens, top_k), dtype=torch.int16, device="cuda")

    def launch() -> None:
        fused_topk_deepseek(
            scores, bias, groups, selected_groups, top_k, scale, weights, indices,
            replay, use_ptx=args.variant == "ptx",
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    runtime = torch.cuda.cudart()
    torch.cuda.check_error(runtime.cudaProfilerStart())
    try:
        for _ in range(args.iterations):
            launch()
        torch.cuda.synchronize()
    finally:
        torch.cuda.check_error(runtime.cudaProfilerStop())

    logical_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (scores, bias, weights, indices, replay)
    )
    print(
        f"variant={args.variant} tokens={args.tokens} experts={experts} groups={groups} "
        f"selected_groups={selected_groups} top_k={top_k} scale={scale} "
        f"score_dtype=bfloat16 bias_dtype=float32 replay=true seed=2026 "
        f"warmup={args.warmup} iterations={args.iterations} logical_bytes={logical_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()