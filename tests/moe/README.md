# MoE fused routing tests

This module implements DeepSeek-V3-style grouped expert routing independently
of FlashInfer's optimized kernels. It selects groups by the sum of their two
highest `sigmoid(logit) + bias` scores, then selects experts by the same biased
scores. Output weights are the selected *unbiased* sigmoid scores divided by
their sum plus `1e-20`, multiplied by the routing scale. Equal group or expert
scores are resolved in favor of the smaller index.

The CUDA C++ and inline PTX kernels are in
[fused_routing.cu](../../calculator/moe/fused_routing.cu) and
[fused_routing_ptx.cu](../../calculator/moe/fused_routing_ptx.cu). The PTX
variant uses inline PTX for selected arithmetic; its control flow, exponential,
and data-type conversions use CUDA C++. These are correctness-oriented
implementations, not the optimized FlashInfer default or Cake backends, and do
not implement Programmatic Dependent Launch.

## Requirements and supported inputs

- Bazel with Bzlmod, a C++ compiler, system GTest, CUDA Toolkit at
  `/usr/local/cuda`, and an NVIDIA GPU with compute capability 8.0 or later.
- CUDA-enabled PyTorch for the Python tests.
- Contiguous CUDA scores `[tokens, experts]` and bias `[experts]`; each may
  independently use float32, float16, or bfloat16. Computation uses float32
  for sigmoid, bias, selection, and the sum, with double-precision final
  scaling/division. Inputs are expected to contain finite values.
- `2 <= experts <= 384`, `1 <= groups <= 32`, with experts evenly divided
  among groups and at least two experts per group. Select between one and all
  groups and between one and eight experts, not exceeding the number of
  experts in selected groups.
- Output weights have shape `[tokens, top_k]` and the score dtype; indices
  have the same shape and int32 dtype. Optional int16 replay output has at
  least `tokens` rows and exactly `top_k` columns. Outputs must not alias
  inputs. Zero tokens are supported.

## Build and run

For the Python test, activate a CUDA-enabled PyTorch environment and return to
the repository root:

```bash
cd /mnt/c/Users/xincu/python-env
source .venv/bin/activate
cd /mnt/c/Users/xincu/cuInfer-llm-pipeline
```

`bazel test` builds the necessary library and test targets before running:

### Python UT

```bash
bazel test //tests/moe:fused_routing_python_test --test_env=PATH --nocache_test_results --test_output=all
```

### CUDA C++ UT

```bash
bazel test //tests/moe:fused_routing_cpp_test --test_arg='--gtest_filter=FusedRoutingTest.*' --nocache_test_results --test_output=all
```

### CUDA PTX UT

```bash
bazel test //tests/moe:fused_routing_cpp_test --test_arg='--gtest_filter=FusedRoutingPtxTest.*' --nocache_test_results --test_output=all
```

The C++ test exercises both kernel variants against a scalar reference. The
Python test compares them with a PyTorch reference, including mixed dtypes,
negative logits, ties, replay buffers, CUDA graphs, and empty batches.

The Python API `cuinfer_llm.moe.fused_topk_deepseek` writes into preallocated
output buffers on the current PyTorch CUDA stream. Pass `use_ptx=True` to
select the PTX variant; the default is CUDA C++. It does not support autograd.
Outside Bazel, build `//calculator/moe:build_moe` and add `python/` to
`PYTHONPATH`. The library defaults to `bazel-bin/calculator/moe/libmoe.so`;
set `CUINFER_MOE_LIBRARY` to override it.

## Nsight Systems performance

[bench_fused_routing.py](bench_fused_routing.py) prepares fixed-seed inputs,
warms up the selected kernel, and uses the CUDA profiler API to mark the
measured launches. It prints the workload and `logical_bytes`; it does not
itself measure latency or verify correctness.

Measured on 2026-11-15 with Nsight Systems 2026.1.3, CUDA Toolkit 13.3,
PyTorch 2.12.0+cu130, and an NVIDIA RTX 500 Ada Generation Laptop GPU
(SM 8.9, driver 621.14), under WSL2 Ubuntu 24.04. Each table entry is one
trace of 1,000 kernel executions after 200 warmups. Clocks were not locked;
the results are illustrative and small differences are not evidence of a
statistically established speedup.

All cases use 256 experts, 8 groups, 4 selected groups, top-k 8, scale 2.5,
BF16 scores and weights, FP32 bias, int32 indices, and int16 replay. Inputs
are generated with seed 2026 and are identical between variants for each
token count. Latency is the mean GPU kernel duration from
`cuda_gpu_kern_sum`, excluding Python launch, allocation, and warmup time.

| Tokens | CUDA C++ latency (us) | CUDA C++ effective bandwidth (GB/s) | Inline PTX latency (us) | Inline PTX effective bandwidth (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 329.925 | 0.004850 | 339.315 | 0.004715 |
| 128 | 554.578 | 0.134791 | 556.135 | 0.134413 |
| 1024 | 3626.199 | 0.162939 | 3614.557 | 0.163463 |

**Bandwidth is calculated, not measured DRAM bandwidth.** Nsight Systems
reported insufficient GPU performance-counter privileges
(`ERR_NVGPUCTRPERM`) in this environment. Effective bandwidth is the unique
logical input/output tensor bytes per launch divided by the mean GPU kernel
duration, in decimal GB/s. It does not account for cache effects, repeated
reads, or actual memory transactions. For `T` tokens the byte count is
`T * (256 * 2 + 8 * (2 + 4 + 2)) + 256 * 4`, or `T * 576 + 1024`;
the shared bias is counted once.

To reproduce from the repository root with the PyTorch environment activated:

```bash
bazel build //calculator/moe:build_moe
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
report_dir="$(mktemp -d)"
for tokens in 1 128 1024; do
  for variant in cpp ptx; do
    report="$report_dir/${variant}_${tokens}"
    nsys profile --trace=cuda --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output="$report" \
      python tests/moe/bench_fused_routing.py "$variant" "$tokens" --warmup 200 --iterations 1000
    nsys stats --report cuda_gpu_kern_sum "$report.nsys-rep"
  done
done
```

Divide the printed `logical_bytes` by the report's `Avg (ns)` to obtain
effective bandwidth in GB/s. To run one workload without Nsight Systems:

```bash
python tests/moe/bench_fused_routing.py cpp 128 --warmup 200 --iterations 1000
```