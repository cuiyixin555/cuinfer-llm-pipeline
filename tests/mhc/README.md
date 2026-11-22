# mHC CUDA tests

These tests exercise independent CUDA implementations of the HC=4 mHC post-map
and pre-map (including split inputs and residual prenormalization), using both
C++/GTest and Python/PyTorch references. C++ and Python tests are each separated
into post and pre big-fuse files. The implementation is intended for correctness;
it does not reproduce FlashInfer's optimized kernel dispatch.

The C++ tests cover both the standard CUDA C++ kernels and separate inline PTX
variants. The PTX variants explicitly implement BF16 conversion and core
floating-point arithmetic in PTX; kernel control flow and `expf` remain CUDA
C++. Their kernels are defined separately in
[mhc_post_ptx.cu](../../calculator/mhc/mhc_post_ptx.cu) and
[mhc_pre_ptx.cu](../../calculator/mhc/mhc_pre_ptx.cu); the shared arithmetic
helpers are in [mhc_common.h](../../calculator/mhc/mhc_common.h).

Requirements:

- Bazel, a C++ compiler, GTest, and the CUDA Toolkit at `/usr/local/cuda`.
- An NVIDIA GPU with compute capability 8.0 or newer and a compatible driver.
- A Python environment with CUDA-enabled PyTorch for the Python test.

Run the following commands from the repository root. `bazel test` builds the
necessary targets before running the tests.

### Python UT

First activate a Python environment containing CUDA-enabled PyTorch. Bazel
uses `python3` from the test's `PATH`:

```bash
cd /mnt/c/Users/xincu/python-env
source .venv/bin/activate
cd /mnt/c/Users/xincu/cuInfer-llm-pipeline
```

Build and run both Python tests:

```bash
bazel test //tests/mhc:mhc_python_test --test_env=PATH --nocache_test_results --test_output=all
```

### CUDA C++ UT

Build and run the standard CUDA C++ kernel tests:

```bash
bazel test //tests/mhc:mhc_cpp_test --test_arg='--gtest_filter=MhcPostTest.*:MhcPostArgumentsTest.*:MhcPreTest.*:MhcPreArgumentsTest.*' --nocache_test_results --test_output=all
```

### CUDA PTX UT

Build and run the inline PTX kernel tests:

```bash
bazel test //tests/mhc:mhc_cpp_test --test_arg='--gtest_filter=MhcPostPtxTest.*:MhcPrePtxTest.*' --nocache_test_results --test_output=all
```

To build and run each PTX test separately:

```bash
bazel test //tests/mhc:mhc_pre_big_fuse_cpp_test --test_arg='--gtest_filter=MhcPrePtxTest.*' --nocache_test_results --test_output=all
bazel test //tests/mhc:mhc_post_cpp_test --test_arg='--gtest_filter=MhcPostPtxTest.*' --nocache_test_results --test_output=all
```

The CUDA C++ and PTX cases share the same C++ test binaries; the GTest filters
select the desired variant. Without a filter, `mhc_cpp_test` runs both. Each
suite covers post and pre big-fuse. Individual files and Bazel targets are:

| Operator | C++ test | Python test |
| --- | --- | --- |
| Post | [mhc_post_test.cc](mhc_post_test.cc) (`//tests/mhc:mhc_post_cpp_test`) | [test_mhc_post.py](test_mhc_post.py) (`//tests/mhc:mhc_post_python_test`) |
| Pre big-fuse | [mhc_pre_big_fuse_test.cc](mhc_pre_big_fuse_test.cc) (`//tests/mhc:mhc_pre_big_fuse_cpp_test`) | [test_mhc_pre_big_fuse.py](test_mhc_pre_big_fuse.py) (`//tests/mhc:mhc_pre_big_fuse_python_test`) |

For example, to run only the C++ post test (both CUDA C++ and PTX cases):

```bash
bazel test //tests/mhc:mhc_post_cpp_test --nocache_test_results --test_output=errors
```

Outside Bazel, build `//calculator/mhc:build_mhc` and run the Python tests
directly with the activated environment:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" python tests/mhc/test_mhc_post.py
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" python tests/mhc/test_mhc_pre_big_fuse.py
```

The Python wrapper loads `bazel-bin/calculator/mhc/libmhc.so` by default;
`CUINFER_MHC_LIBRARY` can override its path. The operators use the current
PyTorch CUDA stream and do not implement autograd.

## Performance benchmark

[mhc_benchmark.cc](mhc_benchmark.cc) is a performance workload for comparing
the CUDA C++ and inline PTX kernels with Nsight Systems. It initializes inputs,
warms up the selected kernel, and marks the repeated launches to profile with
`cudaProfilerStart` and `cudaProfilerStop`. It does not check output
correctness; use the C++ and Python tests above for that.

Build and run it from the repository root:

```bash
bazel build //tests/mhc:mhc_benchmark
bazel-bin/tests/mhc/mhc_benchmark post cpp 128 4096 200 1000
```

The arguments are `<post|pre|prenorm> <cpp|ptx> <tokens> <hidden> <warmup>
<iterations>`. `prenorm` computes the square sum inside the pre kernel;
`pre` supplies it as an input. The numeric arguments must be positive
integers; `tokens` is limited to 65535 and `hidden` to 1048576.

The program prints the workload parameters and `logical_bytes`, the unique
logical input/output tensor bytes for one kernel launch. Running it alone does
not report latency or bandwidth: use Nsight Systems as shown below to obtain
the average kernel duration, then divide `logical_bytes` by that duration to
calculate effective bandwidth.

## Nsight Systems performance

Measured on 2026-11-15 with Nsight Systems 2026.1.3, CUDA 13.3, and an NVIDIA
RTX 500 Ada Generation Laptop GPU (SM 8.9, driver 621.14), under WSL2 Ubuntu
24.04. Each result is one trace of 1,000 GPU kernel executions after 200 warmup
executions. Latency is the mean GPU kernel duration reported by
`cuda_gpu_kern_sum`, excluding CPU launch and allocation time. Clocks were not
locked; these results are illustrative, not a statistically established speedup.
They were collected before the PTX kernels were split into separate source
files; rerun the benchmark below to measure the current build.

All cases use HC=4, BF16 residuals, FP32 mix inputs, H=4096, and the same inputs
for both variants. Pre uses one split, K=4H, 20 Sinkhorn iterations, epsilon
values of 1e-6 and the default block size. `prenorm` computes the square sum
inside the kernel; `pre` takes it as input.

| Operator | Tokens | CUDA C++ latency (us) | CUDA C++ effective bandwidth (GB/s) | Inline PTX latency (us) | Inline PTX effective bandwidth (GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post | 1 | 1.546 | 47.75 | 1.583 | 46.64 |
| Pre | 1 | 27.811 | 1.48 | 27.156 | 1.52 |
| Pre with prenorm | 1 | 42.533 | 0.97 | 43.022 | 0.96 |
| Post | 128 | 45.027 | 209.82 | 45.067 | 209.63 |
| Pre | 128 | 31.367 | 167.89 | 31.278 | 168.36 |
| Pre with prenorm | 128 | 49.563 | 106.24 | 49.134 | 107.17 |

**Bandwidth is calculated, not a hardware-counter measurement.** Nsight
Systems reported insufficient privileges (`ERR_NVGPUCTRPERM`) for GPU metrics
in this environment, so actual DRAM bandwidth is unavailable. Effective
bandwidth is unique logical input/output tensor bytes per launch divided by
mean kernel duration, using decimal GB/s; caching and repeated accesses are not
counted. With T tokens and hidden size H, the byte counts are:

- Post: `T * (18 * H + 80)`.
- Pre with external square sum: `T * (10 * H + 180) + 108`.
- Pre with prenorm: `T * (10 * H + 176) + 108`.

The shared scale and base tensors account for the final 108 bytes in pre;
prenorm reads the residual more than once, which this logical-byte metric does
not reflect.

To reproduce the traces and obtain the average kernel duration:

```bash
bazel build //tests/mhc:mhc_benchmark
report_dir="$(mktemp -d)"
for tokens in 1 128; do
  for op in post pre prenorm; do
    for variant in cpp ptx; do
      report="$report_dir/${op}_${variant}_${tokens}"
      nsys profile --trace=cuda --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="$report" \
        bazel-bin/tests/mhc/mhc_benchmark "$op" "$variant" "$tokens" 4096 200 1000
      nsys stats --report cuda_gpu_kern_sum "$report.nsys-rep"
    done
  done
done
```

The benchmark prints `logical_bytes`; divide it by the report's `Avg (ns)` to
obtain effective GB/s. Its input tensors and warmups are prepared before the
Nsight capture range.
